"""A LoRA adapter's weights in slot order, placed by tracing the pool's install."""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple

import msgspec
import numpy as np
import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.lora.lora_prepare_process import strided
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_model

if TYPE_CHECKING:
    from sglang.srt.lora.lora_config import LoRAConfig

# (buffer dict attribute, module name, layer id or None for the non-layer buffers)
SlotBufferKey = Tuple[str, str, Optional[int]]


class TracedSlotWrite(NamedTuple):
    """An install write: its view in a slot row, and its weight (None: zeros)."""

    key: SlotBufferKey
    slot_numel: int
    offset: int
    strides: Tuple[int, ...]
    shape: Tuple[int, ...]
    weight: Optional[torch.Tensor]


class AdapterFileTensor(msgspec.Struct, frozen=True):
    name: str
    path: str
    shape: Tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def adapter_files(path: str, load_config: LoadConfig) -> Optional[List[str]]:
    """The safetensors files the default loader reads; None if others or no mmap."""
    if get_model().weight_loader_disable_mmap:
        return None
    _, files, use_safetensors = DefaultModelLoader(load_config)._prepare_weights(
        path, None, True
    )
    return files if use_safetensors else None


def slot_placement_key(config: LoRAConfig, inventory_digest: str) -> Tuple:
    targets = config.target_modules
    return (
        config.r,
        targets if isinstance(targets, str) else tuple(sorted(targets)),
        inventory_digest,
    )


class _SlotRegion(NamedTuple):
    key: SlotBufferKey
    offset: int
    numel: int


class _SlotCopy(NamedTuple):
    """One file tensor's elements placed by one install write, as two matching views."""

    tensor_index: int
    src_offset: int
    src_strides: Tuple[int, ...]
    dst_offset: int
    dst_strides: Tuple[int, ...]
    shape: Tuple[int, ...]
    scaled: bool


class LoRASlotPlacement(msgspec.Struct, frozen=True):
    """Where this rank's slot takes each adapter element from, per signature."""

    dtype: torch.dtype
    regions: Tuple[_SlotRegion, ...]
    copies: Tuple[_SlotCopy, ...]
    numel: int
    embedding_layer_names: Tuple[str, ...]


class SlotPlacementTracer:
    """Turns the install's writes on placeholder adapters into a LoRASlotPlacement.

    A placeholder element holds its file element index + 1 and the scaling is SCALING,
    so a written value names the file element it came from and whether it was scaled;
    zero is a written zero.
    """

    UID = "slot-placement"
    # A placeholder element holds its file index + 1: a written value names its source,
    # negated when scaled; zero is a written zero.
    SCALING = -1
    # Beyond this many copies for one write, the LoRAAdapter path is the faster one.
    MAX_COPIES_PER_WRITE = 1024

    def __init__(self, tensors: Sequence[AdapterFileTensor], num_layers: int):
        self.tensors = tensors
        self._starts = torch.tensor([0] + [t.numel for t in tensors]).cumsum(0)
        self._starts_np = self._starts.numpy()
        self._starts_list = self._starts.tolist()
        self._by_layer: Dict[Optional[int], List[int]] = {}
        for index, tensor in enumerate(tensors):
            self._by_layer.setdefault(get_layer_id(tensor.name), []).append(index)
        self.layer_ids: List[Optional[int]] = [None, *range(num_layers)]
        beyond = set(self._by_layer) - set(self.layer_ids)
        if beyond:
            raise ValueError(f"LoRA adapter has weights for layers {sorted(beyond)}")
        self._regions: Dict[SlotBufferKey, _SlotRegion] = {}
        self._copies: List[_SlotCopy] = []
        self._numel = 0

    def placeholders(self, layer_id: Optional[int]) -> Dict[str, torch.Tensor]:
        placeholders = {}
        indices = self._by_layer.get(layer_id, [])
        starts = self._starts_list
        for first, last in _consecutive_runs(indices):
            values = torch.arange(starts[first] + 1, starts[last + 1] + 1)
            for i in range(first, last + 1):
                begin, end = starts[i] - starts[first], starts[i + 1] - starts[first]
                placeholders[self.tensors[i].name] = values[begin:end].view(
                    self.tensors[i].shape
                )
        return placeholders

    def add(self, writes: Sequence[TracedSlotWrite], layer_id: Optional[int]) -> None:
        """Take the writes to `layer_id`'s buffers; overlapping writes raise."""
        copied: Dict[SlotBufferKey, np.ndarray] = {}
        for write in writes:
            if write.key[2] != layer_id:
                continue
            region = self._regions.get(write.key)
            if region is None:
                region = _SlotRegion(write.key, self._numel, write.slot_numel)
                self._regions[write.key] = region
                self._numel += write.slot_numel
            if write.key not in copied:
                copied[write.key] = np.zeros(write.slot_numel, dtype=np.bool_)
            written = np.lib.stride_tricks.as_strided(
                copied[write.key][write.offset :], write.shape, write.strides
            )
            if written.any():
                raise ValueError(f"LoRA install writes overlap in {write.key}")
            if write.weight is None or write.weight.numel() == 0:
                continue
            written[...] = True
            self._copies.extend(self._copies_of(write, region.offset))

    def placement(
        self, dtype: torch.dtype, embedding_layer_names: Tuple[str, ...]
    ) -> LoRASlotPlacement:
        return LoRASlotPlacement(
            dtype=dtype,
            regions=tuple(self._regions.values()),
            copies=tuple(self._copies),
            numel=self._numel,
            embedding_layer_names=embedding_layer_names,
        )

    def _copies_of(self, write: TracedSlotWrite, region_offset: int) -> List[_SlotCopy]:
        """A strided copy per block of rows of one file tensor, else element runs."""
        values = write.weight
        first_of_row = np.abs(values.numpy().reshape(values.shape[0], -1)[:, 0]) - 1
        tensor_of_row = (
            np.searchsorted(self._starts_np, first_of_row.clip(min=0), side="right") - 1
        )
        breaks = (np.flatnonzero(np.diff(tensor_of_row)) + 1).tolist()
        bounds = [0, *breaks, values.shape[0]]
        copies = []
        for start, end in zip(bounds, bounds[1:]):
            block = write._replace(
                offset=write.offset + start * write.strides[0],
                shape=(end - start, *write.shape[1:]),
                weight=values[start:end],
            )
            copy = self._strided_copy(block, region_offset)
            if copy is not None:
                copies.append(copy)
            else:
                copies.extend(self._element_copies(block, region_offset))
            if len(copies) > self.MAX_COPIES_PER_WRITE:
                raise ValueError(f"LoRA install write to {write.key} is not strided")
        return copies

    def _strided_copy(
        self, write: TracedSlotWrite, region_offset: int
    ) -> Optional[_SlotCopy]:
        """The write as one copy, when its values step through one file tensor at
        fixed non-negative strides."""
        values = write.weight.numpy()
        # All negative: scaled; all positive: not; zeros or both: not one copy.
        low, high = values.min(), values.max()
        if low > 0:
            scaled, src = False, values - 1
        elif high < 0:
            scaled, src = True, -values - 1
        else:
            return None
        origin = int(src.flat[0])
        strides = []
        for dim, size in enumerate(src.shape):
            step = [0] * src.ndim
            step[dim] = 1
            stride = int(src[tuple(step)]) - origin if size > 1 else 0
            if stride < 0:
                return None
            strides.append(stride)
        last = origin + sum((n - 1) * s for n, s in zip(src.shape, strides))
        span = np.arange(origin, last + 1, dtype=src.dtype)
        expected = np.lib.stride_tricks.as_strided(
            span, src.shape, [s * span.itemsize for s in strides]
        )
        if not np.array_equal(src, expected):
            return None
        tensor_index = bisect.bisect_right(self._starts_list, origin) - 1
        if last >= self._starts_list[tensor_index + 1]:
            return None
        return _SlotCopy(
            tensor_index=tensor_index,
            src_offset=origin - self._starts_list[tensor_index],
            src_strides=tuple(strides),
            dst_offset=region_offset + write.offset,
            dst_strides=write.strides,
            shape=write.shape,
            scaled=scaled,
        )

    def _element_copies(
        self, write: TracedSlotWrite, region_offset: int
    ) -> List[_SlotCopy]:
        """The write as runs of consecutive elements; written zeros need no copy."""
        index = _index(write.strides, write.shape)
        dst = write.offset + strided(index, 0, write.strides, write.shape).reshape(-1)
        values = write.weight.reshape(-1)
        dst, values = dst[values != 0], values[values != 0]
        if dst.numel() == 0:
            return []
        order = torch.argsort(dst)
        dst, values = dst[order], values[order]
        scaled = values < 0
        src = values.abs() - 1
        tensor = torch.searchsorted(self._starts, src, right=True) - 1
        breaks = torch.nonzero(
            (torch.diff(dst) != 1)
            | (torch.diff(src) != 1)
            | (torch.diff(scaled.int()) != 0)
            | (torch.diff(tensor) != 0)
        ).flatten()
        first = torch.cat([torch.zeros(1, dtype=torch.long), breaks + 1])
        length = torch.diff(torch.cat([first, torch.tensor([dst.numel()])]))
        return [
            _SlotCopy(
                tensor_index=int(tensor[f]),
                src_offset=int(src[f] - self._starts[tensor[f]]),
                src_strides=(1,),
                dst_offset=region_offset + int(dst[f]),
                dst_strides=(1,),
                shape=(n,),
                scaled=bool(scaled[f]),
            )
            for f, n in zip(first.tolist(), length.tolist())
        ]


class LoRASlotWeights:
    """An adapter's weights in this rank's slot order, a tensor per region."""

    def __init__(
        self,
        config: LoRAConfig,
        placement: LoRASlotPlacement,
        regions: Sequence[torch.Tensor],
    ):
        self.config = config
        self.scaling: float = config.lora_alpha / config.r
        self.placement = placement
        self.regions = list(regions)


def slot_regions(
    placement: LoRASlotPlacement, data: torch.Tensor
) -> List[torch.Tensor]:
    """Views of `data`, which holds a slot in order, one per region of `placement`."""
    return [data[r.offset : r.offset + r.numel] for r in placement.regions]


def _consecutive_runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """(first, last) of each run of consecutive ints in ascending `indices`."""
    runs: List[Tuple[int, int]] = []
    for i in indices:
        if runs and runs[-1][1] == i - 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def _index(strides: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
    """0..n-1 covering every element a view of `shape` and `strides` reaches."""
    return torch.arange(sum((n - 1) * s for n, s in zip(shape, strides)) + 1)
