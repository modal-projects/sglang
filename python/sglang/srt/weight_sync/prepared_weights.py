"""Rank-local CPU images for full-model weight updates.

The image stores the final byte layout of every model-owned CUDA weight
storage. Background preparation produces a complete image; the paused update
only copies it into the existing CUDA storages. This preserves tensor aliases
and CUDA graph addresses and is independent of the checkpoint or delta format.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_ALIGNMENT = 4096
_COPY_CHUNK_BYTES = 256 << 20


class PartialWeightUpdateError(RuntimeError):
    """A CPU-to-GPU update may have changed only part of the live model."""


@dataclass(frozen=True)
class PreparedWeightSegment:
    """One unique live CUDA storage and its range in the host image."""

    name: str
    image_offset: int
    nbytes: int
    device_bytes: torch.Tensor


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _storage_as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage,
        0,
        (storage.nbytes(),),
        (1,),
    )


def iter_weight_tensors(
    model: torch.nn.Module,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield registered weights plus explicitly declared derived weights.

    Parameters and persistent buffers are checkpoint state. Non-persistent
    buffers are runtime caches or other architecture state and must remain
    untouched by a weight reload. Some kernels retain checkpoint-derived
    tensors as ordinary attributes. A module exposes those through
    ``prepared_weight_tensors()`` rather than making this code guess which
    arbitrary CUDA attributes are safe to overwrite.
    """

    yield from model.named_parameters(remove_duplicate=False)
    for module_name, module in model.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for name, tensor in module._buffers.items():
            if tensor is not None and name not in module._non_persistent_buffers_set:
                yield f"{prefix}{name}", tensor
    for module_name, module in model.named_modules():
        get_extra = getattr(module, "prepared_weight_tensors", None)
        if get_extra is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        for name, tensor in get_extra():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "prepared_weight_tensors() must yield (str, torch.Tensor)"
                )
            yield f"{prefix}{name}", tensor


def build_prepared_weight_plan(
    model: torch.nn.Module,
    *,
    device_type: str = "cuda",
) -> tuple[list[PreparedWeightSegment], int]:
    """Inventory unique model weight storages without changing their address."""

    unique: dict[tuple[int | None, int, int], tuple[str, torch.Tensor]] = {}
    for name, tensor in iter_weight_tensors(model):
        if tensor.device.type != device_type:
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
        current = unique.get(key)
        if current is None or name < current[0]:
            unique[key] = (name, tensor)

    offset = 0
    segments: list[PreparedWeightSegment] = []
    for name, tensor in sorted(unique.values(), key=lambda item: item[0]):
        offset = _align_up(offset)
        device_bytes = _storage_as_bytes(tensor)
        segments.append(
            PreparedWeightSegment(
                name=name,
                image_offset=offset,
                nbytes=device_bytes.numel(),
                device_bytes=device_bytes,
            )
        )
        offset += device_bytes.numel()
    if not segments:
        raise RuntimeError(f"model has no {device_type} weight storage")
    return segments, _align_up(offset)


class PreparedWeightImage:
    """Own one rank-local CPU image and commit it to live CUDA weights."""

    def __init__(self, model: torch.nn.Module):
        self.segments, self.image_nbytes = build_prepared_weight_plan(model)
        segments_by_storage = {
            (
                segment.device_bytes.device.index,
                segment.device_bytes.data_ptr(),
                segment.nbytes,
            ): segment
            for segment in self.segments
        }
        self._segments_by_device_storage = segments_by_storage
        self.segments_by_name: dict[str, PreparedWeightSegment] = {}
        for name, tensor in iter_weight_tensors(model):
            if tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
            self.segments_by_name[name] = segments_by_storage[key]
        self.image = torch.empty(self.image_nbytes, dtype=torch.uint8)
        self._image_buffer = memoryview(self.image.numpy()).cast("B")
        self._stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.registered = False
        self.registration_wall_s: float | None = None
        self.identity: str | None = None
        self.prepared = False
        self.preparing = False
        self.valid = False
        self.invalid_reason: str | None = "image has not been initialized"

    def storage_image_bytes(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return an isolated CPU storage view for one live weight storage.

        A normal image slice retains the full image storage, which would make
        unrelated weights appear aliased to module-cloning code. ``frombuffer``
        gives each segment its own storage identity while writing into the same
        registered image pages.
        """

        storage = tensor.untyped_storage()
        key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
        segment = self._segments_by_device_storage.get(key)
        if segment is None:
            raise KeyError("tensor storage is not part of the prepared image")
        begin = segment.image_offset
        end = begin + segment.nbytes
        return torch.frombuffer(
            self._image_buffer[begin:end],
            dtype=torch.uint8,
        )

    def capture_active(self, identity: str) -> dict[str, int | float | str]:
        """Seed the image from the active model while generation may continue."""

        self.invalidate(f"capture of {identity!r} is incomplete")
        started = time.perf_counter()
        try:
            self.copy_device_segments_to_image(
                (segment, segment.device_bytes) for segment in self.segments
            )
        except Exception as exc:
            self.invalid_reason = (
                f"capture of {identity!r} failed: {type(exc).__name__}: {exc}"
            )
            raise

        self.identity = identity
        self.prepared = False
        self.preparing = False
        self.valid = True
        self.invalid_reason = None
        return self.stats("capture", time.perf_counter() - started)

    def begin_preparation(self, identity: str) -> torch.Tensor:
        """Mark the single image unsafe until a full-target preparer finishes."""

        if self.preparing:
            raise RuntimeError("a host weight image preparation is already running")
        self.invalidate(f"preparation of {identity!r} is incomplete")
        self.identity = identity
        self.preparing = True
        return self.image

    def finish_preparation(self, identity: str) -> None:
        if not self.preparing:
            raise RuntimeError("no host weight image preparation is running")
        if self.identity != identity:
            raise RuntimeError(
                "prepared identity changed during preparation: "
                f"expected={identity!r}, actual={self.identity!r}"
            )
        self.valid = True
        self.prepared = True
        self.preparing = False
        self.invalid_reason = None

    def accept_prepared_baseline(self) -> None:
        """Keep a fully compiled baseline image without exposing a commit."""

        if not self.valid or not self.prepared or self.preparing:
            raise RuntimeError("no complete prepared image can become the baseline")
        self.identity = ""
        self.prepared = False
        self.invalid_reason = None

    def invalidate(self, reason: str) -> None:
        """Fail closed after a partial or otherwise untrusted preparation."""

        self.valid = False
        self.prepared = False
        self.preparing = False
        self.invalid_reason = reason

    def _chunks(
        self,
        segment: PreparedWeightSegment,
        device_bytes: torch.Tensor,
    ):
        if device_bytes.device.type != "cuda":
            raise ValueError(
                f"prepared source for {segment.name!r} is not a CUDA tensor"
            )
        if device_bytes.dtype != torch.uint8 or device_bytes.ndim != 1:
            raise ValueError(
                f"prepared source for {segment.name!r} is not a flat byte view"
            )
        if device_bytes.numel() != segment.nbytes:
            raise ValueError(
                "prepared source size mismatch: "
                f"name={segment.name!r} source={device_bytes.numel()} "
                f"target={segment.nbytes}"
            )
        window_nbytes = _COPY_CHUNK_BYTES
        for offset in range(0, segment.nbytes, window_nbytes):
            nbytes = min(window_nbytes, segment.nbytes - offset)
            yield (
                self.image[
                    segment.image_offset
                    + offset : segment.image_offset
                    + offset
                    + nbytes
                ],
                device_bytes[offset : offset + nbytes],
            )

    def register_host_memory(self) -> dict[str, int | float | str]:
        """Pin the persistent image once, outside the inference pause."""

        if self.registered:
            return self.stats("register_reused", 0.0)
        started = time.perf_counter()
        error = torch.cuda.cudart().cudaHostRegister(
            self.image.data_ptr(),
            self.image_nbytes,
            0,
        )
        if int(error) != 0:
            self.invalidate(f"cudaHostRegister failed: {error}")
            raise RuntimeError(f"cudaHostRegister failed: {error}")
        self.registered = True
        self.registration_wall_s = time.perf_counter() - started
        return self.stats("register", self.registration_wall_s)

    def copy_device_segments_to_image(
        self,
        segments: Iterable[tuple[PreparedWeightSegment, torch.Tensor]],
    ) -> None:
        """Copy device segments into the full host image."""

        if not self.registered:
            raise RuntimeError("host weight image must be registered before copying")
        with torch.cuda.stream(self._stream):
            for segment, device_bytes in segments:
                for image_chunk, device_chunk in self._chunks(
                    segment,
                    device_bytes,
                ):
                    image_chunk.copy_(device_chunk, non_blocking=True)
        self._stream.synchronize()

    def validate_commit(self, expected_identity: str) -> None:
        """Validate a requested commit without changing live weights."""
        if not self.valid:
            raise RuntimeError(
                "prepared weight image is invalid"
                + (f": {self.invalid_reason}" if self.invalid_reason else "")
            )
        if not self.prepared:
            raise RuntimeError("no host weight image is prepared")
        if self.identity != expected_identity:
            raise RuntimeError(
                "prepared weight identity mismatch: "
                f"expected={expected_identity!r}, prepared={self.identity!r}"
            )
        if not self.registered:
            raise RuntimeError(
                "prepared host image is not registered; registration must "
                "finish before the engine pauses"
            )

    def commit(self, expected_identity: str) -> dict[str, int | float | str]:
        """Overwrite every live weight storage from the prepared host image."""

        self.validate_commit(expected_identity)
        started = time.perf_counter()
        try:
            with torch.cuda.stream(self._stream):
                for segment in self.segments:
                    for image_chunk, device_chunk in self._chunks(
                        segment,
                        segment.device_bytes,
                    ):
                        device_chunk.copy_(
                            image_chunk,
                            non_blocking=True,
                        )
            self._stream.synchronize()
        except Exception as exc:
            self.invalidate(
                f"commit of {expected_identity!r} failed after live weights may "
                f"have been partially overwritten: {type(exc).__name__}: {exc}"
            )
            raise PartialWeightUpdateError(self.invalid_reason) from exc

        self.prepared = False
        return self.stats("commit", time.perf_counter() - started)

    def validate_against_active(self) -> dict[str, int | float | str]:
        """Validate the compiled image against the active startup weights."""

        if not self.valid or not self.prepared or not self.registered:
            raise RuntimeError("no complete CPU weight image is available")
        started = time.perf_counter()
        scratch = torch.empty(
            min(_COPY_CHUNK_BYTES, self.image_nbytes),
            dtype=torch.uint8,
            device=self.segments[0].device_bytes.device,
        )
        compared_bytes = 0
        for segment in self.segments:
            for image_chunk, device_chunk in self._chunks(
                segment, segment.device_bytes
            ):
                nbytes = image_chunk.numel()
                scratch[:nbytes].copy_(image_chunk)
                if not torch.equal(scratch[:nbytes], device_chunk):
                    raise RuntimeError(
                        "CPU weight cache does not reproduce the active model: "
                        f"storage={segment.name!r}"
                    )
                compared_bytes += nbytes
        return {
            "operation": "validate",
            "bytes": compared_bytes,
            "storages": len(self.segments),
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def stats(
        self,
        operation: str,
        wall_s: float,
    ) -> dict[str, int | float | str]:
        return {
            "operation": operation,
            "identity": self.identity or "",
            "bytes": self.image_nbytes,
            "storages": len(self.segments),
            "registered": self.registered,
            "registration_wall_s": round(self.registration_wall_s or 0.0, 6),
            "wall_s": round(wall_s, 6),
            "gbps": round(self.image_nbytes / max(wall_s, 1e-9) / 1e9, 3),
        }
