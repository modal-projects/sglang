import dataclasses
import logging
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    HostTensorAllocator,
    _cuda_host_unregister,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def encode_tensor_b64(tensor: torch.Tensor) -> str:
    """Zero-copy base64 encoding of a CPU tensor's raw bytes."""
    if tensor.numel() == 0:
        return ""
    return pybase64.b64encode_as_string(
        memoryview(tensor.contiguous().numpy()).cast("B")
    )


class BaseDeviceCache:
    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
        dtype: torch.dtype = torch.int32,
    ):
        self.buffer = torch.zeros(
            (max_batch_size, num_layers, topk_size),
            dtype=dtype,
            device=device,
        )
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        batch = topk_indices.shape[0]
        self.buffer[:batch, layer_id, :] = topk_indices

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"DeviceCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"dtype={self.buffer.dtype}, size={size_mb:.2f} MB"
        )


class BaseHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_layers: int,
        topk_size: int,
        name: str,
        device: str,
        dtype: torch.dtype = torch.int32,
    ):
        alloc = ALLOC_MEMORY_FUNCS[device]
        self.buffer = alloc(
            (num_tokens, num_layers, topk_size),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
            allocator=HostTensorAllocator(),
        )
        self.buffer.zero_()
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()

    def destroy(self):
        if self.buffer is None:
            return
        if _is_cuda or _is_hip:
            _cuda_host_unregister(self.buffer)
        self.buffer = None

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_gb = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"HostCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"dtype={self.buffer.dtype}, size={size_gb:.2f} GB"
        )


@dataclasses.dataclass
class TopkCaptureOutput:
    """Holds GPU tensors captured during forward for overlap scheduling.
    map_device_tensors() D2H-copies them before copy_done.record() (may run on
    the dedicated result-copy stream); finalize() runs after copy_done.synchronize().
    """

    out_cache_loc: torch.Tensor
    topk: torch.Tensor
    host_cache: BaseHostCache

    def map_device_tensors(self, fn):
        # Device-tensor fields only; caller injects the copy+safety primitive
        # (see GenerationBatchResult.copy_to_cpu).
        self.out_cache_loc = fn(self.out_cache_loc)
        self.topk = fn(self.topk)

    def finalize(self):
        self.host_cache.buffer[self.out_cache_loc] = self.topk


class BaseTopkCapturer:
    def __init__(
        self,
        num_tokens: int,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
        device_topk_size: Optional[int] = None,
        dtype: torch.dtype = torch.int32,
        host_dtype: Optional[torch.dtype] = None,
        device_dtype: Optional[torch.dtype] = None,
    ):
        """device_topk_size defaults to topk_size; pass a different value when
        the device buffer needs extra columns (e.g. fused shared experts) that
        are dropped before writing to host_cache via [:topk_size] truncation.
        """
        self.num_layers = num_layers
        self.topk_size = topk_size
        resolved_host_dtype = dtype if host_dtype is None else host_dtype
        resolved_device_dtype = dtype if device_dtype is None else device_dtype

        self.host_cache = BaseHostCache(
            num_tokens,
            num_layers,
            topk_size,
            name=name,
            device=device,
            dtype=resolved_host_dtype,
        )
        self.device_cache = BaseDeviceCache(
            max_batch_size,
            num_layers,
            device_topk_size if device_topk_size is not None else topk_size,
            device,
            name=name,
            dtype=resolved_device_dtype,
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        self.device_cache.capture(layer_id, topk_indices)

    def destroy(self):
        self.host_cache.destroy()

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        """Return the device_cache slice for this forward batch, GPU-resident.

        Default assumes per-rank-local capture: each rank writes [:local_num_tokens)
        to its own device_cache. Subclasses with global-tensor capture semantics
        (e.g. shared cuda graph buffer indexed by dp_rank) should override and
        consume can_run_graph / cuda_graph_batch.
        """
        del can_run_graph, cuda_graph_batch  # reserved for subclass override
        num_tokens = forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[:num_tokens, :, : self.topk_size]

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        end_len = max(0, seqlen - 1)
        start_len = min(start_len, end_len)
        buf = self.host_cache.buffer
        target_dtype = buf.dtype if dtype is None else dtype
        if target_dtype == torch.uint8 and buf.dtype != torch.uint8:
            target_dtype = buf.dtype
        num_rows = end_len - start_len
        if num_rows == 0:
            return torch.empty(
                (0, self.num_layers, self.topk_size),
                dtype=target_dtype,
                device="cpu",
            )

        idx = req_to_token_pool.req_to_token[req_pool_idx][start_len:end_len]
        if idx.device.type != "cpu":
            idx = idx.cpu()
        idx_np = idx.numpy()

        if num_rows == 1:
            start = int(idx_np[0])
            out = buf[start : start + 1]
            return out.to(target_dtype) if out.dtype != target_dtype else out.clone()

        breaks = np.flatnonzero(idx_np[1:] != idx_np[:-1] + 1) + 1
        if len(breaks) == 0:
            start = int(idx_np[0])
            out = buf[start : start + num_rows]
            return out.to(target_dtype) if out.dtype != target_dtype else out.clone()

        if len(breaks) < num_rows // 8:
            out = torch.empty(
                (num_rows, self.num_layers, self.topk_size),
                dtype=target_dtype,
                device="cpu",
            )
            starts = np.empty(len(breaks) + 1, dtype=np.intp)
            starts[0] = 0
            starts[1:] = breaks
            ends = np.empty(len(breaks) + 1, dtype=np.intp)
            ends[:-1] = breaks
            ends[-1] = num_rows
            for s, e in zip(starts.tolist(), ends.tolist()):
                src_start = int(idx_np[s])
                out[s:e].copy_(buf[src_start : src_start + (e - s)])
            return out

        out = torch.index_select(buf, 0, idx.to(torch.int64))
        return out.to(target_dtype) if out.dtype != target_dtype else out

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        """If no_copy_to_cpu is True, return a TopkCaptureOutput holding GPU tensors so
        the overlap thread can do non-blocking D2H + finalize itself. Otherwise sync
        D2H inline and return None (legacy non-overlap path).
        """
        slice_gpu = self._get_local_slice(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        # get_topk reads req_to_token IDs; attention may have rebound the write
        # loc to physical IDs, which are neither stable nor the same space.
        out_cache_loc = forward_batch.out_cache_loc_virtual
        if out_cache_loc is None:
            out_cache_loc = forward_batch.out_cache_loc
        else:
            # Kernel batches may be padded; only real request tokens are stored.
            slice_gpu = slice_gpu[: out_cache_loc.shape[0]]
        casted = slice_gpu.dtype != self.host_cache.buffer.dtype
        if casted:
            slice_gpu = slice_gpu.to(self.host_cache.buffer.dtype)
        if no_copy_to_cpu:
            # Clone before the next overlapping forward reuses these buffers.
            return TopkCaptureOutput(
                out_cache_loc=out_cache_loc.clone(),
                topk=slice_gpu if casted else slice_gpu.clone(),
                host_cache=self.host_cache,
            )
        out_cache_loc_cpu = out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = slice_gpu.cpu()
        return None
