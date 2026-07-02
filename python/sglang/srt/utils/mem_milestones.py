"""Env-gated GPU memory attribution tooling (zero-cost when off).

Two independent gates:

SGLANG_MEM_MILESTONES=1
    Log one line per named init stage with (device used, torch reserved,
    torch allocated, dark = used - reserved, free) in GB. "Dark" memory is
    everything on the device that the torch caching allocator does not know
    about: CUDA context, NCCL channel/registration buffers, symmetric memory,
    cudaMalloc calls from libraries (cublas/flashinfer/tokenspeed workspaces),
    and CUDA graph private pools.

SGLANG_TORCH_MEM_HISTORY=1
    Start torch.cuda.memory._record_memory_history() as early as possible in
    every model-runner process so that every live caching-allocator block
    carries its allocating Python stack, attach an out-of-memory observer that
    auto-dumps a snapshot on OOM, and dump pickled snapshots at named stages to
    SGLANG_TORCH_MEM_HISTORY_DIR. Snapshots are readable with
    torch.cuda._memory_viz or a plain pickle.load.

When SGLANG_TORCH_MEM_HISTORY=1 the profiler "MEM" activity keeps recording
alive across /start_profile//stop_profile cycles (it only dumps), so
boot-time allocation stacks are preserved in post-warmup snapshots.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GB = 1 << 30
_history_started = False
_oom_observer_attached = False
_lock = threading.Lock()


def mem_milestones_enabled() -> bool:
    return envs.SGLANG_MEM_MILESTONES.get()


def mem_history_active() -> bool:
    """True if boot-time allocator-history recording owns the recorder."""
    return _history_started


def log_mem_milestone(tag: str, gpu_id: int | None = None, tp_rank: int = 0) -> None:
    """Log one attribution line for the current device. No-op unless
    SGLANG_MEM_MILESTONES=1. Never raises."""
    if not mem_milestones_enabled():
        return
    try:
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device() if gpu_id is None else gpu_id
        free, total = torch.cuda.mem_get_info(dev)
        used = total - free
        reserved = torch.cuda.memory_reserved(dev)
        allocated = torch.cuda.memory_allocated(dev)
        logger.info(
            "[mem-milestone] TP%d %s: used=%.3f reserved=%.3f allocated=%.3f "
            "dark=%.3f free=%.3f total=%.3f (GB)",
            tp_rank,
            tag,
            used / _GB,
            reserved / _GB,
            allocated / _GB,
            (used - reserved) / _GB,
            free / _GB,
            total / _GB,
        )
    except Exception as e:  # never break serving for instrumentation
        logger.warning("[mem-milestone] %s failed: %s", tag, e)


def _snapshot_dir() -> str:
    return envs.SGLANG_TORCH_MEM_HISTORY_DIR.get()


def _oom_observer(device, alloc, device_allocated, device_free):
    # Called by the caching allocator right before raising CUDA OOM.
    try:
        out_dir = _snapshot_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, f"oom-dev{device}-{int(time.time())}.pickle"
        )
        logger.error(
            "[mem-history] OOM observed (device=%s alloc=%s device_allocated=%s "
            "device_free=%s); dumping allocator snapshot to %s",
            device,
            alloc,
            device_allocated,
            device_free,
            path,
        )
        torch.cuda.memory._dump_snapshot(path)
    except Exception as e:
        logger.error("[mem-history] OOM snapshot dump failed: %s", e)


def maybe_start_torch_mem_history() -> None:
    """Start allocator-history recording + OOM observer. Idempotent; no-op
    unless SGLANG_TORCH_MEM_HISTORY=1. Call as early as possible in each
    process that owns a GPU (before weight load)."""
    global _history_started, _oom_observer_attached
    if not envs.SGLANG_TORCH_MEM_HISTORY.get():
        return
    with _lock:
        if _history_started:
            return
        try:
            torch.cuda.memory._record_memory_history(
                max_entries=envs.SGLANG_TORCH_MEM_HISTORY_MAX_ENTRIES.get()
            )
            _history_started = True
            logger.info(
                "[mem-history] allocator history recording started "
                "(max_entries=%d, dir=%s)",
                envs.SGLANG_TORCH_MEM_HISTORY_MAX_ENTRIES.get(),
                _snapshot_dir(),
            )
        except Exception as e:
            logger.warning("[mem-history] failed to start recording: %s", e)
            return
        if not _oom_observer_attached:
            try:
                torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
                _oom_observer_attached = True
            except Exception as e:
                logger.warning("[mem-history] OOM observer not attached: %s", e)


def maybe_dump_mem_snapshot(tag: str, tp_rank: int = 0) -> None:
    """Dump a pickled allocator snapshot. No-op unless recording is active."""
    if not _history_started:
        return
    try:
        out_dir = _snapshot_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"memsnap-TP{tp_rank}-{tag}.pickle")
        torch.cuda.memory._dump_snapshot(path)
        logger.info("[mem-history] snapshot dumped: %s", path)
    except Exception as e:
        logger.warning("[mem-history] snapshot dump %s failed: %s", tag, e)
