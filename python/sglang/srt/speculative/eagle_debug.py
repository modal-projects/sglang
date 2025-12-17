"""
EAGLE3 V2 Debug Infrastructure

This module provides comprehensive debugging tools for EAGLE3 speculative decoding.
Designed for tracking stream synchronization issues and kv_indices manipulation.

═══════════════════════════════════════════════════════════════════════════════════
KEY INSIGHT (from exploration): The stream sync is about accept_index dependency!
═══════════════════════════════════════════════════════════════════════════════════

The root cause of Tree Mode's stream sync requirement:
- plan_stream needs verification results (accept_index)
- accept_index is computed by sampling in main_stream
- Whether we compact req_to_token or use sparse indices, we STILL need to wait

Chain Mode doesn't need sync because:
- It only needs the COUNT of accepted tokens, not WHICH positions
- req_to_token is never modified (append-only)

Environment Variables:
─────────────────────────────────────────────────────────────────────────────────
EAGLE3_DEBUG         : int  = 0      # Debug level (0=off, 1=events, 2=state, 3=tensors, 4=all)
EAGLE3_DEBUG_RUN_NAME: str  = ""     # Run identifier (default: timestamp)
EAGLE3_DEBUG_MAX_ITERS: int = 100    # Stop detailed logging after N iterations
EAGLE3_DEBUG_TENSOR_DUMP: bool = 0   # Dump tensors to files (expensive!)
EAGLE3_DEBUG_CHECKPOINTS: str = "all"  # Comma-separated: all, sync, compact, extend, verify
EAGLE3_DEBUG_REQ_FILTER: str = ""    # Only log for specific request indices (e.g., "0,1")
EAGLE3_DEBUG_STREAM_TRACE: bool = 0  # Trace stream sync points
EAGLE3_DEBUG_KV_TRACE: bool = 0      # Trace kv_indices/req_to_token reads/writes
EAGLE3_DEBUG_ACCEPT_TRACE: bool = 0  # Trace accept_index lifecycle (ROOT DEPENDENCY)
EAGLE3_DEBUG_DATA_MOVE_TRACE: bool = 0  # Trace KV data move operations (Data Move approach)
EAGLE3_DEBUG_SYNC_TIMING: bool = 0   # Measure actual sync overhead with CUDA events
─────────────────────────────────────────────────────────────────────────────────

Usage:
    from sglang.srt.speculative.eagle_debug import get_debug_state

    debug = get_debug_state()
    debug.checkpoint("VERIFY_START", stream="main", seq_lens=batch.seq_lens.tolist()[:4])
    debug.tensor_dump("req_to_token", req_to_token_slice)

    # NEW: Track accept_index dependency
    debug.accept_trace("COMPUTED", accept_index, accept_length)  # After sampling
    debug.accept_trace("CONSUMED", accept_index, accept_length)  # In compaction

    # NEW: Measure sync overhead
    debug.sync_start("SYNC_2")  # Before wait_stream
    ...
    debug.sync_end("SYNC_2")    # After wait_stream
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


@dataclass
class EagleDebugConfig:
    """Configuration for EAGLE3 debugging, read from environment variables."""

    # Debug level: 0=off, 1=events, 2=state, 3=tensors, 4=all
    level: int = 0

    # Run identification
    run_name: str = ""
    base_dir: Path = Path("/debug")

    # Iteration control
    max_iters: int = 100  # Stop detailed logging after N iterations

    # Feature flags
    tensor_dump: bool = False  # Dump tensors to files
    stream_trace: bool = False  # Trace stream sync points
    kv_trace: bool = False  # Trace kv_indices/req_to_token
    accept_trace: bool = False  # Trace accept_index lifecycle (ROOT DEPENDENCY)
    data_move_trace: bool = False  # Trace KV data move operations
    sync_timing: bool = False  # Measure actual sync overhead with CUDA events

    # Checkpoint filtering
    checkpoints: set = field(default_factory=lambda: {"all"})

    # Request filtering (empty = all requests)
    req_filter: set = field(default_factory=set)

    # Derived paths
    run_dir: Path = field(init=False)
    events_file: Path = field(init=False)
    tensors_dir: Path = field(init=False)
    config_file: Path = field(init=False)

    def __post_init__(self):
        # Set default run name if not provided
        if not self.run_name:
            self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up directory structure
        self.run_dir = self.base_dir / self.run_name
        self.events_file = self.run_dir / "events.jsonl"
        self.tensors_dir = self.run_dir / "tensors"
        self.config_file = self.run_dir / "config.json"

    @classmethod
    def from_env(cls) -> "EagleDebugConfig":
        """Create config from environment variables."""
        level = int(os.environ.get("EAGLE3_DEBUG", "0"))

        config = cls(
            level=level,
            run_name=os.environ.get("EAGLE3_DEBUG_RUN_NAME", ""),
            max_iters=int(os.environ.get("EAGLE3_DEBUG_MAX_ITERS", "100")),
            tensor_dump=os.environ.get("EAGLE3_DEBUG_TENSOR_DUMP", "0") == "1",
            stream_trace=os.environ.get("EAGLE3_DEBUG_STREAM_TRACE", "0") == "1",
            kv_trace=os.environ.get("EAGLE3_DEBUG_KV_TRACE", "0") == "1",
            accept_trace=os.environ.get("EAGLE3_DEBUG_ACCEPT_TRACE", "0") == "1",
            data_move_trace=os.environ.get("EAGLE3_DEBUG_DATA_MOVE_TRACE", "0") == "1",
            sync_timing=os.environ.get("EAGLE3_DEBUG_SYNC_TIMING", "0") == "1",
        )

        # Parse checkpoints
        checkpoints_str = os.environ.get("EAGLE3_DEBUG_CHECKPOINTS", "all")
        config.checkpoints = set(checkpoints_str.split(","))

        # Parse request filter
        req_filter_str = os.environ.get("EAGLE3_DEBUG_REQ_FILTER", "")
        if req_filter_str:
            config.req_filter = set(int(x) for x in req_filter_str.split(","))

        return config

    def is_enabled(self, level: int = 1) -> bool:
        """Check if debugging is enabled at the given level."""
        return self.level >= level

    def should_log_checkpoint(self, name: str) -> bool:
        """Check if a checkpoint should be logged."""
        if "all" in self.checkpoints:
            return True
        # Map checkpoint names to categories
        categories = {
            "DECODE_START": "all",
            "SYNC_1_BEFORE_DRAFT": "sync",
            "DRAFT_START": "draft",
            "DRAFT_DONE": "draft",
            "SYNC_2_BEFORE_VERIFY": "sync",
            "VERIFY_START": "verify",
            "VERIFY_SAMPLE_DONE": "verify",
            "COMPACT_START": "compact",
            "COMPACT_DONE": "compact",
            "VERIFY_DONE": "verify",
            "SYNC_3_BEFORE_EXTEND": "sync",
            "EXTEND_START": "extend",
            "EXTEND_DONE": "extend",
            "DECODE_DONE": "all",
        }
        category = categories.get(name, "all")
        return category in self.checkpoints or name in self.checkpoints

    def should_log_request(self, req_idx: int) -> bool:
        """Check if a request should be logged."""
        if not self.req_filter:
            return True
        return req_idx in self.req_filter

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level,
            "run_name": self.run_name,
            "base_dir": str(self.base_dir),
            "max_iters": self.max_iters,
            "tensor_dump": self.tensor_dump,
            "stream_trace": self.stream_trace,
            "kv_trace": self.kv_trace,
            "accept_trace": self.accept_trace,
            "sync_timing": self.sync_timing,
            "checkpoints": list(self.checkpoints),
            "req_filter": list(self.req_filter),
        }


class EagleDebugState:
    """
    Shared debug state for EAGLE3 V2 speculative decoding.

    This class manages:
    - Iteration counter (shared across EAGLEWorkerV2 and EagleDraftWorker)
    - Event logging to JSONL file
    - Tensor snapshots for deep debugging
    - Stream synchronization tracing
    - kv_indices/req_to_token read/write tracing

    Thread-safe: Uses locks for counter increments and file writes.
    """

    # Predefined checkpoint names for validation
    VALID_CHECKPOINTS = frozenset(
        [
            # Main decode loop
            "DECODE_START",
            "DECODE_DONE",
            # Stream syncs
            "SYNC_1_BEFORE_DRAFT",
            "SYNC_2_BEFORE_VERIFY",
            "SYNC_3_BEFORE_EXTEND",
            # Draft phase
            "DRAFT_START",
            "DRAFT_PREPARE_START",
            "DRAFT_PREPARE_DONE",
            "DRAFT_FORWARD_START",
            "DRAFT_FORWARD_DONE",
            "DRAFT_DONE",
            # Verify phase
            "VERIFY_START",
            "VERIFY_PREPARE_START",
            "VERIFY_PREPARE_DONE",
            "VERIFY_FORWARD_START",
            "VERIFY_FORWARD_DONE",
            "VERIFY_SAMPLE_DONE",
            "VERIFY_DONE",
            # Compaction phase
            "COMPACT_START",
            "COMPACT_BUILD_PERM",
            "COMPACT_GATHER_TENSORS",
            "COMPACT_REQ_TO_TOKEN",
            "COMPACT_DONE",
            # Draft extend phase
            "EXTEND_START",
            "EXTEND_PREPARE_START",
            "EXTEND_PREPARE_DONE",
            "EXTEND_FORWARD_START",
            "EXTEND_FORWARD_DONE",
            "EXTEND_DONE",
            # KV tracing
            "KV_READ_REQ_TO_TOKEN",
            "KV_WRITE_REQ_TO_TOKEN",
            "KV_READ_OUT_CACHE_LOC",
            "KV_WRITE_OUT_CACHE_LOC",
            # Custom/free-form
            "CUSTOM",
        ]
    )

    def __init__(self, config: Optional[EagleDebugConfig] = None):
        self.config = config or EagleDebugConfig.from_env()

        # Iteration tracking
        self._iter = 0
        self._iter_lock = threading.Lock()

        # File I/O
        self._file_lock = threading.Lock()
        self._events_file = None

        # Timing
        self._start_time = time.perf_counter()
        self._last_checkpoint_time = self._start_time

        # CUDA event timing for sync overhead measurement
        self._sync_events: dict = {}  # sync_id -> (start_event, end_event)
        self._sync_timings: list = []  # List of (sync_id, elapsed_us)

        # Initialize if enabled
        if self.config.is_enabled():
            self._initialize_output()

    def _initialize_output(self):
        """Initialize output directory and files."""
        try:
            # Create directories
            self.config.run_dir.mkdir(parents=True, exist_ok=True)
            self.config.tensors_dir.mkdir(parents=True, exist_ok=True)

            # Write config
            with open(self.config.config_file, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)

            # Open events file
            self._events_file = open(self.config.events_file, "a")

            # Log initialization
            self._log_event(
                "DEBUG_INIT",
                level=0,
                config=self.config.to_dict(),
                pid=os.getpid(),
            )
        except Exception as e:
            print(f"[EAGLE3_DEBUG] Failed to initialize: {e}")
            self.config.level = 0  # Disable debugging on error

    @property
    def iter(self) -> int:
        """Current iteration number."""
        return self._iter

    def next_iter(self) -> int:
        """Increment and return the new iteration number."""
        with self._iter_lock:
            self._iter += 1
            return self._iter

    def is_enabled(self, level: int = 1) -> bool:
        """Check if debugging is enabled at the given level."""
        return self.config.is_enabled(level)

    def should_log_iter(self) -> bool:
        """Check if current iteration should be logged (respects max_iters)."""
        return self._iter <= self.config.max_iters

    def _log_event(self, event: str, level: int = 1, **data):
        """Internal: Log an event to the events file."""
        if not self.config.is_enabled(level):
            return
        if not self._events_file:
            return

        now = time.perf_counter()
        entry = {
            "ts": now - self._start_time,
            "delta": now - self._last_checkpoint_time,
            "iter": self._iter,
            "event": event,
            **data,
        }
        self._last_checkpoint_time = now

        with self._file_lock:
            try:
                self._events_file.write(json.dumps(entry) + "\n")
                self._events_file.flush()
            except Exception as e:
                print(f"[EAGLE3_DEBUG] Log error: {e}")

    def checkpoint(
        self,
        name: str,
        stream: str = "main",
        level: int = 1,
        **data,
    ):
        """
        Log a checkpoint event.

        Args:
            name: Checkpoint name (should be in VALID_CHECKPOINTS or 'CUSTOM')
            stream: Stream context ('main' or 'plan')
            level: Minimum debug level to log (1=events, 2=state, 3=tensors)
            **data: Additional data to log (will be JSON serialized)
        """
        if not self.config.is_enabled(level):
            return
        if not self.should_log_iter():
            return
        if not self.config.should_log_checkpoint(name):
            return

        # Warn on unknown checkpoint names
        if name not in self.VALID_CHECKPOINTS and not name.startswith("CUSTOM_"):
            print(f"[EAGLE3_DEBUG] Warning: Unknown checkpoint '{name}'")

        # Convert tensors to lists for JSON serialization
        serialized_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                # Only serialize small tensors inline
                if v.numel() <= 16:
                    serialized_data[k] = v.tolist()
                else:
                    serialized_data[k] = f"<Tensor shape={list(v.shape)}>"
            else:
                serialized_data[k] = v

        self._log_event(
            name,
            level=level,
            stream=stream,
            **serialized_data,
        )

        # Also print to stdout for real-time debugging
        if self.config.level >= 2:
            data_str = ", ".join(f"{k}={v}" for k, v in serialized_data.items())
            print(f"[ITER {self._iter}][{stream}] {name}: {data_str}")

    def stream_sync(
        self,
        sync_id: int,
        wait_stream: str,
        signal_stream: str,
        reason: str = "",
    ):
        """
        Log a stream synchronization event.

        Args:
            sync_id: Unique identifier for this sync point (1, 2, or 3)
            wait_stream: Stream that is waiting ('plan' or 'main')
            signal_stream: Stream being waited on ('main' or 'plan')
            reason: Human-readable reason for the sync
        """
        if not self.config.stream_trace:
            return

        self.checkpoint(
            f"SYNC_{sync_id}_TRACE",
            stream=wait_stream,
            level=2,
            wait_stream=wait_stream,
            signal_stream=signal_stream,
            reason=reason,
        )

    def kv_trace(
        self,
        operation: str,  # "read" or "write"
        target: str,  # "req_to_token", "out_cache_loc", "kv_indices"
        req_idx: int,
        start_idx: int,
        end_idx: int,
        stream: str = "main",
        values: Optional[torch.Tensor] = None,
    ):
        """
        Log a KV-related tensor access.

        Args:
            operation: "read" or "write"
            target: Name of the tensor being accessed
            req_idx: Request index
            start_idx: Start of the slice
            end_idx: End of the slice
            stream: Stream context
            values: Optional tensor values (small slice for debugging)
        """
        if not self.config.kv_trace:
            return
        if not self.config.should_log_request(req_idx):
            return

        data = {
            "operation": operation,
            "target": target,
            "req_idx": req_idx,
            "slice": f"[{start_idx}:{end_idx}]",
        }

        if values is not None and values.numel() <= 16:
            data["values"] = values.tolist()

        event_name = f"KV_{operation.upper()}_{target.upper()}"
        self.checkpoint(event_name, stream=stream, level=3, **data)

    # =========================================================================
    # NEW: accept_index dependency tracing (ROOT CAUSE of stream syncs)
    # =========================================================================

    def accept_trace(
        self,
        operation: str,  # "COMPUTED" (after sampling) or "CONSUMED" (in compaction/kv_indices)
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
        stream: str = "main",
        context: str = "",
    ):
        """
        Trace accept_index lifecycle - THE ROOT DEPENDENCY for stream syncs.

        ═══════════════════════════════════════════════════════════════════════
        KEY INSIGHT: The stream sync is NOT about req_to_token writes.
        It's about plan_stream needing accept_index (computed in main_stream).
        ═══════════════════════════════════════════════════════════════════════

        Args:
            operation: "COMPUTED" (after sampling) or "CONSUMED" (in compaction)
            accept_index: The accept_index tensor [bs, max_accept]
            accept_length: The accept_length tensor [bs]
            stream: Stream context
            context: Additional context (e.g., "compaction", "kv_indices_build")
        """
        if not self.config.accept_trace:
            return
        if not self.should_log_iter():
            return

        # Extract key info without full sync
        bs = accept_length.shape[0]
        # Small sample of accept pattern (first request)
        accept_sample = (
            accept_index[0, :4].tolist()
            if accept_index.numel() >= 4
            else accept_index.flatten().tolist()
        )
        accept_lens = accept_length.tolist()

        # Calculate sparsity pattern (how scattered are acceptances?)
        # This is the key insight: Tree Mode has scattered, Chain has contiguous
        total_positions = (
            bs * accept_index.shape[1]
            if accept_index.dim() == 2
            else accept_index.shape[0]
        )

        self._log_event(
            f"ACCEPT_{operation}",
            level=2,
            stream=stream,
            context=context,
            bs=bs,
            accept_lens=accept_lens,
            accept_sample=accept_sample,
            total_positions=total_positions,
        )

        if self.config.level >= 2:
            print(
                f"[ITER {self._iter}][{stream}] ACCEPT_{operation}: "
                f"lens={accept_lens}, sample={accept_sample[:4]}, ctx={context}"
            )

    # =========================================================================
    # NEW: Data Move tracing (Zero-Sync Tree Mode approach)
    # =========================================================================

    def data_move_trace(
        self,
        operation: str,  # "BUILD_INDICES", "EXECUTE", "SKIP" (no moves needed)
        src_slots: Optional[torch.Tensor] = None,
        dst_slots: Optional[torch.Tensor] = None,
        num_moves: Optional[int] = None,
        stream: str = "main",
        context: str = "",
    ):
        """
        Trace KV cache data move operations.

        The Data Move approach eliminates stream syncs by:
        - Keeping req_to_token STATIC (CPU allocates once, never compacts)
        - Moving KV DATA from scattered slots to contiguous slots on GPU

        Args:
            operation: "BUILD_INDICES", "EXECUTE", "SKIP"
            src_slots: Source physical slot IDs (slots to read from)
            dst_slots: Destination physical slot IDs (slots to write to)
            num_moves: Number of actual moves (src != dst)
            stream: Stream context
            context: Additional context
        """
        if not self.config.data_move_trace:
            return
        if not self.should_log_iter():
            return

        data = {
            "operation": operation,
            "context": context,
        }

        if num_moves is not None:
            data["num_moves"] = num_moves

        if src_slots is not None and self.config.tensor_dump:
            # Only dump first few for debugging
            data["src_sample"] = src_slots[:8].tolist() if src_slots.numel() > 0 else []
            data["dst_sample"] = dst_slots[:8].tolist() if dst_slots.numel() > 0 else []

        self._log_event(
            f"DATA_MOVE_{operation}",
            level=2,
            stream=stream,
            **data,
        )

        if self.config.level >= 2:
            moves_str = f"moves={num_moves}" if num_moves is not None else ""
            print(
                f"[ITER {self._iter}][{stream}] DATA_MOVE_{operation}: {moves_str} {context}"
            )

    # =========================================================================
    # NEW: Sync overhead measurement with CUDA events
    # =========================================================================

    def sync_start(self, sync_id: str, device: str = "cuda"):
        """
        Record start of a stream sync for timing measurement.

        Args:
            sync_id: Unique identifier (e.g., "SYNC_2", "SYNC_3")
            device: CUDA device
        """
        if not self.config.sync_timing:
            return

        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._sync_events[sync_id] = (start_event, None)

    def sync_end(self, sync_id: str, device: str = "cuda"):
        """
        Record end of a stream sync and log elapsed time.

        Args:
            sync_id: Must match a previous sync_start call
            device: CUDA device
        """
        if not self.config.sync_timing:
            return
        if sync_id not in self._sync_events:
            return

        start_event, _ = self._sync_events[sync_id]
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()  # NOTE: This is a sync, but only when timing is enabled

        elapsed_us = start_event.elapsed_time(end_event) * 1000  # ms -> us

        self._sync_timings.append((self._iter, sync_id, elapsed_us))

        self._log_event(
            "SYNC_TIMING",
            level=2,
            sync_id=sync_id,
            elapsed_us=elapsed_us,
        )

        if self.config.level >= 2:
            print(f"[ITER {self._iter}] {sync_id} elapsed: {elapsed_us:.1f}µs")

        # Cleanup
        del self._sync_events[sync_id]

    def get_sync_summary(self) -> dict:
        """Get summary statistics for sync timings."""
        if not self._sync_timings:
            return {}

        from collections import defaultdict

        by_id = defaultdict(list)
        for _, sync_id, elapsed in self._sync_timings:
            by_id[sync_id].append(elapsed)

        summary = {}
        for sync_id, timings in by_id.items():
            summary[sync_id] = {
                "count": len(timings),
                "mean_us": sum(timings) / len(timings),
                "min_us": min(timings),
                "max_us": max(timings),
            }
        return summary

    def tensor_dump(
        self,
        name: str,
        tensor: torch.Tensor,
        checkpoint: str = "",
        req_idx: Optional[int] = None,
    ):
        """
        Dump a tensor to a file for offline analysis.

        WARNING: This introduces a GPU->CPU sync! Only use for deep debugging.

        Args:
            name: Name for the tensor file
            tensor: Tensor to dump
            checkpoint: Associated checkpoint name
            req_idx: Optional request index for filtering
        """
        if not self.config.tensor_dump:
            return
        if not self.should_log_iter():
            return
        if req_idx is not None and not self.config.should_log_request(req_idx):
            return

        filename = f"iter_{self._iter}_{checkpoint}_{name}.pt"
        filepath = self.config.tensors_dir / filename

        try:
            # WARNING: This clones to CPU, introducing a sync!
            torch.save(tensor.clone().cpu(), filepath)
            self.checkpoint(
                "TENSOR_DUMP",
                level=3,
                filename=filename,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
            )
        except Exception as e:
            print(f"[EAGLE3_DEBUG] Tensor dump error: {e}")

    def state_snapshot(
        self,
        checkpoint: str,
        batch,
        req_to_token_pool,
        accept_index: Optional[torch.Tensor] = None,
        accept_length: Optional[torch.Tensor] = None,
    ):
        """
        Capture a comprehensive state snapshot.

        WARNING: This introduces GPU->CPU syncs! Only use for deep debugging.

        Args:
            checkpoint: Checkpoint name for context
            batch: ModelWorkerBatch or ForwardBatch
            req_to_token_pool: The req_to_token pool
            accept_index: Optional accept_index tensor
            accept_length: Optional accept_length tensor
        """
        if not self.config.is_enabled(3):
            return
        if not self.should_log_iter():
            return

        snapshot = {
            "checkpoint": checkpoint,
            "iter": self._iter,
        }

        # Batch info (careful: some may be GPU tensors)
        try:
            if hasattr(batch, "seq_lens"):
                snapshot["seq_lens"] = batch.seq_lens.tolist()[:4]
            if hasattr(batch, "req_pool_indices"):
                snapshot["req_pool_indices"] = batch.req_pool_indices.tolist()[:4]
            if hasattr(batch, "out_cache_loc"):
                snapshot["out_cache_loc_shape"] = list(batch.out_cache_loc.shape)
        except Exception as e:
            snapshot["batch_error"] = str(e)

        # req_to_token window for first request
        try:
            if len(batch.req_pool_indices) > 0:
                req_idx = batch.req_pool_indices[0].item()
                seq_len = batch.seq_lens[0].item()
                tree_size = 32  # Approximate
                window_start = max(0, seq_len - tree_size)
                window = req_to_token_pool.req_to_token[
                    req_idx, window_start : seq_len + tree_size
                ]
                snapshot["req_to_token_window"] = {
                    "req_idx": req_idx,
                    "range": f"[{window_start}:{seq_len + tree_size}]",
                    "values": window.tolist()[:16],  # First 16 values
                }
        except Exception as e:
            snapshot["req_to_token_error"] = str(e)

        # Accept info
        try:
            if accept_index is not None:
                snapshot["accept_index"] = accept_index.tolist()[:8]
            if accept_length is not None:
                snapshot["accept_length"] = accept_length.tolist()
        except Exception as e:
            snapshot["accept_error"] = str(e)

        self._log_event("STATE_SNAPSHOT", level=3, **snapshot)

    def close(self):
        """Close the events file."""
        if self._events_file:
            try:
                self._events_file.close()
            except Exception:
                pass


# Global singleton
_debug_state: Optional[EagleDebugState] = None
_debug_lock = threading.Lock()


def get_debug_state() -> EagleDebugState:
    """Get the global debug state singleton."""
    global _debug_state
    if _debug_state is None:
        with _debug_lock:
            if _debug_state is None:
                _debug_state = EagleDebugState()
    return _debug_state


def reset_debug_state():
    """Reset the global debug state (for testing)."""
    global _debug_state
    with _debug_lock:
        if _debug_state is not None:
            _debug_state.close()
        _debug_state = None


# Convenience functions
def debug_checkpoint(name: str, stream: str = "main", level: int = 1, **data):
    """Log a checkpoint (convenience wrapper)."""
    get_debug_state().checkpoint(name, stream=stream, level=level, **data)


def debug_enabled(level: int = 1) -> bool:
    """Check if debugging is enabled at the given level."""
    return get_debug_state().is_enabled(level)


def debug_next_iter() -> int:
    """Increment iteration counter (convenience wrapper)."""
    return get_debug_state().next_iter()


def debug_iter() -> int:
    """Get current iteration (convenience wrapper)."""
    return get_debug_state().iter


def debug_data_move_trace(
    operation: str,
    src_slots: Optional[torch.Tensor] = None,
    dst_slots: Optional[torch.Tensor] = None,
    num_moves: Optional[int] = None,
    stream: str = "main",
    context: str = "",
):
    """Log KV cache data move operations (convenience wrapper)."""
    get_debug_state().data_move_trace(
        operation=operation,
        src_slots=src_slots,
        dst_slots=dst_slots,
        num_moves=num_moves,
        stream=stream,
        context=context,
    )
