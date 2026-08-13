"""Lifecycle for canonical checkpoints and rank-ready CPU weight images."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from sglang.srt.weight_sync.canonical_checkpoint import (
    CanonicalCheckpoint,
    DiskCanonicalCheckpointUpdate,
)
from sglang.srt.weight_sync.cpu_delta_checkpoint import DeltaCheckpointTransform
from sglang.srt.weight_sync.cpu_weight_compiler import CPUWeightCompiler

logger = logging.getLogger(__name__)


class CPUWeightCache:
    """Own the background preparation state for one target-model worker.

    One canonical checkpoint is shared by the model workers on a host. Each
    worker compiles that source into its own rank-ready pinned image. Staging
    never mutates live model weights; :meth:`commit` is the only operation that
    copies prepared bytes to CUDA weights.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_compile_group_bytes: int,
        host_group: torch.distributed.ProcessGroup | None,
        canonical_checkpoint_dir: str | Path | None = None,
    ):
        self.host_group = host_group
        self.compiler = CPUWeightCompiler(
            model,
            max_group_bytes=max_compile_group_bytes,
        )
        self._base_checkpoint_dir: str | None = None
        self._base_version: int | None = None
        self._checkpoint_source_dir: str | None = None
        self._canonical_checkpoint_dir = (
            os.path.realpath(canonical_checkpoint_dir)
            if canonical_checkpoint_dir is not None
            else None
        )
        self._canonical_checkpoint: CanonicalCheckpoint | None = None
        self._canonical_materialization_stats: dict[str, Any] | None = None
        self._operation_lock = threading.Lock()

    @contextmanager
    def _exclusive(self, operation: str):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError(
                f"cannot {operation} while another CPU weight cache operation "
                "is running"
            )
        try:
            yield
        finally:
            self._operation_lock.release()

    @property
    def image(self):
        return self.compiler.image

    @property
    def canonical_version(self) -> int | None:
        checkpoint = self._canonical_checkpoint
        return checkpoint.version if checkpoint is not None else None

    def _run_on_host_ranks(
        self,
        operation: str,
        function: Callable[[], Any],
    ) -> Any:
        distributed = torch.distributed.is_initialized()
        world_size = (
            torch.distributed.get_world_size(group=self.host_group)
            if distributed
            else 1
        )
        rank = torch.distributed.get_rank(group=self.host_group) if distributed else 0
        if world_size > 1 and self.host_group is None:
            raise RuntimeError(f"{operation} requires a host-local process group")

        result = None
        local_error = None
        local_exception = None
        try:
            result = function()
        except Exception as exc:
            local_exception = exc
            local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        if world_size > 1:
            errors: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                errors,
                local_error,
                group=self.host_group,
            )
        else:
            errors = [local_error]
        errors = [error for error in errors if error is not None]
        if errors:
            if world_size == 1 and local_exception is not None:
                raise local_exception
            raise RuntimeError(f"{operation} failed: " + "; ".join(errors))
        return result

    def _seed_canonical_checkpoint(
        self, *, reason: str | None = None
    ) -> CanonicalCheckpoint:
        if self._base_checkpoint_dir is None:
            raise RuntimeError("CPU weight cache has no base checkpoint")
        if self._base_version is None:
            raise RuntimeError("CPU weight cache has no base version")
        previous = self._canonical_checkpoint
        self._canonical_checkpoint = None
        self._checkpoint_source_dir = None
        if previous is not None:
            previous.close()
        checkpoint_dir = self._base_checkpoint_dir
        storage = "memory"
        if self._canonical_checkpoint_dir is not None:
            from sglang.srt.weight_sync.disk_checkpoint import materialize

            self._canonical_materialization_stats = materialize(
                local_checkpoint_dir=self._canonical_checkpoint_dir,
                base_checkpoint_dir=self._base_checkpoint_dir,
                checkpoint_source_dir=self._base_checkpoint_dir,
                target_version=self._base_version,
                base_version=self._base_version,
            )
            checkpoint_dir = self._canonical_checkpoint_dir
            storage = "disk"
        else:
            self._canonical_materialization_stats = None
        self._canonical_checkpoint = CanonicalCheckpoint(
            checkpoint_dir,
            host_group=self.host_group,
            version=self._base_version,
            storage=storage,
        )
        if reason is not None:
            logger.info("Reset canonical CPU checkpoint: %s", reason)
        return self._canonical_checkpoint

    def initialize_from_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        seed_from_active_weights: bool,
        base_version: int = 0,
    ) -> dict[str, Any]:
        """Build and validate reusable CPU state from the loaded checkpoint."""

        with self._exclusive("initialize"):
            return self._initialize_from_checkpoint(
                checkpoint_dir,
                seed_from_active_weights=seed_from_active_weights,
                base_version=base_version,
            )

    def _initialize_from_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        seed_from_active_weights: bool,
        base_version: int,
    ) -> dict[str, Any]:
        if self._base_checkpoint_dir is not None:
            raise RuntimeError("CPU weight cache is already initialized")
        if base_version < 0:
            raise ValueError("base_version must be non-negative")
        started = time.perf_counter()
        self._base_checkpoint_dir = os.path.realpath(checkpoint_dir)
        self._base_version = base_version
        try:
            image_initialization = self._run_on_host_ranks(
                "CPU weight image initialization",
                self.compiler.initialize_from_active,
            )
            checkpoint = self._seed_canonical_checkpoint()
            try:
                loader_view_preparation = self._run_on_host_ranks(
                    "CPU weight loader view preparation",
                    lambda: self.compiler.prepare_loader_views(checkpoint.weight_map),
                )
                if seed_from_active_weights:
                    baseline_compile = None
                    validation = None
                    rank_image_source = "active_model"
                else:

                    def compile_baseline():
                        compile_stats = self.compiler.compile(
                            checkpoint,
                            target_version=base_version,
                        )
                        validation_stats = self.image.validate_against_active()
                        self.image.accept_staged_baseline()
                        return compile_stats, validation_stats

                    baseline_compile, validation = self._run_on_host_ranks(
                        "CPU weight cache baseline validation",
                        compile_baseline,
                    )
                    rank_image_source = "checkpoint"
            finally:
                cache_release = self._run_on_host_ranks(
                    "canonical checkpoint page-cache release",
                    checkpoint.release_cached_pages,
                )
        except Exception:
            self._discard_canonical_checkpoint("CPU weight cache initialization failed")
            self._base_checkpoint_dir = None
            self._base_version = None
            raise

        canonical_stats = checkpoint.stats()
        stats = {
            "operation": "initialize_cpu_weight_cache",
            "base_version": base_version,
            "canonical_checkpoint": canonical_stats,
            "canonical_materialization": self._canonical_materialization_stats,
            "rank_image_bytes": self.image.image_nbytes,
            "rank_weight_bytes": self.image.weight_nbytes,
            "rank_image_source": rank_image_source,
            "image_initialization": image_initialization,
            "loader_view_preparation": loader_view_preparation,
            "baseline_compile": baseline_compile,
            "validation": validation,
            "canonical_cache_release": cache_release,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "Initialized CPU weight cache: canonical_bytes=%d "
            "rank_image_bytes=%d wall_time=%.3fs",
            canonical_stats["checkpoint_bytes"],
            self.image.image_nbytes,
            stats["wall_s"],
        )
        return stats

    def _canonical_checkpoint_for_lineage(
        self,
        *,
        checkpoint_source_dir: str | Path,
        target_version: int,
    ) -> tuple[CanonicalCheckpoint, bool]:
        if self._base_checkpoint_dir is None:
            raise RuntimeError("CPU weight cache is not initialized")
        source = os.path.realpath(checkpoint_source_dir)
        checkpoint = self._canonical_checkpoint
        reset_reason = None
        if checkpoint is None or not checkpoint.valid:
            reset_reason = "canonical checkpoint is unavailable"
        elif target_version < checkpoint.version:
            reset_reason = (
                f"requested rollback from v{checkpoint.version} to v{target_version}"
            )
        elif (
            self._checkpoint_source_dir is not None
            and self._checkpoint_source_dir != source
            and self._base_version is not None
            and checkpoint.version > self._base_version
        ):
            reset_reason = "checkpoint lineage changed"

        if reset_reason is not None:
            checkpoint = self._seed_canonical_checkpoint(reason=reset_reason)
        if checkpoint is None:
            raise RuntimeError("canonical checkpoint is unavailable")
        self._checkpoint_source_dir = source
        return checkpoint, reset_reason is not None

    def stage_delta_lineage(
        self,
        *,
        checkpoint_source_dir: str | Path,
        target_version: int,
    ) -> dict[str, Any]:
        """Advance and compile one delta target while inference continues."""

        with self._exclusive("stage weights"):
            return self._stage_delta_lineage(
                checkpoint_source_dir=checkpoint_source_dir,
                target_version=target_version,
            )

    def _stage_delta_lineage(
        self,
        *,
        checkpoint_source_dir: str | Path,
        target_version: int,
    ) -> dict[str, Any]:
        if self._base_version is None:
            raise RuntimeError("CPU weight cache has no base version")
        if target_version <= self._base_version:
            raise ValueError("CPU delta staging target must follow the base version")
        started = time.perf_counter()
        checkpoint, canonical_reset = self._canonical_checkpoint_for_lineage(
            checkpoint_source_dir=checkpoint_source_dir,
            target_version=target_version,
        )
        disk_canonical_invalid = False
        compile_started = False
        try:
            if self._canonical_checkpoint_dir is None:
                transform = DeltaCheckpointTransform(
                    checkpoint,
                    checkpoint_source_dir=checkpoint_source_dir,
                    target_version=target_version,
                    host_group=self.host_group,
                )
                setup_stats = transform.setup_stats
                self.compiler.validate_delta_names(transform.operations_by_name)
                transform_stats = transform.apply()
                materialization_stats = None
                try:
                    compile_started = True
                    compile_stats = self._run_on_host_ranks(
                        f"CPU weight image compilation for version {target_version}",
                        lambda: self.compiler.compile(
                            checkpoint,
                            target_version=target_version,
                        ),
                    )
                finally:
                    cache_release = self._run_on_host_ranks(
                        "canonical checkpoint page-cache release",
                        checkpoint.release_cached_pages,
                    )
            elif checkpoint.version == target_version:
                setup_stats = None
                transform_stats = None
                materialization_stats = {
                    "operation": "noop",
                    "target_version": target_version,
                    "wall_s": 0.0,
                }
                try:
                    compile_started = True
                    compile_stats = self._run_on_host_ranks(
                        f"CPU weight image compilation for version {target_version}",
                        lambda: self.compiler.compile(
                            checkpoint,
                            target_version=target_version,
                        ),
                    )
                finally:
                    cache_release = self._run_on_host_ranks(
                        "canonical checkpoint page-cache release",
                        checkpoint.release_cached_pages,
                    )
            else:
                transform = DeltaCheckpointTransform(
                    checkpoint,
                    checkpoint_source_dir=checkpoint_source_dir,
                    target_version=target_version,
                    host_group=self.host_group,
                )
                setup_stats = transform.setup_stats
                self.compiler.validate_delta_names(transform.operations_by_name)
                names_by_group = self.compiler.checkpoint_groups(checkpoint.weight_map)
                update = self._run_on_host_ranks(
                    "disk canonical update construction",
                    lambda: DiskCanonicalCheckpointUpdate(
                        checkpoint,
                        names_by_group=names_by_group,
                        transform=transform,
                        target_version=target_version,
                        host_group=self.host_group,
                    ),
                )
                self._canonical_checkpoint = None
                checkpoint.close()
                disk_canonical_invalid = True
                with update:
                    compile_started = True
                    compile_stats = self._run_on_host_ranks(
                        f"CPU weight image compilation for version {target_version}",
                        lambda: self.compiler.compile(
                            update,
                            target_version=target_version,
                        ),
                    )
                    materialization_stats = update.finish()
                    transform_stats = materialization_stats["delta_transform"]
                checkpoint = CanonicalCheckpoint(
                    self._canonical_checkpoint_dir,
                    host_group=self.host_group,
                    version=target_version,
                    storage="disk",
                )
                self._canonical_checkpoint = checkpoint
                self._canonical_materialization_stats = materialization_stats
                disk_canonical_invalid = False
                cache_release = self._run_on_host_ranks(
                    "canonical checkpoint page-cache release",
                    checkpoint.release_cached_pages,
                )
        except Exception:
            if compile_started:
                self.image.invalidate(
                    f"distributed staging of version {target_version} failed"
                )
            if disk_canonical_invalid:
                self._discard_canonical_checkpoint(
                    f"staging of version {target_version} invalidated the "
                    "disk canonical checkpoint"
                )
            elif not checkpoint.valid:
                self._discard_canonical_checkpoint(
                    f"staging of version {target_version} left it invalid"
                )
            raise

        stats = {
            "operation": "stage_cpu_weight_update",
            "target_version": target_version,
            "canonical_reset": canonical_reset,
            "delta_setup": setup_stats,
            "delta_transform": transform_stats,
            "canonical_materialization": materialization_stats,
            "compile": compile_stats,
            "canonical_cache_release": cache_release,
            "canonical_checkpoint": checkpoint.stats(),
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "Staged CPU weight image v%d: canonical_reset=%s wall_time=%.3fs",
            target_version,
            canonical_reset,
            stats["wall_s"],
        )
        return stats

    def invalidate_stage(self, reason: str) -> None:
        """Discard distributed preparation state after any worker fails."""

        with self._operation_lock:
            self.image.invalidate(reason)
            self._discard_canonical_checkpoint(reason)

    def _discard_canonical_checkpoint(self, reason: str) -> None:
        checkpoint = self._canonical_checkpoint
        self._canonical_checkpoint = None
        self._checkpoint_source_dir = None
        if checkpoint is not None:
            checkpoint.close()
            logger.warning("Discarded canonical CPU checkpoint: %s", reason)

    def commit(self, target_version: int) -> dict[str, Any]:
        with self._exclusive("commit weights"):
            return self.image.commit(target_version)

    def close(self, reason: str = "CPU weight cache closed") -> None:
        with self._operation_lock:
            self._discard_canonical_checkpoint(reason)
            self.compiler.close()
