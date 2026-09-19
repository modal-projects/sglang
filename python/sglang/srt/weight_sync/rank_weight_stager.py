"""Stage versioned checkpoints into rank-local host weight images."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from sglang.srt.weight_sync.canonical_checkpoint import CanonicalCheckpoint
from sglang.srt.weight_sync.canonical_delta import CanonicalDeltaTransform
from sglang.srt.weight_sync.disk_checkpoint import materialize
from sglang.srt.weight_sync.rank_weight_compiler import RankWeightCompiler

logger = logging.getLogger(__name__)


class RankWeightStager:
    """Own canonical host weights and one inactive rank-local image.

    The canonical checkpoint may live in shared host memory or on host-local
    storage. Preparation advances that checkpoint and compiles a complete
    rank-local image without touching live model storage. ``commit`` is the
    only operation that mutates live weights.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_compile_group_bytes: int,
        host_group: torch.distributed.ProcessGroup | None,
        cuda_stream: torch.cuda.Stream,
        canonical_checkpoint_dir: str | Path | None = None,
    ):
        if max_compile_group_bytes <= 0:
            raise ValueError("max_compile_group_bytes must be positive")
        self.host_group = host_group
        self.compiler = RankWeightCompiler(
            model,
            max_group_bytes=max_compile_group_bytes,
            cuda_stream=cuda_stream,
        )
        self._canonical_checkpoint_dir = (
            os.path.realpath(canonical_checkpoint_dir)
            if canonical_checkpoint_dir is not None
            else None
        )
        self._base_checkpoint_dir: str | None = None
        self._base_version: int | None = None
        self._served_version: int | None = None
        self._checkpoint_source_dir: str | None = None
        self._canonical: CanonicalCheckpoint | None = None
        self._operation_lock = threading.Lock()

    @contextmanager
    def _exclusive(self, operation: str):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError(
                f"cannot {operation} while another staged weight operation is running"
            )
        try:
            yield
        finally:
            self._operation_lock.release()

    @property
    def served_version(self) -> int | None:
        return self._served_version

    @property
    def canonical_version(self) -> int | None:
        return self._canonical.version if self._canonical is not None else None

    @property
    def prepared_version(self) -> int | None:
        image = self.compiler.image
        return image.target_version if image.staged else None

    def _require_initialized(self) -> tuple[str, int]:
        if self._base_checkpoint_dir is None or self._base_version is None:
            raise RuntimeError("rank weight staging is not initialized")
        return self._base_checkpoint_dir, self._base_version

    def _close_canonical(self) -> None:
        checkpoint = self._canonical
        self._canonical = None
        if checkpoint is not None:
            checkpoint.close()

    def _seed_canonical(self) -> tuple[CanonicalCheckpoint, dict[str, Any] | None]:
        base_checkpoint_dir, base_version = self._require_initialized()
        self._close_canonical()

        checkpoint_dir = base_checkpoint_dir
        storage = "memory"
        materialization_stats = None
        if self._canonical_checkpoint_dir is not None:
            materialization_stats = materialize(
                local_checkpoint_dir=self._canonical_checkpoint_dir,
                base_checkpoint_dir=base_checkpoint_dir,
                checkpoint_source_dir=base_checkpoint_dir,
                target_version=base_version,
                base_version=base_version,
            )
            checkpoint_dir = self._canonical_checkpoint_dir
            storage = "disk"

        checkpoint = CanonicalCheckpoint(
            checkpoint_dir,
            host_group=self.host_group,
            version=base_version,
            storage=storage,
        )
        self._canonical = checkpoint
        self._checkpoint_source_dir = None
        return checkpoint, materialization_stats

    def initialize(
        self,
        checkpoint_dir: str | Path,
        *,
        version: int,
    ) -> dict[str, Any]:
        """Initialize from the checkpoint that produced the live model."""

        with self._exclusive("initialize rank weight staging"):
            if self._base_checkpoint_dir is not None:
                raise RuntimeError("rank weight staging is already initialized")
            if version < 0:
                raise ValueError("version must be non-negative")

            started = time.perf_counter()
            self._base_checkpoint_dir = os.path.realpath(checkpoint_dir)
            self._base_version = version
            self._served_version = version
            try:
                image_stats = self.compiler.initialize_from_active()
                checkpoint, materialization_stats = self._seed_canonical()
                loader_stats = self.compiler.prepare_loader_views(checkpoint.weight_map)
            except Exception:
                self._close_canonical()
                self.compiler.close()
                self._base_checkpoint_dir = None
                self._base_version = None
                self._served_version = None
                raise

            stats = {
                "operation": "initialize_rank_weight_staging",
                "version": version,
                "canonical_checkpoint": checkpoint.stats(),
                "canonical_materialization": materialization_stats,
                "rank_image": image_stats,
                "loader_views": loader_stats,
                "wall_s": round(time.perf_counter() - started, 6),
            }
            logger.info(
                "Initialized rank weight staging at v%d: canonical_bytes=%d "
                "rank_image_bytes=%d wall_time=%.3fs",
                version,
                checkpoint.checkpoint_bytes,
                self.compiler.image.image_nbytes,
                stats["wall_s"],
            )
            return stats

    def _canonical_for_target(
        self,
        checkpoint_source_dir: str | Path,
        target_version: int,
    ) -> tuple[CanonicalCheckpoint, bool, dict[str, Any] | None]:
        _, base_version = self._require_initialized()
        source = os.path.realpath(checkpoint_source_dir)
        checkpoint = self._canonical
        reset = (
            checkpoint is None
            or not checkpoint.valid
            or target_version < checkpoint.version
            or (
                self._checkpoint_source_dir is not None
                and self._checkpoint_source_dir != source
                and checkpoint.version > base_version
            )
        )
        materialization_stats = None
        if reset:
            checkpoint, materialization_stats = self._seed_canonical()
        if checkpoint is None:
            raise RuntimeError("canonical checkpoint is unavailable")
        self._checkpoint_source_dir = source
        return checkpoint, reset, materialization_stats

    def stage(
        self,
        *,
        checkpoint_source_dir: str | Path,
        target_version: int,
    ) -> dict[str, Any]:
        """Prepare one complete target without mutating live weights."""

        with self._exclusive("stage weights"):
            return self._stage(
                checkpoint_source_dir=checkpoint_source_dir,
                target_version=target_version,
            )

    def _stage(
        self,
        *,
        checkpoint_source_dir: str | Path,
        target_version: int,
    ) -> dict[str, Any]:
        base_checkpoint_dir, base_version = self._require_initialized()
        served_version = self._served_version
        if served_version is None:
            raise RuntimeError("served weight version is unavailable")
        if target_version <= served_version:
            raise ValueError(
                f"target version {target_version} must follow served version "
                f"{served_version}"
            )
        if target_version <= base_version:
            raise ValueError(
                f"target version {target_version} must follow base version "
                f"{base_version}"
            )

        image = self.compiler.image
        if image.staged:
            if image.target_version == target_version:
                return {
                    "operation": "stage_rank_weight_update",
                    "target_version": target_version,
                    "reused": True,
                    "wall_s": 0.0,
                }
            if (
                image.target_version is not None
                and target_version < image.target_version
            ):
                raise RuntimeError(
                    f"target version {target_version} must not precede prepared "
                    f"version {image.target_version}"
                )
            image.invalidate(f"superseded by version {target_version}")

        started = time.perf_counter()
        checkpoint, canonical_reset, seed_stats = self._canonical_for_target(
            checkpoint_source_dir,
            target_version,
        )
        transform_stats = None
        materialization_stats = seed_stats
        try:
            if self._canonical_checkpoint_dir is None:
                transform = CanonicalDeltaTransform(
                    checkpoint,
                    checkpoint_source_dir=checkpoint_source_dir,
                    target_version=target_version,
                    host_group=self.host_group,
                )
                self.compiler.validate_delta_names(transform.operations_by_name)
                transform_stats = transform.apply()
            elif checkpoint.version != target_version:
                self._close_canonical()
                try:
                    materialization_stats = materialize(
                        local_checkpoint_dir=self._canonical_checkpoint_dir,
                        base_checkpoint_dir=base_checkpoint_dir,
                        checkpoint_source_dir=os.path.realpath(checkpoint_source_dir),
                        target_version=target_version,
                        base_version=base_version,
                    )
                    checkpoint = CanonicalCheckpoint(
                        self._canonical_checkpoint_dir,
                        host_group=self.host_group,
                        version=target_version,
                        storage="disk",
                    )
                    self._canonical = checkpoint
                except Exception:
                    self._canonical = None
                    raise

            compile_stats = self.compiler.compile(
                checkpoint,
                target_version=target_version,
            )
            cache_release_stats = checkpoint.release_cached_pages()
        except Exception as exc:
            image.invalidate(
                f"staging of version {target_version} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if self._canonical is not None and not self._canonical.valid:
                self._close_canonical()
            raise

        stats = {
            "operation": "stage_rank_weight_update",
            "target_version": target_version,
            "canonical_reset": canonical_reset,
            "canonical_materialization": materialization_stats,
            "canonical_transform": transform_stats,
            "compile": compile_stats,
            "canonical_checkpoint": checkpoint.stats(),
            "canonical_cache_release": cache_release_stats,
            "reused": False,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "Prepared rank weight image v%d: canonical_storage=%s "
            "canonical_reset=%s wall_time=%.3fs",
            target_version,
            checkpoint.storage,
            canonical_reset,
            stats["wall_s"],
        )
        return stats

    def validate_commit(self, target_version: int) -> None:
        self._require_initialized()
        self.compiler.image.validate_commit(target_version)

    def commit(self, target_version: int) -> dict[str, Any]:
        """Copy the prepared image into live weight storage."""

        with self._exclusive("commit weights"):
            self.validate_commit(target_version)
            stats = self.compiler.image.commit(target_version)
            self._served_version = target_version
            return stats

    def discard_prepared(self, reason: str) -> None:
        """Invalidate an uncommitted image while retaining canonical bytes."""

        with self._operation_lock:
            self.compiler.image.invalidate(reason)

    def close(self) -> None:
        with self._operation_lock:
            self._close_canonical()
            self.compiler.close()
