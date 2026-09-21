"""Compile canonical checkpoints into rank-local host weight images."""

from __future__ import annotations

import gc
import logging
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.post_load import stage_module_for_post_load
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.weight_sync.canonical_checkpoint import CanonicalCheckpoint
from sglang.srt.weight_sync.rank_weight_image import (
    RankWeightImage,
    iter_weight_tensors,
)
from sglang.srt.weight_sync.weight_load_isolation import (
    WeightLoadGroup,
    build_weight_load_groups,
    build_weight_loader_view,
)

logger = logging.getLogger(__name__)


def _storage_key(tensor: torch.Tensor) -> tuple[int | None, int, int]:
    storage = tensor.untyped_storage()
    return tensor.device.index, storage.data_ptr(), storage.nbytes()


def _checkpoint_name_mapper(model: torch.nn.Module) -> WeightsMapper | None:
    """Return the model's raw-checkpoint to runtime-module name mapping."""

    mapper = getattr(model, "checkpoint_name_mapper", None)
    if mapper is not None:
        return mapper
    mapper_factory = getattr(type(model), "get_hf_to_sglang_mapper", None)
    if mapper_factory is not None:
        return mapper_factory(model.config)
    return getattr(model, "hf_to_sglang_mapper", None)


def _longest_group_prefix(name: str, paths: set[str]) -> str | None:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in paths:
            return candidate
    return "" if "" in paths else None


def _checkpoint_groups(
    model: torch.nn.Module,
    names: Iterable[str],
    groups: list[WeightLoadGroup],
) -> tuple[dict[str, list[str]], set[str]]:
    paths = {group.path for group in groups}
    mapper = _checkpoint_name_mapper(model)
    names_by_group = {group.path: [] for group in groups}
    ignored = set()
    unmapped = []

    for checkpoint_name in names:
        mapped = (
            checkpoint_name if mapper is None else mapper._map_name(checkpoint_name)
        )
        if mapped is None:
            ignored.add(checkpoint_name)
            continue

        group_path = _longest_group_prefix(mapped, paths)
        if group_path is None:
            unmapped.append(checkpoint_name)
        else:
            names_by_group[group_path].append(checkpoint_name)

    if unmapped:
        raise ValueError(
            "checkpoint tensors do not map to rank-local weight storage: "
            f"{unmapped[:20]}"
        )
    return names_by_group, ignored


@dataclass
class _PreparedLoadGroup:
    group: WeightLoadGroup
    checkpoint_names: tuple[str, ...]
    model_view: torch.nn.Module
    shadow: torch.nn.Module
    loader_state_bytes: int


class RankWeightCompiler:
    """Use native model loaders to build a complete rank-local host image."""

    def __init__(self, model: torch.nn.Module, *, max_group_bytes: int):
        if getattr(model, "secondary_weights", None):
            raise NotImplementedError(
                "rank weight compilation does not support secondary checkpoints"
            )
        self.model = model
        self.groups = build_weight_load_groups(
            model,
            max_group_bytes=max_group_bytes,
        )
        self.image = RankWeightImage(model)
        self._stream = (
            torch.cuda.Stream(device=self.image.device)
            if self.image.device.type == "cuda"
            else None
        )
        self._prepared_groups: list[_PreparedLoadGroup] | None = None
        self._checkpoint_names: frozenset[str] | None = None
        self._ignored_checkpoint_names: frozenset[str] = frozenset()
        logger.info(
            "Rank weight compiler layout: groups=%d storages=%d bytes=%d "
            "max_group_bytes=%d",
            len(self.groups),
            len(self.image.segments),
            self.image.weight_nbytes,
            max_group_bytes,
        )

    def initialize_from_active(self) -> dict[str, Any]:
        """Seed and register the host image from the serving weights."""

        started = time.perf_counter()
        registration = self.image.register_host_memory()
        capture = self.image.capture_active_weights()
        return {
            "operation": "initialize_rank_weight_compiler",
            "registration": registration,
            "capture": capture,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def prepare_loader_views(self, weight_map: dict[str, str]) -> dict[str, Any]:
        """Build reusable native-loader views backed by the host image."""

        started = time.perf_counter()
        checkpoint_names = frozenset(weight_map)
        if self._checkpoint_names is not None:
            if checkpoint_names != self._checkpoint_names:
                raise RuntimeError(
                    "canonical checkpoint tensor names changed after loader "
                    "views were prepared"
                )
            if self._prepared_groups is None:
                raise RuntimeError("rank weight loader view preparation is incomplete")
            return {
                "operation": "prepare_rank_weight_loader_views",
                "groups": len(self._prepared_groups),
                "loader_state_bytes": sum(
                    group.loader_state_bytes for group in self._prepared_groups
                ),
                "reused": True,
                "wall_s": round(time.perf_counter() - started, 6),
            }

        names_by_group, ignored = _checkpoint_groups(
            self.model,
            weight_map,
            self.groups,
        )
        prepared_groups = []
        try:
            for group in self.groups:
                names = tuple(names_by_group[group.path])
                if not names:
                    continue
                loader_state_bytes = 0

                def storage_factory(
                    tensor: torch.Tensor,
                    source_bytes: torch.Tensor,
                ) -> torch.Tensor:
                    nonlocal loader_state_bytes
                    try:
                        storage_bytes = self.image.storage_image_bytes(tensor)
                    except KeyError:
                        storage_bytes = source_bytes.to("cpu").clone()
                        loader_state_bytes += storage_bytes.numel()
                    return storage_bytes

                model_view, shadow = build_weight_loader_view(
                    self.model,
                    group.path,
                    target_device=torch.device("cpu"),
                    copy_data=False,
                    storage_factory=storage_factory,
                )
                prepared_groups.append(
                    _PreparedLoadGroup(
                        group=group,
                        checkpoint_names=names,
                        model_view=model_view,
                        shadow=shadow,
                        loader_state_bytes=loader_state_bytes,
                    )
                )
        except Exception:
            prepared_groups.clear()
            gc.collect()
            raise

        self._prepared_groups = prepared_groups
        self._checkpoint_names = checkpoint_names
        self._ignored_checkpoint_names = frozenset(ignored)
        stats = {
            "operation": "prepare_rank_weight_loader_views",
            "groups": len(prepared_groups),
            "loader_state_bytes": sum(
                group.loader_state_bytes for group in prepared_groups
            ),
            "ignored_checkpoint_tensors": len(ignored),
            "reused": False,
            "wall_s": round(time.perf_counter() - started, 6),
        }
        logger.info(
            "Prepared rank weight loader views: groups=%d loader_state_bytes=%d "
            "ignored_checkpoint_tensors=%d wall_time=%.3fs",
            stats["groups"],
            stats["loader_state_bytes"],
            stats["ignored_checkpoint_tensors"],
            stats["wall_s"],
        )
        return stats

    def validate_delta_names(self, names: Iterable[str]) -> None:
        ignored = self._ignored_checkpoint_names.intersection(names)
        if ignored:
            raise ValueError(
                "delta changes checkpoint tensors excluded from this model: "
                f"{sorted(ignored)[:20]}"
            )

    def _copy_shadow_to_image(
        self,
        path: str,
        shadow: torch.nn.Module,
    ) -> tuple[set[int], int, int, int]:
        updated = set()
        runtime_bytes = 0
        cpu_copy_bytes = 0
        device_copy_bytes = 0
        seen_storages = set()
        device_copies = []

        for relative_name, tensor in iter_weight_tensors(shadow):
            if tensor.device.type not in {"cpu", "cuda"}:
                continue
            if tensor.untyped_storage().nbytes() == 0:
                continue
            key = _storage_key(tensor)
            if key in seen_storages:
                continue
            seen_storages.add(key)
            full_name = ".".join(part for part in (path, relative_name) if part)
            segment = self.image.segments_by_name.get(full_name)
            if segment is None:
                raise RuntimeError(
                    f"native weight load produced unknown storage {full_name!r}"
                )
            if id(segment) in updated:
                raise RuntimeError(
                    "native weight load split aliased runtime storage: "
                    f"name={full_name!r} storage={segment.name!r}"
                )
            source = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                tensor.untyped_storage(),
                0,
                (tensor.untyped_storage().nbytes(),),
                (1,),
            )
            if source.numel() != segment.nbytes:
                raise RuntimeError(
                    "native weight load changed storage size: "
                    f"name={full_name!r} source={source.numel()} "
                    f"target={segment.nbytes}"
                )
            if tensor.device.type == "cuda":
                device_copies.append((segment, source))
                device_copy_bytes += segment.nbytes
            else:
                target = self.image.image[
                    segment.image_offset : segment.image_offset + segment.nbytes
                ]
                if source.data_ptr() != target.data_ptr():
                    target.copy_(source)
                    cpu_copy_bytes += segment.nbytes
            updated.add(id(segment))
            runtime_bytes += segment.nbytes

        if device_copies:
            self.image.copy_device_segments_to_image(device_copies)
        return updated, runtime_bytes, cpu_copy_bytes, device_copy_bytes

    def _compile_group(
        self,
        prepared: _PreparedLoadGroup,
        checkpoint: CanonicalCheckpoint,
    ) -> tuple[set[int], int, dict[str, Any]]:
        started = time.perf_counter()
        phase_started = time.perf_counter()
        DefaultModelLoader.restore_weights_before_loading(
            prepared.shadow,
            torch.device("cpu"),
        )
        restore_s = time.perf_counter() - phase_started

        weights = (
            (name, checkpoint.get_tensor(name)) for name in prepared.checkpoint_names
        )
        phase_started = time.perf_counter()
        DefaultModelLoader.load_weights_only(
            prepared.model_view,
            weights,
            torch.device("cpu"),
        )
        load_s = time.perf_counter() - phase_started
        del weights

        phase_started = time.perf_counter()
        if self._stream is None:
            DefaultModelLoader.postprocess_weights(
                prepared.shadow,
                self.image.device,
            )
        else:
            # Stage one bounded load group at a time so host/device traffic is
            # submitted as a batch rather than synchronized once per module.
            with torch.cuda.stream(self._stream):
                with stage_module_for_post_load(
                    prepared.shadow,
                    self.image.device,
                    pin_memory=True,
                    non_blocking=True,
                ):
                    for _, module in prepared.shadow.named_modules():
                        DefaultModelLoader.process_module_weights_after_loading(module)
        postprocess_s = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        updated, group_bytes, cpu_copy_bytes, device_copy_bytes = (
            self._copy_shadow_to_image(prepared.group.path, prepared.shadow)
        )
        image_copy_s = time.perf_counter() - phase_started
        stats = {
            "path": prepared.group.path,
            "checkpoint_tensors": len(prepared.checkpoint_names),
            "bytes": group_bytes,
            "cpu_image_copy_bytes": cpu_copy_bytes,
            "device_image_copy_bytes": device_copy_bytes,
            "restore_s": round(restore_s, 6),
            "load_s": round(load_s, 6),
            "postprocess_s": round(postprocess_s, 6),
            "image_copy_s": round(image_copy_s, 6),
            "wall_s": round(time.perf_counter() - started, 6),
        }
        return updated, group_bytes, stats

    def compile(
        self,
        checkpoint: CanonicalCheckpoint,
        *,
        target_version: int,
    ) -> dict[str, Any]:
        """Compile every rank-local weight storage without changing live weights."""

        if target_version < 0:
            raise ValueError("target_version must be non-negative")
        if checkpoint.version != target_version:
            raise ValueError(
                "canonical checkpoint version does not match compile target: "
                f"checkpoint={checkpoint.version} target={target_version}"
            )
        if self.image.staging:
            raise RuntimeError("a rank weight image stage is already running")
        if self.image.staged:
            raise RuntimeError(
                "a rank weight image is already staged; commit it before "
                "compiling another target"
            )

        started = time.perf_counter()
        view_stats = self.prepare_loader_views(checkpoint.weight_map)
        if self._prepared_groups is None:
            raise RuntimeError("rank weight loader views were not prepared")
        prepared_by_path = {
            prepared.group.path: prepared for prepared in self._prepared_groups
        }
        covered_segments = set()
        preserved_segments = set()
        staged_bytes = 0
        group_stats = []

        try:
            if not self.image.registered:
                self.image.register_host_memory()
            if not self.image.valid:
                self.image.capture_active_weights()
            self.image.begin_stage(target_version)

            progress_interval = max(1, math.ceil(len(self.groups) / 10))
            for index, group in enumerate(self.groups, start=1):
                prepared = prepared_by_path.get(group.path)
                if prepared is None:
                    prefix = f"{group.path}." if group.path else ""
                    segments = {
                        id(segment): segment
                        for name, segment in self.image.segments_by_name.items()
                        if not group.path
                        or name == group.path
                        or name.startswith(prefix)
                    }
                    if not segments:
                        raise RuntimeError(
                            "weight load group has no rank-local storage: "
                            f"path={group.path!r}"
                        )
                    covered_segments.update(segments)
                    preserved_segments.update(segments)
                    staged_bytes += sum(segment.nbytes for segment in segments.values())
                else:
                    try:
                        updated, group_bytes, stats = self._compile_group(
                            prepared,
                            checkpoint,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "rank weight compilation failed for group "
                            f"{index}/{len(self.groups)} at "
                            f"{group.path or '<root>'!r}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    covered_segments.update(updated)
                    staged_bytes += group_bytes
                    group_stats.append(stats)

                if (
                    index == 1
                    or index % progress_interval == 0
                    or index == len(self.groups)
                ):
                    logger.info(
                        "Rank weight image v%d progress: groups=%d/%d bytes=%d "
                        "elapsed=%.3fs",
                        target_version,
                        index,
                        len(self.groups),
                        staged_bytes,
                        time.perf_counter() - started,
                    )

            expected_segments = {id(segment) for segment in self.image.segments}
            missing = expected_segments - covered_segments
            if missing:
                missing_names = [
                    segment.name
                    for segment in self.image.segments
                    if id(segment) in missing
                ]
                raise RuntimeError(
                    "canonical checkpoint did not produce every rank-local "
                    f"weight storage: {missing_names[:20]}"
                )
            commit_segments = [
                segment
                for segment in self.image.segments
                if id(segment) not in preserved_segments
            ]
            self.image.finish_stage(target_version, commit_segments)
        except Exception as exc:
            self.image.invalidate(
                f"compilation of version {target_version} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

        phases = {
            phase: round(sum(group[phase] for group in group_stats), 6)
            for phase in (
                "restore_s",
                "load_s",
                "postprocess_s",
                "image_copy_s",
            )
        }
        traffic = {
            name: sum(group[name] for group in group_stats)
            for name in (
                "cpu_image_copy_bytes",
                "device_image_copy_bytes",
            )
        }
        stats = {
            "operation": "compile_rank_weight_image",
            "target_version": target_version,
            "groups": len(self.groups),
            "loader_views": view_stats,
            "checkpoint_tensors": len(checkpoint.weight_map),
            "ignored_checkpoint_tensors": len(self._ignored_checkpoint_names),
            "runtime_storages": len(covered_segments),
            "preserved_storages": len(preserved_segments),
            "bytes": staged_bytes,
            "preserved_bytes": sum(
                segment.nbytes
                for segment in self.image.segments
                if id(segment) in preserved_segments
            ),
            "commit_bytes": sum(segment.nbytes for segment in commit_segments),
            "wall_s": round(time.perf_counter() - started, 6),
            "phases": phases,
            "traffic": traffic,
        }
        logger.info(
            "Compiled rank weight image v%d: bytes=%d wall_time=%.3fs phases=%s",
            target_version,
            staged_bytes,
            stats["wall_s"],
            phases,
        )
        return stats

    def close(self) -> None:
        if self._prepared_groups is not None:
            self._prepared_groups.clear()
            self._prepared_groups = None
            self._checkpoint_names = None
            self._ignored_checkpoint_names = frozenset()
            gc.collect()
        self.image.close()
