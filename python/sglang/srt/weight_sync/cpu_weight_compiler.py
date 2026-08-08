"""Compile canonical checkpoints into rank-local CPU weight images."""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.weight_sync.canonical_checkpoint import CanonicalCheckpoint
from sglang.srt.weight_sync.cpu_weight_image import (
    CPUWeightImage,
    iter_weight_tensors,
)
from sglang.srt.weight_sync.weight_loader_isolation import (
    WeightModuleGroup,
    build_weight_loader_proxy,
    build_weight_module_groups,
    clone_weight_module,
    map_checkpoint_names_to_groups,
)

logger = logging.getLogger(__name__)


def _storage_key(tensor: torch.Tensor) -> tuple[int | None, int, int]:
    storage = tensor.untyped_storage()
    return tensor.device.index, storage.data_ptr(), storage.nbytes()


def _staging_postprocess_device(model: torch.nn.Module) -> str:
    """Return the device required by this model's post-load transformations."""

    device = "cpu"
    for module_name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None:
            continue
        get_device = getattr(
            quant_method,
            "weight_staging_postprocess_device",
            None,
        )
        method_device = get_device(module) if callable(get_device) else None
        if method_device not in {"cpu", "cuda"}:
            raise NotImplementedError(
                "CPU weight staging is unsupported for quantization method "
                f"{type(quant_method).__name__} at {module_name or '<root>'}"
            )
        if method_device == "cuda":
            device = "cuda"
    return device


@dataclass
class _LoadedWeightGroup:
    index: int
    group: WeightModuleGroup
    checkpoint_tensors: int
    cpu_shadow: torch.nn.Module
    image_storage_keys: set[tuple[int, int]]
    started: float
    cpu_clone_s: float
    restore_s: float
    cpu_load_s: float


class CPUWeightCompiler:
    """Use SGLang's native loader to build a complete rank-ready host image."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_group_bytes: int,
    ):
        if getattr(model, "secondary_weights", None):
            raise NotImplementedError(
                "CPU weight staging does not support models with secondary "
                "checkpoint sources"
            )
        self.model = model
        self.groups = build_weight_module_groups(
            model,
            max_group_bytes=max_group_bytes,
        )
        _staging_postprocess_device(model)
        self.image = CPUWeightImage(model)
        self.target_device = self.image.device
        self._compile_stream = torch.cuda.Stream(device=self.target_device)
        logger.info(
            "CPU weight compiler layout: groups=%d storages=%d bytes=%d "
            "max_group_bytes=%d",
            len(self.groups),
            len(self.image.segments),
            self.image.image_nbytes,
            max_group_bytes,
        )

    def initialize_from_active(self) -> dict[str, Any]:
        """Seed and register the persistent image while inference may continue."""

        started = time.perf_counter()
        registration = self.image.register_host_memory()
        capture = self.image.capture_active_weights()
        return {
            "operation": "initialize_cpu_weight_compiler",
            "registration": registration,
            "capture": capture,
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def _load_group(
        self,
        *,
        index: int,
        group: WeightModuleGroup,
        names: list[str],
        checkpoint: CanonicalCheckpoint,
    ) -> _LoadedWeightGroup:
        started = time.perf_counter()
        image_storage_keys: set[tuple[int, int]] = set()

        def storage_factory(
            tensor: torch.Tensor,
            source_bytes: torch.Tensor,
        ) -> torch.Tensor:
            try:
                storage_bytes = self.image.storage_image_bytes(tensor)
            except KeyError:
                # Loader metadata outside the image contract stays in bounded
                # group scratch rather than retaining live model storage.
                storage_bytes = source_bytes.to("cpu").clone()
            if tensor.device.type == "cuda":
                storage = storage_bytes.untyped_storage()
                image_storage_keys.add((storage.data_ptr(), storage.nbytes()))
            return storage_bytes

        phase_started = time.perf_counter()
        proxy, cpu_shadow = build_weight_loader_proxy(
            self.model,
            group.path,
            target_device=torch.device("cpu"),
            copy_data=False,
            storage_factory=storage_factory,
        )
        cpu_clone_s = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        DefaultModelLoader.restore_weights_before_cpu_staging(cpu_shadow)
        restore_s = time.perf_counter() - phase_started

        weights = ((name, checkpoint.get_tensor(name)) for name in names)
        phase_started = time.perf_counter()
        with DefaultModelLoader.weight_loading_context(proxy):
            proxy.load_weights(weights)
        cpu_load_s = time.perf_counter() - phase_started
        del proxy, weights

        return _LoadedWeightGroup(
            index=index,
            group=group,
            checkpoint_tensors=len(names),
            cpu_shadow=cpu_shadow,
            image_storage_keys=image_storage_keys,
            started=started,
            cpu_clone_s=cpu_clone_s,
            restore_s=restore_s,
            cpu_load_s=cpu_load_s,
        )

    def _copy_shadow_to_image(
        self,
        path: str,
        shadow: torch.nn.Module,
    ) -> tuple[set[int], int, int, int]:
        updated = set()
        runtime_bytes = 0
        cpu_copy_bytes = 0
        d2h_bytes = 0
        seen_storages = set()
        device_copies = []
        for relative_name, tensor in iter_weight_tensors(shadow):
            if tensor.device.type not in {"cpu", "cuda"}:
                continue
            key = _storage_key(tensor)
            if key in seen_storages:
                continue
            seen_storages.add(key)
            full_name = f"{path}.{relative_name}" if relative_name else path
            segment = self.image.segments_by_name.get(full_name)
            if segment is None:
                raise RuntimeError(
                    f"compiled shadow produced unknown weight {full_name!r}"
                )
            source = torch.empty(
                0,
                dtype=torch.uint8,
                device=tensor.device,
            ).set_(
                tensor.untyped_storage(),
                0,
                (tensor.untyped_storage().nbytes(),),
                (1,),
            )
            if source.numel() != segment.nbytes:
                raise RuntimeError(
                    "compiled shadow storage size changed: "
                    f"name={full_name!r} source={source.numel()} "
                    f"target={segment.nbytes}"
                )
            if tensor.device.type == "cuda":
                device_copies.append((segment, source))
                d2h_bytes += segment.nbytes
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
        return updated, runtime_bytes, cpu_copy_bytes, d2h_bytes

    @staticmethod
    def _postprocess(shadow: torch.nn.Module) -> None:
        for module in shadow.modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)

    def _finalize_group(
        self,
        loaded: _LoadedWeightGroup,
    ) -> tuple[set[int], int, dict[str, Any]]:
        group = loaded.group
        cpu_shadow = loaded.cpu_shadow
        postprocess_device = _staging_postprocess_device(cpu_shadow)
        background_h2d_bytes = 0

        if postprocess_device == "cpu":
            phase_started = time.perf_counter()
            self._postprocess(cpu_shadow)
            postprocess_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            (
                updated,
                group_bytes,
                image_copy_bytes,
                background_d2h_bytes,
            ) = self._copy_shadow_to_image(group.path, cpu_shadow)
            image_copy_s = time.perf_counter() - phase_started
            h2d_submit_s = 0.0
            device_sync_s = 0.0
            gpu_shadow = None
        else:
            model_state_ids = {
                id(tensor) for _, tensor in iter_weight_tensors(cpu_shadow)
            }

            def storage_factory(
                tensor: torch.Tensor,
                source_bytes: torch.Tensor,
            ) -> torch.Tensor:
                nonlocal background_h2d_bytes
                storage = tensor.untyped_storage()
                source_key = (storage.data_ptr(), storage.nbytes())
                target_device = (
                    self.target_device
                    if source_key in loaded.image_storage_keys
                    or id(tensor) in model_state_ids
                    else tensor.device
                )
                storage_bytes = torch.empty(
                    source_bytes.numel(),
                    dtype=torch.uint8,
                    device=target_device,
                )
                storage_bytes.copy_(source_bytes, non_blocking=True)
                if target_device.type == "cuda":
                    background_h2d_bytes += source_bytes.numel()
                return storage_bytes

            with torch.cuda.stream(self._compile_stream):
                phase_started = time.perf_counter()
                gpu_shadow = clone_weight_module(
                    cpu_shadow,
                    target_device=self.target_device,
                    copy_data=True,
                    storage_factory=storage_factory,
                )
                h2d_submit_s = time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                self._postprocess(gpu_shadow)
                postprocess_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            self._compile_stream.synchronize()
            device_sync_s = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            (
                updated,
                group_bytes,
                image_copy_bytes,
                background_d2h_bytes,
            ) = self._copy_shadow_to_image(group.path, gpu_shadow)
            image_copy_s = time.perf_counter() - phase_started

        stats = {
            "path": group.path,
            "checkpoint_tensors": loaded.checkpoint_tensors,
            "bytes": group_bytes,
            "postprocess_device": postprocess_device,
            "background_h2d_bytes": background_h2d_bytes,
            "background_d2h_bytes": background_d2h_bytes,
            "cpu_image_copy_bytes": image_copy_bytes,
            "cpu_clone_s": round(loaded.cpu_clone_s, 6),
            "restore_s": round(loaded.restore_s, 6),
            "cpu_load_s": round(loaded.cpu_load_s, 6),
            "h2d_submit_s": round(h2d_submit_s, 6),
            "postprocess_s": round(postprocess_s, 6),
            "device_sync_s": round(device_sync_s, 6),
            "image_copy_s": round(image_copy_s, 6),
            "wall_s": round(time.perf_counter() - loaded.started, 6),
        }
        logger.debug(
            "Compiled CPU weight group %d/%d: path=%s bytes=%d "
            "postprocess_device=%s wall_s=%.6f",
            loaded.index,
            len(self.groups),
            group.path,
            group_bytes,
            postprocess_device,
            stats["wall_s"],
        )
        del cpu_shadow, gpu_shadow
        gc.collect(0)
        return updated, group_bytes, stats

    def checkpoint_groups(
        self,
        weight_map: dict[str, str],
    ) -> dict[str, list[str]]:
        """Map every canonical tensor to its bounded runtime compile group."""

        group_for_name = map_checkpoint_names_to_groups(
            self.model,
            weight_map,
            self.groups,
        )
        names_by_group = {group.path: [] for group in self.groups}
        unmapped = []
        for name, group_path in group_for_name.items():
            if group_path is None:
                unmapped.append(name)
            else:
                names_by_group[group_path].append(name)
        if unmapped:
            raise RuntimeError(
                "CPU weight staging cannot map every checkpoint tensor to a "
                f"runtime weight group; unmapped={unmapped[:20]}"
            )
        return names_by_group

    def compile(
        self,
        checkpoint: CanonicalCheckpoint,
        *,
        target_version: int,
    ) -> dict[str, Any]:
        """Compile every runtime storage without changing the live model."""

        if target_version < 0:
            raise ValueError("target_version must be non-negative")
        if self.image.staging:
            raise RuntimeError("a CPU weight image stage is already running")
        started = time.perf_counter()
        names_by_group = self.checkpoint_groups(checkpoint.weight_map)

        updated_segments = set()
        copied_bytes = 0
        group_stats = []
        try:

            def begin_stage():
                if not self.image.registered:
                    self.image.register_host_memory()
                if not self.image.valid:
                    self.image.capture_active_weights()
                self.image.begin_stage(target_version)

            checkpoint.run_on_host_ranks(
                "CPU weight image stage initialization",
                begin_stage,
            )

            progress_interval = max(1, math.ceil(len(self.groups) / 10))
            for index, group in enumerate(self.groups, start=1):
                names = names_by_group[group.path]
                with checkpoint.tensor_group(group.path, names) as group_checkpoint:
                    loaded = self._load_group(
                        index=index,
                        group=group,
                        names=names,
                        checkpoint=group_checkpoint,
                    )
                    updated, group_bytes, stats = self._finalize_group(loaded)
                    del loaded
                updated_segments.update(updated)
                copied_bytes += group_bytes
                group_stats.append(stats)
                if (
                    index == 1
                    or index % progress_interval == 0
                    or index == len(self.groups)
                ):
                    logger.info(
                        "CPU weight image v%d progress: groups=%d/%d "
                        "bytes=%d elapsed=%.3fs",
                        target_version,
                        index,
                        len(self.groups),
                        copied_bytes,
                        time.perf_counter() - started,
                    )

            expected_segments = {id(segment) for segment in self.image.segments}
            missing = expected_segments - updated_segments
            if missing:
                missing_names = [
                    segment.name
                    for segment in self.image.segments
                    if id(segment) in missing
                ]
                raise RuntimeError(
                    "checkpoint did not produce every runtime weight storage; "
                    f"missing={missing_names[:20]}"
                )
            self.image.finish_stage(target_version)
        except Exception as exc:
            self.image.invalidate(
                f"checkpoint compilation of version {target_version} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

        phases = {
            phase: round(sum(group.get(phase, 0.0) for group in group_stats), 6)
            for phase in (
                "cpu_clone_s",
                "restore_s",
                "cpu_load_s",
                "h2d_submit_s",
                "postprocess_s",
                "device_sync_s",
                "image_copy_s",
            )
        }
        traffic = {
            name: sum(group[name] for group in group_stats)
            for name in (
                "background_h2d_bytes",
                "background_d2h_bytes",
                "cpu_image_copy_bytes",
            )
        }
        postprocess_bytes = {
            device: sum(
                group["bytes"]
                for group in group_stats
                if group["postprocess_device"] == device
            )
            for device in ("cpu", "cuda")
        }
        wall_s = time.perf_counter() - started
        stats = {
            "operation": "compile_cpu_weight_image",
            "target_version": target_version,
            "groups": len(self.groups),
            "checkpoint_tensors": len(checkpoint.weight_map),
            "runtime_storages": len(updated_segments),
            "bytes": copied_bytes,
            "wall_s": round(wall_s, 6),
            "compile_wall_s": round(
                sum(group["wall_s"] for group in group_stats),
                6,
            ),
            "phases": phases,
            "postprocess_bytes": postprocess_bytes,
            "traffic": traffic,
        }
        logger.info(
            "Compiled CPU weight image v%d: bytes=%d wall_time=%.3fs phases=%s",
            target_version,
            copied_bytes,
            wall_s,
            phases,
        )
        return stats

    def close(self) -> None:
        self.image.close()
