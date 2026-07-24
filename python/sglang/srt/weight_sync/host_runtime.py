"""One pinned host mirror for short-pause, full-model weight commits.

The host image stores the final bytes of every unique CUDA storage owned by the
model.  It is not a checkpoint and it never runs a model loader: background
preparation is responsible for advancing these bytes to a verified target
version before :meth:`commit` is called.

Commit always overwrites existing CUDA storages instead of rebinding tensors.
This preserves aliases, parameter addresses, and CUDA graph captures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)

_ALIGNMENT = 4096


@dataclass(frozen=True)
class RuntimeStorageSegment:
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
        storage, 0, (storage.nbytes(),), (1,)
    )


def iter_model_tensors(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield registered state plus explicitly declared runtime weight tensors."""

    yield from model.named_parameters(remove_duplicate=False)
    yield from model.named_buffers(remove_duplicate=False)
    for module_name, module in model.named_modules():
        get_extra = getattr(module, "host_runtime_tensors", None)
        if get_extra is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        for name, tensor in get_extra():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "host_runtime_tensors() must yield (str, torch.Tensor)"
                )
            yield f"{prefix}{name}", tensor


def build_runtime_storage_plan(
    model: torch.nn.Module,
) -> tuple[list[RuntimeStorageSegment], int]:
    """Inventory every unique CUDA storage without changing its address."""

    unique: dict[tuple[int | None, int, int], tuple[str, torch.Tensor]] = {}
    for name, tensor in iter_model_tensors(model):
        if tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        nbytes = storage.nbytes()
        key = (tensor.device.index, storage.data_ptr(), nbytes)
        current = unique.get(key)
        if current is None or name < current[0]:
            unique[key] = (name, tensor)

    offset = 0
    segments: list[RuntimeStorageSegment] = []
    for name, tensor in sorted(unique.values(), key=lambda item: item[0]):
        offset = _align_up(offset)
        device_bytes = _storage_as_bytes(tensor)
        segments.append(
            RuntimeStorageSegment(
                name=name,
                image_offset=offset,
                nbytes=device_bytes.numel(),
                device_bytes=device_bytes,
            )
        )
        offset += device_bytes.numel()
    if not segments:
        raise RuntimeError("model has no CUDA runtime storage to mirror")
    return segments, _align_up(offset)


class HostRuntimeState:
    """Own one versioned pinned host image and commit it to live CUDA storage."""

    def __init__(self, model: torch.nn.Module, *, initial_version: str = "0"):
        self.segments, self.image_nbytes = build_runtime_storage_plan(model)
        self.image = torch.empty(
            self.image_nbytes,
            dtype=torch.uint8,
            pin_memory=True,
        )
        self._stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.host_version = initial_version
        self.gpu_version = initial_version
        self.valid = False
        self.prepared = False
        self.invalid_reason: str | None = None
        capture_stats = self.capture_from_model(initial_version)
        logger.info("[RL_HOST_RUNTIME_INIT] %s", capture_stats)
        self.delta_plan = getattr(model, "_runtime_delta_plan", None)
        if self.delta_plan is None:
            raise RuntimeError(
                "host runtime updates were enabled but the initial model loader "
                "did not record a runtime delta plan"
            )
        plan_stats = self.delta_plan.finalize(model, self.segments)
        logger.info("[RL_HOST_RUNTIME_PLAN] %s", plan_stats)

    def _copy_image_to_model(self) -> tuple[int, float]:
        started = time.perf_counter()
        with torch.cuda.stream(self._stream):
            for segment in self.segments:
                begin = segment.image_offset
                segment.device_bytes.copy_(
                    self.image[begin : begin + segment.nbytes],
                    non_blocking=True,
                )
        self._stream.synchronize()
        return len(self.segments), time.perf_counter() - started

    def _copy_model_to_image(self) -> tuple[int, float]:
        started = time.perf_counter()
        with torch.cuda.stream(self._stream):
            for segment in self.segments:
                begin = segment.image_offset
                self.image[begin : begin + segment.nbytes].copy_(
                    segment.device_bytes,
                    non_blocking=True,
                )
        self._stream.synchronize()
        return len(self.segments), time.perf_counter() - started

    def capture_from_model(self, version: str) -> dict[str, int | float | str]:
        """Seed or explicitly recover the host image from active GPU weights."""

        self.valid = False
        self.prepared = False
        self.invalid_reason = f"host-runtime capture of version {version} is incomplete"
        try:
            copies, wall_s = self._copy_model_to_image()
        except Exception as exc:
            self.invalid_reason = (
                f"host-runtime capture of version {version} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        self.host_version = version
        self.gpu_version = version
        self.valid = True
        self.invalid_reason = None
        return self._stats("capture", version, copies, wall_s)

    def begin_prepare(self, *, base_version: str, target_version: str) -> torch.Tensor:
        """Validate the single-image state before mutating it in the background."""

        if not self.valid:
            raise RuntimeError(
                "host runtime image is invalid"
                + (f": {self.invalid_reason}" if self.invalid_reason else "")
            )
        if self.prepared:
            raise RuntimeError(
                f"host runtime version {self.host_version} is already prepared "
                f"while GPU remains at {self.gpu_version}; commit it before pulling "
                "another runtime version"
            )
        if self.host_version != base_version or self.gpu_version != base_version:
            raise RuntimeError(
                "runtime preparation base mismatch: "
                f"requested={base_version}, host={self.host_version}, "
                f"gpu={self.gpu_version}"
            )
        if target_version == base_version:
            raise ValueError("target runtime version must differ from base version")
        return self.image

    def finish_prepare(self, target_version: str) -> None:
        if not self.valid:
            raise RuntimeError("cannot finish preparation of an invalid runtime image")
        self.host_version = target_version
        self.prepared = True

    def prepare_from_deltas(
        self,
        *,
        model: torch.nn.Module,
        source_dir: str,
        target_version: int,
    ) -> dict[str, Any]:
        """Advance the single host image after disk reconstruction succeeded."""

        base_version = int(self.host_version)
        target = str(target_version)
        self.begin_prepare(
            base_version=str(base_version),
            target_version=target,
        )
        try:
            stats = self.delta_plan.apply_versions(
                model=model,
                host_image=self.image,
                source_dir=source_dir,
                base_version=base_version,
                target_version=target_version,
            )
        except Exception as exc:
            self.invalidate(
                f"runtime preparation to v{target_version} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise
        self.finish_prepare(target)
        return stats

    def invalidate(self, reason: str) -> None:
        """Fail closed after a partial or otherwise untrusted host-image update."""

        self.valid = False
        self.prepared = False
        self.invalid_reason = reason

    def commit(self, expected_version: str) -> dict[str, int | float | str]:
        """Copy the complete prepared image into existing CUDA storages."""

        if not self.valid:
            raise RuntimeError(
                "host runtime image is invalid"
                + (f": {self.invalid_reason}" if self.invalid_reason else "")
            )
        if not self.prepared:
            raise RuntimeError("no host runtime version is prepared")
        if self.host_version != expected_version:
            raise RuntimeError(
                "prepared runtime version mismatch: "
                f"requested={expected_version}, prepared={self.host_version}"
            )
        copies, wall_s = self._copy_image_to_model()
        self.gpu_version = expected_version
        self.prepared = False
        return self._stats("commit", expected_version, copies, wall_s)

    def _stats(
        self, operation: str, version: str, copies: int, wall_s: float
    ) -> dict[str, int | float | str]:
        return {
            "operation": operation,
            "version": version,
            "bytes": self.image_nbytes,
            "storages": copies,
            "wall_s": round(wall_s, 6),
            "gbps": round(self.image_nbytes / max(wall_s, 1e-9) / 1e9, 3),
        }
