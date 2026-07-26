"""Rank-local CPU images for full-model weight updates.

The image stores the final byte layout of every model-owned CUDA weight
storage. Background staging produces a complete image; the paused update
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

_SEGMENT_ALIGNMENT = 64
_IMAGE_ALIGNMENT = 4096
_COPY_CHUNK_BYTES = 256 << 20


class IrrecoverableWeightCommitError(RuntimeError):
    """A CPU-to-GPU commit failed after it may have changed live weights."""


@dataclass(frozen=True)
class CPUWeightSegment:
    """One unique live CUDA storage and its range in the host image."""

    name: str
    image_offset: int
    nbytes: int
    device_bytes: torch.Tensor


def _align_up(value: int, alignment: int) -> int:
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
    ``get_additional_weight_tensors()`` rather than making this code guess which
    arbitrary CUDA attributes are safe to overwrite.
    """

    yield from model.named_parameters(remove_duplicate=False)
    for module_name, module in model.named_modules(remove_duplicate=False):
        prefix = f"{module_name}." if module_name else ""
        for name, tensor in module._buffers.items():
            if tensor is not None and name not in module._non_persistent_buffers_set:
                yield f"{prefix}{name}", tensor
    for module_name, module in model.named_modules():
        get_extra = getattr(module, "get_additional_weight_tensors", None)
        if get_extra is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        for name, tensor in get_extra():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "get_additional_weight_tensors() must yield (str, torch.Tensor)"
                )
            yield f"{prefix}{name}", tensor


def build_cpu_weight_image_plan(
    model: torch.nn.Module,
    *,
    device_type: str = "cuda",
) -> tuple[list[CPUWeightSegment], int]:
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
    segments: list[CPUWeightSegment] = []
    for name, tensor in sorted(unique.values(), key=lambda item: item[0]):
        offset = _align_up(offset, _SEGMENT_ALIGNMENT)
        device_bytes = _storage_as_bytes(tensor)
        segments.append(
            CPUWeightSegment(
                name=name,
                image_offset=offset,
                nbytes=device_bytes.numel(),
                device_bytes=device_bytes,
            )
        )
        offset += device_bytes.numel()
    if not segments:
        raise RuntimeError(f"model has no {device_type} weight storage")
    return segments, _align_up(offset, _IMAGE_ALIGNMENT)


class CPUWeightImage:
    """Own one rank-local CPU image and commit it to live CUDA weights."""

    def __init__(self, model: torch.nn.Module):
        self.segments, self.image_nbytes = build_cpu_weight_image_plan(model)
        self.weight_nbytes = sum(segment.nbytes for segment in self.segments)
        segments_by_storage = {
            (
                segment.device_bytes.device.index,
                segment.device_bytes.data_ptr(),
                segment.nbytes,
            ): segment
            for segment in self.segments
        }
        self._segments_by_device_storage = segments_by_storage
        self.segments_by_name: dict[str, CPUWeightSegment] = {}
        for name, tensor in iter_weight_tensors(model):
            if tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
            self.segments_by_name[name] = segments_by_storage[key]
        self.image = torch.empty(self.image_nbytes, dtype=torch.uint8)
        self._image_buffer = memoryview(self.image.numpy()).cast("B")
        self._stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._post_commit_hooks = [
            (module, hook)
            for module in model.modules()
            if (quant_method := getattr(module, "quant_method", None)) is not None
            and callable(
                hook := getattr(
                    quant_method,
                    "process_weights_after_weight_commit",
                    None,
                )
            )
        ]
        self.registered = False
        self.registration_wall_s: float | None = None
        self.target_version: int | None = None
        self.staged = False
        self.staging = False
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
            raise KeyError("tensor storage is not part of the CPU weight image")
        begin = segment.image_offset
        end = begin + segment.nbytes
        return torch.frombuffer(
            self._image_buffer[begin:end],
            dtype=torch.uint8,
        )

    def capture_active_weights(
        self,
    ) -> dict[str, int | float | str | bool | None]:
        """Seed the image from the active model while generation may continue."""

        self.invalidate("active weight capture is incomplete")
        started = time.perf_counter()
        try:
            self.copy_device_segments_to_image(
                (segment, segment.device_bytes) for segment in self.segments
            )
        except Exception as exc:
            self.invalid_reason = (
                f"active weight capture failed: {type(exc).__name__}: {exc}"
            )
            raise

        self.target_version = None
        self.staged = False
        self.staging = False
        self.valid = True
        self.invalid_reason = None
        return self.stats("capture_active_weights", time.perf_counter() - started)

    def begin_stage(self, target_version: int) -> None:
        """Mark the image unsafe until a complete target has been staged."""

        if self.staging:
            raise RuntimeError("a CPU weight image stage is already running")
        self.invalidate(f"staging of version {target_version} is incomplete")
        self.target_version = target_version
        self.staging = True

    def finish_stage(self, target_version: int) -> None:
        if not self.staging:
            raise RuntimeError("no CPU weight image stage is running")
        if self.target_version != target_version:
            raise RuntimeError(
                "staged version changed while staging: "
                f"expected={target_version}, actual={self.target_version}"
            )
        self.valid = True
        self.staged = True
        self.staging = False
        self.invalid_reason = None

    def accept_staged_baseline(self) -> None:
        """Keep a fully compiled baseline image without exposing a commit."""

        if not self.valid or not self.staged or self.staging:
            raise RuntimeError("no complete staged image can become the baseline")
        self.target_version = None
        self.staged = False
        self.invalid_reason = None

    def invalidate(self, reason: str) -> None:
        """Fail closed after a partial or otherwise untrusted stage."""

        self.valid = False
        self.staged = False
        self.staging = False
        self.invalid_reason = reason

    def _chunks(
        self,
        segment: CPUWeightSegment,
        device_bytes: torch.Tensor,
    ):
        if device_bytes.device.type != "cuda":
            raise ValueError(f"staged source for {segment.name!r} is not a CUDA tensor")
        if device_bytes.dtype != torch.uint8 or device_bytes.ndim != 1:
            raise ValueError(
                f"staged source for {segment.name!r} is not a flat byte view"
            )
        if device_bytes.numel() != segment.nbytes:
            raise ValueError(
                "staged source size mismatch: "
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
            return self.stats("register_cpu_weight_image", 0.0)
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
        return self.stats("register_cpu_weight_image", self.registration_wall_s)

    def copy_device_segments_to_image(
        self,
        segments: Iterable[tuple[CPUWeightSegment, torch.Tensor]],
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

    def validate_commit(self, target_version: int) -> None:
        """Validate a requested commit without changing live weights."""
        if not self.valid:
            raise RuntimeError(
                "staged weight image is invalid"
                + (f": {self.invalid_reason}" if self.invalid_reason else "")
            )
        if not self.staged:
            raise RuntimeError("no CPU weight image is staged")
        if self.target_version != target_version:
            raise RuntimeError(
                "staged weight version mismatch: "
                f"expected={target_version}, staged={self.target_version}"
            )
        if not self.registered:
            raise RuntimeError(
                "staged CPU image is not registered; registration must "
                "finish before the engine pauses"
            )

    def commit(self, target_version: int) -> dict[str, int | float | str | bool | None]:
        """Overwrite every live weight storage from the staged CPU image."""

        self.validate_commit(target_version)
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
            hook_started = time.perf_counter()
            for module, hook in self._post_commit_hooks:
                hook(module)
            torch.cuda.synchronize(self.segments[0].device_bytes.device)
            post_commit_hook_wall_s = time.perf_counter() - hook_started
        except Exception as exc:
            self.invalidate(
                f"commit of version {target_version} failed after live weights may "
                f"have been partially overwritten: {type(exc).__name__}: {exc}"
            )
            raise IrrecoverableWeightCommitError(self.invalid_reason) from exc

        self.staged = False
        stats = self.stats("commit_cpu_weight_image", time.perf_counter() - started)
        stats["post_commit_hook_wall_s"] = round(post_commit_hook_wall_s, 6)
        return stats

    def validate_against_active(self) -> dict[str, int | float | str]:
        """Validate the compiled image against the active startup weights."""

        if not self.valid or not self.staged or not self.registered:
            raise RuntimeError("no complete CPU weight image is available")
        started = time.perf_counter()
        scratch = torch.empty(
            min(_COPY_CHUNK_BYTES, self.image_nbytes),
            dtype=torch.uint8,
            device=self.segments[0].device_bytes.device,
        )
        segment_mismatches = torch.zeros(
            len(self.segments),
            dtype=torch.bool,
            device=scratch.device,
        )
        compared_bytes = 0
        with torch.cuda.stream(self._stream):
            for segment_index, segment in enumerate(self.segments):
                for image_chunk, device_chunk in self._chunks(
                    segment, segment.device_bytes
                ):
                    nbytes = image_chunk.numel()
                    scratch[:nbytes].copy_(image_chunk)
                    segment_mismatches[segment_index].logical_or_(
                        torch.any(
                            scratch[:nbytes] != device_chunk,
                        )
                    )
                    compared_bytes += nbytes
        self._stream.synchronize()
        mismatches = torch.nonzero(segment_mismatches).flatten().tolist()
        if mismatches:
            segment = self.segments[mismatches[0]]
            raise RuntimeError(
                "CPU weight cache does not reproduce the active model: "
                f"storage={segment.name!r}"
            )
        return {
            "operation": "validate_cpu_weight_image",
            "bytes": compared_bytes,
            "storages": len(self.segments),
            "wall_s": round(time.perf_counter() - started, 6),
        }

    def stats(
        self,
        operation: str,
        wall_s: float,
    ) -> dict[str, int | float | str | bool | None]:
        transferred_bytes = (
            self.weight_nbytes
            if operation in {"capture_active_weights", "commit_cpu_weight_image"}
            else 0
        )
        return {
            "operation": operation,
            "target_version": self.target_version,
            "bytes": transferred_bytes,
            "weight_bytes": self.weight_nbytes,
            "allocated_bytes": self.image_nbytes,
            "storages": len(self.segments),
            "registered": self.registered,
            "registration_wall_s": round(self.registration_wall_s or 0.0, 6),
            "wall_s": round(wall_s, 6),
            "gbps": round(transferred_bytes / max(wall_s, 1e-9) / 1e9, 3),
        }

    def close(self) -> None:
        if self.registered:
            try:
                error = torch.cuda.cudart().cudaHostUnregister(self.image.data_ptr())
                if int(error) != 0:
                    logger.warning("cudaHostUnregister failed: %s", error)
            except Exception:
                logger.exception("Failed to unregister CPU weight image")
            self.registered = False
        image_buffer = getattr(self, "_image_buffer", None)
        if image_buffer is not None:
            try:
                image_buffer.release()
                del self._image_buffer
            except BufferError:
                logger.warning("CPU weight image still has exported buffer views")
