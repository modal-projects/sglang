from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch

from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    MultiprocessingSerializer,
    dynamic_import,
    get_available_gpu_memory,
    init_custom_process_group,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _unsupported_derived_weight_cache_error(
    model_runner: ModelRunner | None = None,
) -> Optional[str]:
    """Reject online weight updates that derived-weight caches cannot survive.

    These caches are built from loaded weights outside the ordinary post-load
    lifecycle. In-place writes are invisible to them and would silently keep
    serving stale values. Each check is startup-determined and rank-uniform,
    so an update never proceeds on only a subset of workers.
    """
    from sglang.kernels.ops.attention.dsv4.gemm import hpc_bf16xfp32_gemm_enabled

    if hpc_bf16xfp32_gemm_enabled():
        return (
            "Online weight updates are not supported while the HPC-Ops "
            "bf16xfp32 GEMM optimization is enabled: the cached weight "
            "split would keep serving the old weights."
        )
    if model_runner is not None and getattr(
        model_runner.server_args,
        "dcp_replicate_q_proj",
        False,
    ):
        return (
            "Online weight updates are not supported with "
            "--dcp-replicate-q-proj: its full-head MLA weights are derived "
            "once before CUDA graph capture and would become stale."
        )
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightUpdater:
    tp_rank: int
    device: str
    gpu_id: int
    model_config: ModelConfig
    custom_weight_loaders: dict
    get_model: Callable[[], Any]
    update_model_fields: Callable[..., None]
    recapture_cuda_graph: Callable[[], None]
    get_model_runner: Callable[[], ModelRunner]
    _model_update_group: dict = field(default_factory=dict)

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def _assert_in_place_update_allowed(self: WeightUpdater, op: str) -> None:
        """Reject mutations that would invalidate an active weight cache."""
        runner = self.get_model_runner()
        mode = runner.server_args.weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} is not supported while the weight cache is "
                f"active (--weight-cache-mode {mode}): model weights are shared "
                f"with the daemon via CUDA IPC, so mutating them in place would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )
        if runner.cpu_weight_cache is not None:
            raise RuntimeError(
                f"{op} is not supported while the CPU weight cache is active: "
                "an out-of-band update would make its canonical checkpoint and "
                "rank-ready image stale. Discard the CPU weight cache first."
            )

    def update_weights_from_disk(
        self: WeightUpdater,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        self._assert_in_place_update_allowed("update_weights_from_disk")
        error = _unsupported_derived_weight_cache_error(self.get_model_runner())
        if error is not None:
            return False, error

        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        model = self.get_model()
        runner = self.get_model_runner()
        original_model_path = self.model_config.model_path
        original_load_config = runner.load_config

        if (
            weight_name_filter is not None
            and self.model_config.quantization is not None
        ):
            return False, (
                "weight_name_filter is not supported for quantized models: "
                "post-load processing requires every checkpoint-facing weight."
            )

        self.model_config.model_path = model_path
        load_config = replace(original_load_config, load_format=load_format)

        try:
            loader = get_model_loader(load_config, self.model_config)
        except Exception as e:
            self.model_config.model_path = original_model_path
            return False, f"Failed to get model loader: {e}."
        if not isinstance(loader, DefaultModelLoader):
            self.model_config.model_path = original_model_path
            message = f"Failed to get model loader: {loader}."
            return False, message

        def load_weights(
            active_loader: DefaultModelLoader,
            *,
            weight_filter: Optional[Callable[[str], bool]],
        ) -> None:
            active_loader.restore_weights_before_loading(model, target_device)
            weights = active_loader._get_all_weights(self.model_config, model)
            if weight_filter is not None:
                weights = (
                    (name, weight) for name, weight in weights if weight_filter(name)
                )
            active_loader.load_weights_and_postprocess(model, weights, target_device)

        def rollback() -> None:
            self.model_config.model_path = original_model_path
            original_loader = get_model_loader(
                original_load_config,
                self.model_config,
            )
            if not isinstance(original_loader, DefaultModelLoader):
                raise TypeError(
                    "original model loader does not support in-place rollback: "
                    f"{original_loader}"
                )
            load_weights(original_loader, weight_filter=None)

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                load_weights(loader, weight_filter=weight_name_filter)
            except Exception as e:
                message = f"Failed to update weights: {e}."
                gc.collect()
                if os.path.realpath(original_model_path) == os.path.realpath(
                    model_path
                ):
                    raise RuntimeError(
                        "Weight reload failed after mutating the active checkpoint "
                        "path; terminating the engine because the previous weights "
                        f"are no longer available for rollback. Reload error: {e}."
                    ) from e
                try:
                    rollback()
                except Exception as rollback_error:
                    logger.exception("Failed to roll back model weights")
                    raise RuntimeError(
                        "Weight reload and rollback both failed; terminating the "
                        "engine to avoid serving a partially updated model. "
                        f"Reload error: {e}. Rollback error: {rollback_error}."
                    ) from rollback_error
                message += " Rolled back to the original weights."
                return False, message

        self.update_model_fields(
            model,
            model_path=model_path,
            load_format=load_format,
            load_config=load_config,
        )

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.recapture_cuda_graph()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    @torch.no_grad()
    def initialize_cpu_weight_cache(
        self,
        *,
        checkpoint_dir: str,
        seed_from_active_weights: bool,
        base_version: int,
        host_group: torch.distributed.ProcessGroup | None,
        max_compile_group_bytes: int,
        canonical_checkpoint_dir: str | None,
    ) -> dict[str, Any]:
        """Construct and populate the target model's CPU weight cache."""

        runner = self.get_model_runner()
        if self.device != "cuda":
            raise RuntimeError("CPU weight caching requires a CUDA model runner")
        if runner.is_draft_worker:
            raise RuntimeError("CPU weight caching is only supported for target models")
        if max_compile_group_bytes <= 0:
            raise ValueError("max_compile_group_bytes must be positive")
        if runner.cpu_weight_cache is not None:
            raise RuntimeError("CPU weight cache is already initialized")
        self._assert_in_place_update_allowed("initialize_cpu_weight_cache")
        error = _unsupported_derived_weight_cache_error(runner)
        if error is not None:
            raise RuntimeError(error)

        from sglang.srt.weight_sync.cpu_weight_cache import CPUWeightCache

        distributed = torch.distributed.is_initialized()
        host_world_size = (
            torch.distributed.get_world_size(group=host_group) if distributed else 1
        )
        host_rank = torch.distributed.get_rank(group=host_group) if distributed else 0
        if host_world_size > 1 and host_group is None:
            raise RuntimeError("CPU weight caching requires a host-local process group")

        started = time.perf_counter()
        cache = None
        construction_error = None
        with torch.cuda.device(self.gpu_id):
            try:
                cache = CPUWeightCache(
                    self.get_model(),
                    max_compile_group_bytes=max_compile_group_bytes,
                    host_group=host_group,
                    canonical_checkpoint_dir=canonical_checkpoint_dir,
                )
            except Exception as exc:
                construction_error = f"rank {host_rank}: {type(exc).__name__}: {exc}"

            if host_world_size > 1:
                construction_errors: list[str | None] = [None] * host_world_size
                torch.distributed.all_gather_object(
                    construction_errors,
                    construction_error,
                    group=host_group,
                )
            else:
                construction_errors = [construction_error]
            errors = [error for error in construction_errors if error is not None]
            if errors:
                if cache is not None:
                    cache.close("distributed CPU weight cache construction failed")
                raise RuntimeError(
                    "CPU weight cache construction failed: " + "; ".join(errors)
                )
            if cache is None:
                raise RuntimeError(
                    "CPU weight cache construction completed without a cache"
                )

            construction_wall_s = time.perf_counter() - started
            try:
                stats = cache.initialize_from_checkpoint(
                    checkpoint_dir,
                    seed_from_active_weights=seed_from_active_weights,
                    base_version=base_version,
                )
            except Exception:
                cache.close("CPU weight cache initialization failed")
                raise

        runner.cpu_weight_cache = cache
        stats["cache_population_wall_s"] = stats["wall_s"]
        stats["cache_construction_wall_s"] = round(construction_wall_s, 6)
        stats["wall_s"] = round(time.perf_counter() - started, 6)
        return stats

    @torch.no_grad()
    def stage_cpu_weight_update_from_delta_lineage(
        self,
        *,
        checkpoint_source_dir: str,
        target_version: int,
        host_group: torch.distributed.ProcessGroup | None,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Compile a complete target image without changing live weights."""

        try:
            runner = self.get_model_runner()
            cache = runner.cpu_weight_cache
            if cache is None:
                raise RuntimeError("CPU weight cache is not initialized")
            if cache.host_group is not host_group:
                raise ValueError(
                    "host-local process group cannot change after cache initialization"
                )
            with torch.cuda.device(self.gpu_id):
                stats = cache.stage_delta_lineage(
                    checkpoint_source_dir=checkpoint_source_dir,
                    target_version=target_version,
                )
            logger.info(
                "Staged CPU weights for version %d: bytes=%d groups=%d "
                "wall_time=%.3fs",
                target_version,
                stats["compile"]["bytes"],
                stats["compile"]["groups"],
                stats["wall_s"],
            )
            return True, "Staged weights in CPU memory.", stats
        except Exception as exc:
            logger.exception(
                "Failed to stage CPU weights for version %s",
                target_version,
            )
            return (
                False,
                f"Failed to stage CPU weights: {type(exc).__name__}: {exc}",
                None,
            )

    def validate_staged_cpu_weight_update(
        self,
        target_version: int,
    ) -> tuple[bool, str]:
        cache = self.get_model_runner().cpu_weight_cache
        if cache is None:
            return False, "CPU weight cache is not initialized."
        try:
            cache.image.validate_commit(target_version)
            return True, "CPU weights are ready."
        except Exception as exc:
            return (
                False,
                f"CPU weights are not ready: {type(exc).__name__}: {exc}",
            )

    @torch.no_grad()
    def update_weights_from_cpu(
        self,
        target_version: int,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Copy a prepared rank image into the existing CUDA storages."""

        cache = self.get_model_runner().cpu_weight_cache
        if cache is None:
            return False, "CPU weight cache is not initialized.", None
        try:
            with torch.cuda.device(self.gpu_id):
                torch.cuda.synchronize(self.gpu_id)
                stats = cache.commit(target_version)
            logger.info(
                "Updated weights from CPU for version %d: bytes=%d "
                "wall_time=%.3fs bandwidth=%.3fGB/s",
                target_version,
                stats["bytes"],
                stats["wall_s"],
                stats["gbps"],
            )
            return True, "Updated weights from CPU memory.", stats
        except Exception:
            # A failed commit may have overwritten only part of the live model;
            # recovery requires restarting or fully reloading the worker.
            logger.critical(
                "CPU weight update failed for version %s",
                target_version,
                exc_info=True,
            )
            raise

    def invalidate_staged_cpu_weight_update(self, reason: str) -> None:
        cache = self.get_model_runner().cpu_weight_cache
        if cache is not None:
            cache.invalidate_stage(reason)

    def discard_cpu_weight_cache(self, reason: str) -> None:
        runner = self.get_model_runner()
        cache = runner.cpu_weight_cache
        runner.cpu_weight_cache = None
        if cache is not None:
            cache.close(reason)

    def update_weights_from_distributed(
        self: WeightUpdater,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        self._assert_in_place_update_allowed("update_weights_from_distributed")
        error = _unsupported_derived_weight_cache_error(self.get_model_runner())
        if error is not None:
            return False, error

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.get_model().load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self: WeightUpdater, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (
                        name,
                        torch.empty(shape, dtype=target_dtype, device=self.device),
                    )
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.get_model().load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self: WeightUpdater,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        error = _unsupported_derived_weight_cache_error(self.get_model_runner())
        if error is not None:
            return False, error

        monkey_patch_torch_reductions()
        self._assert_in_place_update_allowed("update_weights_from_tensor")
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.get_model(), named_tensors)
        elif load_format in self.custom_weight_loaders:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.get_model(), named_tensors)
        elif load_format is None:
            self.get_model().load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self: WeightUpdater,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.get_model().load_weights(reconstructed_tensors)

        return True, "Success"

    def update_weights_from_ipc(self: WeightUpdater, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        self._assert_in_place_update_allowed("update_weights_from_ipc")
        error = _unsupported_derived_weight_cache_error(self.get_model_runner())
        if error is not None:
            return False, error

        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self.get_model_runner())
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
