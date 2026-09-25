"""DFlash/DFlash2 draft-only transactions; existing draft KV is preserved.

Control requests run on the scheduler thread, between complete draft/verify
launches. Device fences cover outstanding overlap/graph work. All ranks stage
and validate on CPU before any live parameter is written. Parameters and fused
KV helper storage keep their addresses, so captured graphs remain usable.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.weight_sync.draft_delta import (
    DTYPES,
    config_hash,
    decode_tensor,
    validate_manifest,
)


@dataclass(frozen=True)
class WeightSlice:
    name: str
    parameter: str
    offset: int = 0
    length: int | None = None
    shard_dim: int | None = None
    replicas: int = 1

    def view(self, state):
        value = state[self.parameter]
        return (
            value if self.length is None else value.narrow(0, self.offset, self.length)
        )

    def shape(self, state, world_size):
        shape = list(self.view(state).shape)
        if self.shard_dim is not None:
            shape[self.shard_dim] *= world_size // self.replicas
        return shape

    def store(self, state, full, rank, world_size):
        destination = self.view(state)
        if self.shard_dim is not None:
            full = full.narrow(
                self.shard_dim,
                (rank // self.replicas) * destination.shape[self.shard_dim],
                destination.shape[self.shard_dim],
            )
        destination.copy_(full)


def draft_weight_layout(model, rank, world_size):
    """Map canonical export names to existing unquantized TP parameter slices."""
    if type(model).__name__ not in ("DFlashDraftModel", "DFlash2DraftModel"):
        raise ValueError("draft deltas support DFlashDraftModel and DFlash2DraftModel")
    if getattr(model, "embed_tokens", None) is not None or model.draft_config.is_domino:
        raise ValueError(
            "draft deltas require the shared-target embedding DFlash layout"
        )
    modules = dict(model.named_modules())
    parameters, layout = {}, []
    for name, param in model.named_parameters():
        # DFlash2 attaches the target head as a module before graph capture.
        if any(part in ("lm_head", "embed_tokens") for part in name.split(".")):
            continue
        if (
            str(param.dtype).removeprefix("torch.") not in DTYPES
            or not param.is_contiguous()
        ):
            raise ValueError(
                "draft deltas require contiguous BF16/FP16/FP32 parameters"
            )
        module_name, _, suffix = name.rpartition(".")
        module = modules[module_name]
        if getattr(module, "quant_config", None) is not None:
            raise ValueError("quantized draft deltas are unsupported")
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)) and (
            module.tp_rank != rank or module.tp_size != world_size
        ):
            raise ValueError("draft layer TP layout differs from the update group")
        parameters[name] = param
        if isinstance(module, QKVParallelLinear):
            if module.kv_tp_size != world_size or module.kv_tp_rank != rank:
                raise ValueError("separate KV parallel groups are unsupported")
            offset = 0
            for part, length, replicas in (
                ("q", module.q_proj_shard_size, 1),
                ("k", module.kv_proj_shard_size, module.num_kv_head_replicas),
                ("v", module.v_proj_shard_size, module.num_kv_head_replicas),
            ):
                layout.append(
                    WeightSlice(
                        name.replace("qkv_proj", f"{part}_proj"),
                        name,
                        offset,
                        length,
                        0,
                        replicas,
                    )
                )
                offset += length
            if offset != param.shape[0]:
                raise ValueError("unexpected packed QKV layout")
        elif isinstance(module, MergedColumnParallelLinear):
            if len(module.output_sizes) != 2 or not module_name.endswith(
                "gate_up_proj"
            ):
                raise ValueError("unexpected merged draft projection")
            offset = 0
            for part, size in zip(("gate", "up"), module.output_sizes, strict=True):
                length = size // world_size
                layout.append(
                    WeightSlice(
                        name.replace("gate_up_proj", f"{part}_proj"),
                        name,
                        offset,
                        length,
                        0,
                    )
                )
                offset += length
            if offset != param.shape[0]:
                raise ValueError("unexpected packed MLP layout")
        else:
            dim = None
            if isinstance(module, ColumnParallelLinear):
                dim = 0
            elif isinstance(module, RowParallelLinear) and suffix == "weight":
                dim = 1
            layout.append(WeightSlice(name, name, shard_dim=dim))
    return parameters, sorted(layout, key=lambda item: item.name)


class DFlashWeightUpdater:
    def __init__(self, model, group, refresh, synchronize, *, config_sha256=None):
        self.model, self.group = model, group
        self.refresh, self.synchronize = refresh, synchronize
        self.rank = dist.get_rank(group) if group is not None else 0
        self.world_size = dist.get_world_size(group) if group is not None else 1
        self.version = 0
        self.weights_id = None
        self.initial_weights_id = None
        self.initial_state = None
        self.last_base_weights_id = None
        self.config_sha256 = config_sha256 or config_hash(model.config.to_dict())

    def state(self):
        return {
            "draft_version": self.version,
            "draft_weights_id": self.weights_id,
            "initial_weights_id": self.initial_weights_id,
            "config_sha256": self.config_sha256,
            "tp_size": self.world_size,
            "draft_kv_policy": "preserve",
        }

    def _all(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def _agree(self, error):
        errors = self._all(error)
        if any(errors):
            raise ValueError(
                " | ".join(f"TP{rank}: {e}" for rank, e in enumerate(errors) if e)
            )

    def _full_tensor(self, item, state):
        error, local, raw, gathered = None, None, None, None
        try:
            local = item.view(state).contiguous()
            if item.shard_dim is not None and self.world_size > 1:
                raw = local.reshape(-1).view(torch.uint8)
                gathered = [torch.empty_like(raw) for _ in range(self.world_size)]
        except Exception as exc:
            error = str(exc)
        self._agree(error)
        if gathered is None:
            return local
        dist.all_gather(gathered, raw, group=self.group)
        shards = [raw.view(local.dtype).reshape(local.shape) for raw in gathered]
        for start in range(0, self.world_size, item.replicas):
            if any(
                not torch.equal(shards[start], shards[i])
                for i in range(start + 1, start + item.replicas)
            ):
                raise ValueError(
                    f"replicated KV weights differ across TP ranks: {item.name}"
                )
        return torch.cat(shards[:: item.replicas], dim=item.shard_dim)

    def _frame(self, handle, entry):
        # Only the TP leader accesses the server-side upload file. Other engine
        # nodes receive real bytes through Gloo, never a client filesystem path.
        error, data = None, None
        if self.rank == 0:
            try:
                data = handle.read(entry["length"])
                if len(data) != entry["length"]:
                    raise ValueError("truncated draft delta frame")
            except Exception as exc:
                error = str(exc)
        self._agree(error)
        if self.world_size == 1 or not entry["length"]:
            return data or b""
        error, raw = None, None
        try:
            raw = (
                torch.frombuffer(bytearray(data), dtype=torch.uint8)
                if self.rank == 0
                else torch.empty(entry["length"], dtype=torch.uint8)
            )
        except Exception as exc:
            error = str(exc)
        self._agree(error)
        source = dist.get_global_rank(self.group, 0)
        dist.broadcast(raw, src=source, group=self.group)
        return raw.numpy()

    def _validate_schema(self, manifest, parameters, layout):
        entries = validate_manifest(manifest)
        if manifest["config_sha256"] != self.config_sha256:
            raise ValueError("draft model configuration mismatch")
        if [entry["name"] for entry in entries] != [item.name for item in layout]:
            raise ValueError(
                "delta must describe every canonical draft tensor exactly once"
            )
        for item, entry in zip(layout, entries, strict=True):
            if entry["shape"] != item.shape(parameters, self.world_size) or entry[
                "dtype"
            ] != str(parameters[item.parameter].dtype).removeprefix("torch."):
                raise ValueError(f"draft tensor schema mismatch: {item.name}")

    def _stage(self, request, parameters, layout):
        manifest, error = request.manifest, None
        entries = manifest["delta"]["tensors"]
        backup, staged, handle = None, None, None
        try:
            # The scheduler is not launching work while this handler runs. Fence
            # previously queued target/draft work before snapshots or mutation.
            self.synchronize()
            backup = {
                name: param.detach().to("cpu", copy=True)
                for name, param in parameters.items()
            }
            staged = {name: torch.empty_like(value) for name, value in backup.items()}
            if self.rank == 0:
                handle = Path(request.payload_path).open("rb")
                if (
                    Path(request.payload_path).stat().st_size
                    != manifest["delta"]["compressed_bytes"]
                ):
                    raise ValueError("draft delta payload size mismatch")
        except Exception as exc:
            error = str(exc)
        try:
            self._agree(error)
            base_id = manifest["delta"]["base_weights_id"]
            if self.version == 0 or base_id == self.weights_id:
                base = backup
            elif base_id == self.initial_weights_id:
                base = self.initial_state
            else:
                raise ValueError(
                    "delta base must be the active or initial draft weights"
                )
            for item, entry in zip(layout, entries, strict=True):
                compressed = self._frame(handle, entry)
                # All ranks enter the same gathers, including when validation of
                # this tensor fails on only one replica.
                error = None
                try:
                    full = self._full_tensor(item, base)
                    result = decode_tensor(full, entry, compressed)
                    item.store(staged, result, self.rank, self.world_size)
                except Exception as exc:
                    error = str(exc)
                self._agree(error)
            return backup, staged
        finally:
            if handle is not None:
                handle.close()

    @torch.no_grad()
    def handle(self, request):
        error, parameters, layout = None, None, None
        try:
            parameters, layout = draft_weight_layout(
                self.model, self.rank, self.world_size
            )
            if request.action not in ("status", "apply"):
                raise ValueError("unknown draft update action")
            if request.action == "apply":
                self._validate_schema(request.manifest, parameters, layout)
                version = request.expected_draft_version
                if type(version) is not int or version < 0:
                    raise ValueError(
                        "expected_draft_version must be a nonnegative integer"
                    )
        except Exception as exc:
            error = str(exc)
        try:
            self._agree(error)
            states = self._all(self.state())
            if any(state != states[0] for state in states):
                raise ValueError("draft update state differs across TP ranks")
            if request.action == "status":
                return {
                    "success": True,
                    "message": "Draft update state",
                    **self.state(),
                }
            manifest = request.manifest
            if (
                request.expected_draft_version == self.version - 1
                and manifest["weights_id"] == self.weights_id
                and manifest["delta"]["base_weights_id"] == self.last_base_weights_id
            ):
                return {
                    "success": True,
                    "message": "Already applied",
                    "already_applied": True,
                    **self.state(),
                }
            if request.expected_draft_version != self.version:
                raise ValueError(
                    f"draft version conflict: expected {request.expected_draft_version}, active {self.version}"
                )
            error, backup, staged = None, None, None
            try:
                backup, staged = self._stage(request, parameters, layout)
            except Exception as exc:
                error = str(exc)
            self._agree(error)
            try:
                for name, value in staged.items():
                    parameters[name].copy_(value)
                self.refresh()
                self.synchronize()
                error = None
            except Exception as exc:
                error = str(exc)
            failures = self._all(error)
            if any(failures):
                # A recoverable copy/helper failure restores every rank. CUDA
                # context/process failures are fatal; never resume mixed weights.
                rollback_error = None
                try:
                    for name, value in backup.items():
                        parameters[name].copy_(value)
                    self.refresh()
                    self.synchronize()
                except Exception as exc:
                    rollback_error = str(exc)
                if any(self._all(rollback_error)):
                    raise RuntimeError(
                        "draft update rollback failed; restart this engine"
                    )
                raise ValueError("draft update rolled back: " + str(failures))
            if self.version == 0:
                self.initial_state = backup
                self.initial_weights_id = manifest["delta"]["base_weights_id"]
            self.version += 1
            self.weights_id = manifest["weights_id"]
            self.last_base_weights_id = manifest["delta"]["base_weights_id"]
            return {
                "success": True,
                "message": "Draft weights and fused KV weights updated; existing KV preserved",
                **self.state(),
            }
        except ValueError as exc:
            return {"success": False, "message": str(exc), **self.state()}
