# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Megatron-style LayerNorm sequence parallelism (SP, arXiv:2205.05198).

Under pure tensor parallelism the row-parallel ``all_reduce`` is algebraically a
``reduce_scatter`` (g-bar) followed by an ``all_gather`` (g). Splitting it that
way lets the LayerNorm / residual regions run on sequence-sharded activations --
each rank holds 1/tp of the tokens -- which cuts the transient activation memory
of long-context prefill with no extra communication volume (all_reduce and
reduce_scatter+all_gather move the same bytes).

The shared SP machinery lives here, decoupled from ``dp_attention.py``:

  - which models opt in and config validation,
  - the per-forward ``sp_active`` flag (a ForwardFlags bool) read at depth by the
    participant linears and the layer-boundary batch selector,
  - the entry-scatter / exit-gather collectives, and
  - the fused matmul + collective fast-paths for the participant linears.

Models in ``MODEL_COLLECTIVE_SP_ARCHITECTURES`` place their own collectives
around attention and the FFN, and the participant linears stay unsharded for them.

SP runs for prefill (EXTEND), plus mixed batches for those models. It is off by
default, and with the flag off nothing in this module executes.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
    get_forward,
    get_parallel,
)
from sglang.srt.utils.common import ceil_align

# Architectures whose decoder layers route attention/MLP through
# layer boundaries with the standard participant linears, and for which SP
# has been validated. Other models reject --enable-layernorm-sp at construction.
# The mechanism is generic; extend the allowlist as families are validated.
MUSE_SP_ARCHITECTURES = frozenset({"MuseSparkForConditionalGeneration"})
MODEL_COLLECTIVE_SP_ARCHITECTURES = frozenset({"KimiK3ForConditionalGeneration"})
SP_SUPPORTED_ARCHITECTURES = (
    MUSE_SP_ARCHITECTURES
    | MODEL_COLLECTIVE_SP_ARCHITECTURES
    | frozenset({"Qwen3ForCausalLM"})
)


def is_muse_spark_architecture(architecture: Optional[str]) -> bool:
    return architecture in MUSE_SP_ARCHITECTURES


def initialize_layernorm_sp(*, model_config) -> None:
    """Materialize ``flags.sp.enabled``; runs once per worker after distributed
    setup, alongside ``initialize_dp_attention``."""
    architectures = model_config.hf_config.architectures
    flags = get_flags().sp
    flags.enabled = bool(
        get_parallel().enable_layernorm_sp
        and architectures
        and architectures[0] in SP_SUPPORTED_ARCHITECTURES
    )
    flags.model_collectives = (
        flags.enabled and architectures[0] in MODEL_COLLECTIVE_SP_ARCHITECTURES
    )
    if flags.model_collectives:
        flags.min_tokens = _largest_prefill_graph_tokens() + 1


def _largest_prefill_graph_tokens() -> int:
    """Token count of the largest captured prefill CUDA graph, 0 without one.

    Models that place their own collectives decide SP in Python per forward, and a
    replayed prefill graph cannot switch to the SP layout, so they shard only
    forwards larger than this.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.utils.common import get_cuda_graph_max_batch_size

    prefill = get_exec().graph.cuda_graph_config.prefill
    if prefill.backend == Backend.DISABLED or not prefill.bs:
        return 0
    return max(get_cuda_graph_max_batch_size(n) for n in prefill.bs)


def layernorm_sp_enabled() -> bool:
    return get_flags().sp.enabled


def runs_sp(forward_mode, num_tokens: Optional[int] = None) -> bool:
    """Whether this forward of ``num_tokens`` tokens runs SP: an enabled model,
    prefill (or a mixed batch for a model that places its own collectives), and
    at least ``flags.sp.min_tokens`` tokens.

    Code outside the CUDA-graph-captured region must use this and not
    ``get_forward().sp_active``: Python writes made inside that region do not
    re-execute on graph replay, so the flag is stale there.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if not layernorm_sp_enabled():
        return False
    flags = get_flags().sp
    if forward_mode != ForwardMode.EXTEND and not (
        flags.model_collectives and forward_mode == ForwardMode.MIXED
    ):
        return False
    if not flags.min_tokens:
        return True
    if num_tokens is None:
        raise ValueError("runs_sp needs the token count under a token minimum")
    return num_tokens >= flags.min_tokens


def participants_active() -> bool:
    """Whether the participant linears gather and reduce-scatter themselves."""
    return get_forward().sp_active and not get_flags().sp.model_collectives


class _SPForwardState:
    """Real (unpadded) token count of the current SP forward.

    An instance attribute, not a ForwardFlags int slot: this is read inside
    torch.compile-traced linear code, where dynamo gives an attribute-source int
    automatic-dynamic, while a dict-slot int recompiles per sequence length.
    """

    num_tokens: int = 0


_sp_state = _SPForwardState()


def set_sp_num_tokens(num_tokens: int) -> None:
    _sp_state.num_tokens = num_tokens


def sp_num_tokens() -> int:
    return _sp_state.num_tokens


# --- entry scatter / exit gather (once per forward, at the boundary) ----------
def sp_entry_scatter(hidden_states: torch.Tensor) -> torch.Tensor:
    """Shard the replicated ``[M, h]`` hidden states along the token dim.

    Pads M up to a multiple of tp_size; the padding rows are dropped by the exit
    gather. The input is replicated across the TP group, so this is a local slice.
    """
    num_tokens = hidden_states.shape[0]
    set_sp_num_tokens(num_tokens)
    tp_group = get_parallel().tp_group
    tp_size = tp_group.world_size
    if tp_size == 1:
        return hidden_states
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        hidden_states = torch.nn.functional.pad(
            hidden_states, (0, 0, 0, padded - num_tokens)
        )
    return hidden_states.tensor_split(tp_size)[tp_group.rank_in_group].contiguous()


def sp_all_gather(hidden_states: torch.Tensor) -> torch.Tensor:
    """g: all-gather the per-rank shards back to the padded sequence along dim 0."""
    tp_group = get_parallel().tp_group
    tp_size = tp_group.world_size
    if tp_size == 1:
        return hidden_states
    if hidden_states.element_size() == 1:
        # One-byte rows travel as 16-bit words: the TP group's custom all-gather
        # takes 16- and 32-bit types only.
        words = sp_all_gather(hidden_states.contiguous().view(torch.bfloat16))
        return words.view(hidden_states.dtype)
    hidden_states = hidden_states.contiguous()
    output = hidden_states.new_empty(
        (hidden_states.shape[0] * tp_size, *hidden_states.shape[1:])
    )
    tp_group.all_gather_into_tensor(output, hidden_states)
    return output


def sp_exit_gather(hidden_states: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """g: all-gather the per-rank shards back to the full sequence along dim 0,
    then narrow to ``num_tokens`` (dropping the entry-scatter padding)."""
    return sp_all_gather(hidden_states)[:num_tokens]


def int8_reduce_scatter() -> bool:
    """Whether SP reduce-scatters carry INT8 payloads: when the TP group's
    all-reduces run INT8 QuickReduce, so the reduction keeps the precision already
    accepted for them."""
    from sglang.srt.distributed.device_communicators.quick_all_reduce import (
        QuickReduceRegime,
    )

    qr = getattr(get_parallel().tp_group, "qr_comm", None)
    return (
        qr is not None
        and not qr.disabled
        and qr.qr_quant_level is QuickReduceRegime.INT8
    )


def _int8_payload_applies(partial: torch.Tensor) -> bool:
    from sglang.kernels.ops.communication import q8_reduce_scatter

    return int8_reduce_scatter() and q8_reduce_scatter.supports(partial)


def sp_reduce_scatter(
    partial: torch.Tensor, num_valid: Optional[int] = None
) -> torch.Tensor:
    """g-bar: this rank's rows of the TP sum of a TP-partial ``[rows, h]`` tensor,
    with INT8 payloads under ``int8_reduce_scatter()``.

    Without ``num_valid`` every row is valid and the rows are zero-padded to a
    multiple of the TP size. With it, ``partial`` already has such padded rows,
    and the rows from ``num_valid`` on count as zeros.
    """
    tp_group = get_parallel().tp_group
    tp_size = tp_group.world_size
    if tp_size > 1 and _int8_payload_applies(partial):
        from sglang.kernels.ops.communication.q8_reduce_scatter import (
            reduce_scatter_q8,
        )

        if num_valid is None:
            return reduce_scatter_q8(
                partial, tp_group, ceil_align(partial.shape[0], tp_size)
            )
        return reduce_scatter_q8(partial[:num_valid], tp_group, partial.shape[0])
    if num_valid is None:
        num_valid = partial.shape[0]
        padded = ceil_align(num_valid, tp_size)
        if padded != num_valid:
            partial = torch.nn.functional.pad(partial, (0, 0, 0, padded - num_valid))
    elif num_valid < partial.shape[0]:
        partial[num_valid:].zero_()
    if tp_size == 1:
        return partial
    partial = partial.contiguous()
    output = partial.new_empty((partial.shape[0] // tp_size, *partial.shape[1:]))
    tp_group.reduce_scatter_tensor(output, partial)
    return output


def maybe_exit_gather(
    *,
    hidden_states: torch.Tensor,
    hidden_states_before_norm: Optional[torch.Tensor],
    input_ids: Optional[torch.Tensor],
    forward_mode,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Undo the sequence sharding before the LM head, and leave the region.

    No-op unless this forward runs SP on a model that uses the participant
    linears. The token count comes from ``input_ids``
    and the predicate from ``runs_sp``, so this stays correct on CUDA graph
    replay, where the writes made inside the captured region do not re-execute.
    """
    if (
        input_ids is None
        or get_flags().sp.model_collectives
        or not runs_sp(forward_mode, input_ids.shape[0])
    ):
        return hidden_states, hidden_states_before_norm
    num_tokens = input_ids.shape[0]
    hidden_states = sp_exit_gather(hidden_states, num_tokens=num_tokens)
    if hidden_states_before_norm is not None:
        hidden_states_before_norm = sp_exit_gather(
            hidden_states_before_norm, num_tokens=num_tokens
        )
    get_forward().set("sp_active", False)
    return hidden_states, hidden_states_before_norm


# --- fused matmul + collective fast-paths for the participant linears ---------
# Fused matmul+reduce-scatter (g-bar) and all-gather+matmul (g) overlap the
# collective with the GEMM. Availability is probed once at import; TP groups are
# registered for symmetric memory lazily by the fused ops on first use (the old
# enable_symm_mem_for_group is a deprecated no-op), so we only import the module
# to register the torch.ops.symm_mem namespace the probe checks. NVLink/NVSwitch.
try:
    import torch.distributed._symmetric_memory  # noqa: F401

    _HAS_TORCH_SYMM_MEM_FUSED = hasattr(
        torch.ops.symm_mem, "fused_matmul_reduce_scatter"
    ) and hasattr(torch.ops.symm_mem, "fused_all_gather_matmul")
except Exception:
    _HAS_TORCH_SYMM_MEM_FUSED = False


def sp_fused_matmul_eligible(linear) -> bool:
    """Whether the selected fused matmul+collective path applies.

    Muse Spark selects CUTLASS; other models keep the Torch symmetric-memory
    path. Both require an unquantized, bias-free bf16/fp16 linear. The decision
    depends only on static layer properties and is identical across TP ranks.
    """
    if (
        not (_HAS_TORCH_SYMM_MEM_FUSED or getattr(linear, "_sp_use_cutlass", False))
        or linear.bias is not None
    ):
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return False
    return linear.weight.dtype in (torch.bfloat16, torch.float16)


def column_parallel_g_matmul(
    linear, input_parallel: torch.Tensor, bias
) -> torch.Tensor:
    """Megatron SP g for a ColumnParallelLinear participant (qkv / gate_up).

    The input is this rank's sequence shard ``[M_pad/tp, K]``; all-gather it back
    to the full sequence, matmul, and narrow to the real token count (recorded at
    the entry scatter). Eligible Muse Spark layers use one CUTLASS GEMM with
    tile-gated all-gather; other models retain their existing implementation.
    """
    num_tokens = sp_num_tokens()
    if sp_fused_matmul_eligible(linear):
        if getattr(linear, "_sp_use_cutlass", False):
            from sglang.kernels.ops.communication.sp_collective_gemm import (
                all_gather_matmul,
            )
            from sglang.srt.layers.communication.sp_collective_gemm_workspace import (
                sp_collective_gemm_workspace,
            )

            return all_gather_matmul(
                input_parallel.contiguous(),
                linear.weight,
                get_parallel().tp_group.device_group,
                workspace=sp_collective_gemm_workspace,
            )[:num_tokens]
        group_name = get_parallel().tp_group.device_group.group_name
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            input_parallel.contiguous(),
            [linear.weight.t()],
            gather_dim=0,
            group_name=group_name,
        )
        return mm_outputs[0][:num_tokens]
    gathered = sp_exit_gather(input_parallel, num_tokens=num_tokens)
    return linear.quant_method.apply(linear, gathered, bias)


def row_parallel_gbar_matmul(linear, input_: torch.Tensor, bias) -> torch.Tensor:
    """Megatron SP g-bar for a RowParallelLinear participant (o_proj / down).

    Computes ``input_ @ weight.T`` and reduce-scatters (sum) the result across the
    TP group along dim 0, leaving this rank's ``[M_pad/tp, h]`` shard. The token
    dim is padded to a multiple of tp_size (padding rows are zeros, dropped by the
    exit gather). Eligible Muse Spark layers use one CUTLASS GEMM with a
    multicast reduce-scatter epilogue; other models retain their existing path.
    """
    tp_size = linear.tp_group.world_size if linear.tp_group is not None else 1
    x = input_.contiguous()
    num_tokens = x.shape[0]
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - num_tokens))
    if sp_fused_matmul_eligible(linear):
        if getattr(linear, "_sp_use_cutlass", False):
            from sglang.kernels.ops.communication.sp_collective_gemm import (
                matmul_reduce_scatter,
            )
            from sglang.srt.layers.communication.sp_collective_gemm_workspace import (
                sp_collective_gemm_workspace,
            )

            return matmul_reduce_scatter(
                x,
                linear.weight,
                get_parallel().tp_group.device_group,
                workspace=sp_collective_gemm_workspace,
            )
        group_name = get_parallel().tp_group.device_group.group_name
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x,
            linear.weight.t(),
            "sum",
            scatter_dim=0,
            group_name=group_name,
        )
    full = linear.quant_method.apply(linear, x, bias)
    output = full.new_empty((padded // tp_size, *full.shape[1:]))
    get_parallel().tp_group.reduce_scatter_tensor(output, full)
    return output
