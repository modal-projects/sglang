"""
Support attention backend for TRTLLM MLA kernels from flashinfer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.jit_kernel.fixup_zero_kv import fixup_zero_kv_rows
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
    get_num_page_per_block_flashmla,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    pad_draft_extend_query as pad_draft_extend_query_triton,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    unpad_draft_extend_output as unpad_draft_extend_output_triton,
)
from sglang.srt.layers.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
)
from sglang.srt.layers.cp.dcp import (
    dcp_enabled,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
    get_dcp_lens,
)
from sglang.srt.layers.cp.dcp.kernels import (
    dcp_verify_draft_merge,
    dcp_verify_draft_merge_torch,
)
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_flashinfer_available, is_float4_e2m1fn_x2

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Constants
DEFAULT_WORKSPACE_SIZE_MB = 150  # Memory workspace size in MB

# Block constraint from flashinfer requirements
# From flashinfer.decode._check_trtllm_gen_mla_shape:
#   block_num % (128 / block_size) == 0
# This imposes that the total number of blocks must be divisible by
# (128 / block_size). We capture the 128 constant here so we can
# compute the LCM with other padding constraints.
TRTLLM_BLOCK_CONSTRAINT = 128


def _quantize_fp8_qkv(q, k, v, layer):
    q = q.to(torch.float8_e4m3fn)

    k_scale = getattr(layer, "k_scale_float", None)
    if k_scale is None:
        k_scale = 1.0
    if k_scale != 1.0:
        assert hasattr(layer, "k_scale"), "k_scale is not set"
        k_2d, _ = scaled_fp8_quant(
            k.reshape(-1, k.shape[-1]).contiguous(), layer.k_scale
        )
        k = k_2d.reshape(k.shape)
    else:
        k = k.to(torch.float8_e4m3fn)

    v_scale = getattr(layer, "v_scale_float", None)
    if v_scale is None:
        v_scale = 1.0
    if v_scale != 1.0:
        assert hasattr(layer, "v_scale"), "v_scale is not set"
        v_2d, _ = scaled_fp8_quant(
            v.reshape(-1, v.shape[-1]).contiguous(), layer.v_scale
        )
        v = v_2d.reshape(v.shape)
    else:
        v = v.to(torch.float8_e4m3fn)

    return q, k, v, k_scale, v_scale


global_zero_init_workspace_buffer = None
# cute-dsl needs its own workspace: it overwrites the buffer with split-KV
# partials, which corrupts the trtllm-gen multiCtasKv counters that rely on the
# zero-init buffer (they share it under attention-backend=cutedsl_mla, where
# draft-extend falls back to trtllm-gen) and deadlocks the reduction.
global_cute_dsl_workspace_buffer = None


@dataclass
class TRTLLMMLAPrefillMetadata:
    """Metadata for TRTLLM MLA prefill operations."""

    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor
    fallback_to_flashinfer_impl: bool = False


@dataclass
class TRTLLMMLADecodeMetadata:
    """Metadata for TRTLLM MLA decode operations."""

    block_kv_indices: Optional[torch.Tensor] = None
    max_seq_len_k: Optional[int] = None
    max_seq_len_q: Optional[int] = None
    sum_seq_lens_q: Optional[int] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    seq_lens_q: Optional[torch.Tensor] = None
    seq_lens_k: Optional[torch.Tensor] = None

    # --- Decode context parallel (DCP) ---
    # Block tables above stay RANK-INVARIANT under DCP: they are built with the
    # widened logical page size (page_size * dcp_world_size) over the logical
    # req_to_token, and a logical page maps to the same physical page index on
    # every rank. Only per-request KV LENGTHS become rank-local.
    #
    # decode: rank-local visible KV length (owner rule pos % N == rank).
    dcp_local_seq_lens: Optional[torch.Tensor] = None
    dcp_max_local_seq_len: Optional[int] = None
    # target-verify: rank-local length of the committed PREFIX only (i.e.
    # seq_lens WITHOUT the draft tokens); the draft block is handled by the
    # fused residue-class merge kernel in _forward_target_verify_dcp.
    dcp_local_prefix_lens: Optional[torch.Tensor] = None
    dcp_max_local_prefix_len: Optional[int] = None


class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """TRTLLM MLA attention kernel from flashinfer."""

    # trtllm-gen kernels rebuild metadata from preallocated buffers and never
    # read seq_lens_cpu / seq_lens_sum; opt out of the D2H sync.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
        backend: str = "trtllm-gen",
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config

        # Model parameters
        self.num_q_heads = config.num_attention_heads // get_parallel().attn_tp_size
        self.num_kv_heads = config.get_num_kv_heads(get_parallel().attn_tp_size)
        self.num_local_heads = config.num_attention_heads // get_parallel().attn_tp_size

        # MLA-specific dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Runtime parameters
        self.backend = backend
        self.scaling = config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Workspace allocation
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        if self.backend == "cute-dsl":
            # Separate buffer from trtllm-gen (see note above); safe to share
            # among cute-dsl instances.
            global global_cute_dsl_workspace_buffer
            if global_cute_dsl_workspace_buffer is None:
                global_cute_dsl_workspace_buffer = torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_cute_dsl_workspace_buffer
        else:
            global global_zero_init_workspace_buffer
            if global_zero_init_workspace_buffer is None:
                global_zero_init_workspace_buffer = torch.zeros(
                    self.workspace_size,
                    dtype=torch.int8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
        self.padded_q_buffer = None
        self.unpad_output_buffer = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None

        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.cuda_graph_custom_mask = None

        # Decode context parallel (DCP) info; 1 / 0 when DCP is disabled.
        self.dcp_world_size = get_attention_dcp_world_size()
        self.dcp_rank = get_attention_dcp_rank()
        # Fused Triton kernel for the verify-phase residue-class block
        # attention + LSE merge; SGLANG_DCP_VERIFY_FUSED=0 falls back to the
        # unfused torch reference for A/B debugging.
        self.dcp_verify_fused = envs.SGLANG_DCP_VERIFY_FUSED.get()
        if self.dcp_world_size > 1:
            # create_flashmla_kv_indices_triton processes
            # FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON (4096) tokens per CTA with
            # NUM_PAGE_PER_BLOCK = 4096 // PAGED_SIZE (floor). A logical page
            # size that does not divide 4096 silently drops trailing
            # block-table entries -> the kernel reads garbage pages.
            logical_page_size = self.page_size * self.dcp_world_size
            if 4096 % logical_page_size != 0:
                raise ValueError(
                    f"DCP requires page_size * dcp_size to divide 4096, got "
                    f"page_size={self.page_size} * dcp_size={self.dcp_world_size} "
                    f"= {logical_page_size}."
                )
            # tokenspeed fold_sq (gathered heads < 128): q_len must divide
            # evenly into q_chunk = min(q_len, 128 // H) work tiles. Fail at
            # boot instead of inside CUDA-graph capture on the first verify.
            if self.num_draft_tokens:
                gathered_heads = self.num_q_heads * self.dcp_world_size
                if gathered_heads < 128:
                    q_chunk = min(self.num_draft_tokens, 128 // gathered_heads)
                    if self.num_draft_tokens % q_chunk != 0:
                        raise ValueError(
                            f"DCP + speculative verify with gathered heads "
                            f"{gathered_heads} (< 128) requires "
                            f"speculative_num_draft_tokens divisible by "
                            f"{q_chunk}, got {self.num_draft_tokens}."
                        )

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """
        Calculate padded block count that satisfies both TRT-LLM and Triton constraints.

        Args:
            max_seq_len: Maximum sequence length in tokens

        Returns:
            Number of blocks padded to satisfy all constraints
        """
        # Under DCP a block-table entry covers one LOGICAL page of
        # page_size * dcp_world_size tokens (== one physical page of
        # page_size tokens on every rank, at the same page index), so the
        # per-sequence block count shrinks by the DCP factor.
        logical_page_size = self.page_size * self.dcp_world_size
        blocks = triton.cdiv(max_seq_len, logical_page_size)

        # Apply dual constraints (take LCM to satisfy both):
        # 1. TRT-LLM: block_num % (128 / page_size) == 0. This is a kernel-side
        #    constraint on the PHYSICAL page size (the kernel always sees
        #    per-rank pages of self.page_size), so it keeps self.page_size.
        # 2. Triton: number of pages per index-build block, computed with the
        #    LOGICAL page size used by create_flashmla_kv_indices_triton.
        trtllm_constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        triton_constraint = get_num_page_per_block_flashmla(logical_page_size)
        constraint_lcm = math.lcm(trtllm_constraint, triton_constraint)

        if blocks % constraint_lcm != 0:
            blocks = triton.cdiv(blocks, constraint_lcm) * constraint_lcm
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block KV indices tensor using Triton kernel.

        Args:
            batch_size: Batch size
            max_blocks: Maximum number of blocks per sequence
            req_pool_indices: Request pool indices
            seq_lens: Sequence lengths
            device: Target device

        Returns:
            Block KV indices tensor
        """
        block_kv_indices = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )

        # Under DCP: build over the logical req_to_token with the widened page
        # size. logical_loc // (page_size * dcp) == physical_page_index on
        # every rank, so the resulting table is rank-invariant and directly
        # indexes each rank's physical KV pool pages.
        logical_page_size = self.page_size * self.dcp_world_size
        create_flashmla_kv_indices_triton[
            (
                batch_size,
                get_num_kv_index_blocks_flashmla(max_blocks, logical_page_size),
            )
        ](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=logical_page_size,
        )

        return block_kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MLA."""

        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)

        self.decode_cuda_graph_kv_indices = torch.full(
            (max_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=self.device
        )
        num_tokens_per_bs = max_num_tokens // max_bs

        if is_float4_e2m1fn_x2(self.data_type):
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            self.store_dtype = torch.uint8
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs // 2, self.num_q_heads, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens // 2, self.num_q_heads, 512),
                dtype=self.store_dtype,
                device=self.device,
            )
        else:
            # Buffer for padded query: (max_bs, max_draft_tokens, num_q_heads, v_head_dim)
            self.padded_q_buffer = torch.zeros(
                (max_bs, num_tokens_per_bs, self.num_q_heads, self.kv_cache_dim),
                dtype=self.data_type,
                device=self.device,
            )

            # Buffer for unpadded output: (max_num_tokens, num_q_heads, v_head_dim)
            self.unpad_output_buffer = torch.zeros(
                (max_num_tokens, self.num_q_heads, 512),
                dtype=self.data_type,
                device=self.device,
            )

        if self.num_draft_tokens and not self.skip_prefill:
            # Worst-case FULL_MASK tree-mask scratch (bool); build_tree writes it
            # in-place so the gpu_only path needs no seq_lens_sum.
            self.cuda_graph_custom_mask = torch.zeros(
                max_num_tokens * (self.max_context_len + self.num_draft_tokens),
                dtype=torch.bool,
                device=self.device,
            )

        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    def get_verify_buffers_to_fill_after_draft(self):
        return [self.cuda_graph_custom_mask, None]

    def _init_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        seq_lens: torch.Tensor,
        device: torch.device,
    ):
        """Allocate persistent metadata buffers for CUDA graph capture."""
        metadata = TRTLLMMLADecodeMetadata()

        if dcp_enabled():
            if forward_mode.is_draft_extend_v2():
                raise NotImplementedError(
                    "DCP does not support draft_extend_v2 on the trtllm/tokenspeed "
                    "MLA backend (EAGLE-style draft extend is not DCP-aware)."
                )
            # Persistent buffers for rank-local KV lengths; filled on every
            # capture/replay in _apply_cuda_graph_metadata. The kernel-call
            # max_seq_len is baked at capture, so use the max-context bound.
            dcp_max_local = max(
                triton.cdiv(self.max_context_len, self.dcp_world_size), 1
            )
            if forward_mode.is_target_verify():
                metadata.dcp_local_prefix_lens = torch.zeros(
                    (bs,), dtype=torch.int32, device=device
                )
                metadata.dcp_max_local_prefix_len = dcp_max_local
            else:
                metadata.dcp_local_seq_lens = torch.zeros(
                    (bs,), dtype=torch.int32, device=device
                )
                metadata.dcp_max_local_seq_len = dcp_max_local

        if forward_mode.is_target_verify():
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_bs = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )
            metadata.seq_lens_q = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=device
            )
            metadata.seq_lens_k = torch.zeros((bs,), dtype=torch.int32, device=device)

        # Capture with full width so future longer sequences are safe during replay.
        max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks_per_seq]
        metadata.block_kv_indices = block_kv_indices
        metadata.max_seq_len_k = self.max_context_len

        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        """Shared decode / target-verify / draft-extend capture+replay body.

        Public entry: :py:meth:`init_forward_metadata_out_graph` (which routes
        the non-decode-family modes to the FlashInferMLA parent).
        """
        metadata = self.decode_cuda_graph_metadata[bs]

        if forward_mode.is_target_verify():
            prefix_seq_lens = seq_lens[:bs]
            if metadata.dcp_local_prefix_lens is not None:
                # Rank-local share of the committed prefix (WITHOUT the draft
                # tokens) for the two-phase DCP verify attention.
                metadata.dcp_local_prefix_lens.copy_(
                    get_dcp_lens(
                        prefix_seq_lens.to(torch.int32),
                        self.dcp_world_size,
                        self.dcp_rank,
                    )
                )
            seq_lens = prefix_seq_lens + self.num_draft_tokens
            metadata.seq_lens_k.copy_(seq_lens)
        elif forward_mode.is_draft_extend_v2():
            num_tokens_per_bs = self.num_draft_tokens
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.sum_seq_lens_q = num_tokens_per_bs * bs
            seq_lens = seq_lens[:bs]
            metadata.seq_lens_k.copy_(seq_lens)
        elif metadata.dcp_local_seq_lens is not None:
            # decode / idle
            metadata.dcp_local_seq_lens.copy_(
                get_dcp_lens(
                    seq_lens[:bs].to(torch.int32),
                    self.dcp_world_size,
                    self.dcp_rank,
                )
            )

        # Update block indices for new sequences. Under DCP the table is
        # rank-invariant: widened logical page size over the logical
        # req_to_token (see _create_block_kv_indices).
        logical_page_size = self.page_size * self.dcp_world_size
        create_flashmla_kv_indices_triton[
            (
                bs,
                get_num_kv_index_blocks_flashmla(
                    metadata.block_kv_indices.shape[1], logical_page_size
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            metadata.block_kv_indices,
            self.req_to_token.stride(0),
            metadata.block_kv_indices.shape[1],
            PAGED_SIZE=logical_page_size,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_mha_chunk_metadata(self, forward_batch: ForwardBatch) -> None:
        has_prefix = any(forward_batch.extend_prefix_lens_cpu)
        fallback_to_flashinfer_impl = (
            self.disable_chunked_prefix_cache and has_prefix
        ) or is_in_tc_piecewise_cuda_graph()
        if fallback_to_flashinfer_impl:
            super().init_mha_chunk_metadata(
                forward_batch, disable_flashinfer_ragged=True
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        forward_mode = forward_batch.forward_mode

        if (
            not forward_mode.is_decode_or_idle()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend_v2()
        ):
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

        bs = forward_batch.batch_size
        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._init_cuda_graph_metadata(
                bs,
                num_tokens,
                forward_mode,
                forward_batch.seq_lens,
                forward_batch.seq_lens.device,
            )
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
            )
        else:
            self._apply_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_mode,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""
        # Delegate to parent for non-decode modes.
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # For extend batch with prefix length > 0, fallback to ragged kernel implemented in flashinfer MLA backend
            # when chunked prefix cache is disabled.
            # Also fallback to flashinfer MLA backend when in piecewise cuda graph, since it only supports MLA forward mode.
            has_prefix = any(forward_batch.extend_prefix_lens_cpu)
            fallback_to_flashinfer_impl = (
                self.disable_chunked_prefix_cache and has_prefix
            ) or is_in_tc_piecewise_cuda_graph()
            if fallback_to_flashinfer_impl:
                super().init_forward_metadata(forward_batch)

            seq_lens = forward_batch.seq_lens - forward_batch.extend_prefix_lens
            cum_seq_lens_q = torch.cat(
                (
                    torch.zeros(
                        1, dtype=torch.int32, device=forward_batch.seq_lens.device
                    ),
                    torch.cumsum(seq_lens, dim=0),
                )
            ).int()
            max_seq_len = max(forward_batch.extend_seq_lens_cpu)
            self.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
                max_seq_len,
                cum_seq_lens_q,
                seq_lens,
                fallback_to_flashinfer_impl,
            )
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            bs = forward_batch.batch_size
            self.forward_decode_metadata = TRTLLMMLADecodeMetadata()
            # This is necessary because the backend instance persists across forward passes,
            # and forward_prefill_metadata from a previous regular extend call could still be set.
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                self.forward_prefill_metadata = None
            # Get maximum sequence length.
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                max_seq = forward_batch.seq_lens_cpu.max().item()
            else:
                max_seq = forward_batch.seq_lens.max().item()

            seq_lens = forward_batch.seq_lens

            if dcp_enabled() and forward_batch.forward_mode.is_draft_extend_v2():
                raise NotImplementedError(
                    "DCP does not support draft_extend_v2 on the trtllm/tokenspeed "
                    "MLA backend (EAGLE-style draft extend is not DCP-aware)."
                )

            if forward_batch.forward_mode.is_target_verify():
                if dcp_enabled():
                    # Rank-local prefix lens (pre-draft) for two-phase verify.
                    self.forward_decode_metadata.dcp_local_prefix_lens = get_dcp_lens(
                        seq_lens.to(torch.int32),
                        self.dcp_world_size,
                        self.dcp_rank,
                    )
                    self.forward_decode_metadata.dcp_max_local_prefix_len = max(
                        triton.cdiv(int(max_seq), self.dcp_world_size), 1
                    )
                max_seq = max_seq + self.num_draft_tokens
                seq_lens = seq_lens + self.num_draft_tokens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)
            elif forward_batch.forward_mode.is_draft_extend_v2():
                sum_seq_lens_q = sum(forward_batch.extend_seq_lens_cpu)
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                # see NOTE(draft_extend seq_len handling)
                seq_lens = seq_lens - forward_batch.extend_seq_lens + max_seq_len_q

                self.forward_decode_metadata.max_seq_len_q = max_seq_len_q
                self.forward_decode_metadata.sum_seq_lens_q = sum_seq_lens_q
                self.forward_decode_metadata.cu_seqlens_q = cu_seqlens_q
                self.forward_decode_metadata.seq_lens_q = forward_batch.extend_seq_lens
                self.forward_decode_metadata.seq_lens_k = seq_lens.to(torch.int32)

            if dcp_enabled() and forward_batch.forward_mode.is_decode_or_idle():
                self.forward_decode_metadata.dcp_local_seq_lens = get_dcp_lens(
                    seq_lens.to(torch.int32),
                    self.dcp_world_size,
                    self.dcp_rank,
                )
                self.forward_decode_metadata.dcp_max_local_seq_len = max(
                    triton.cdiv(int(max_seq), self.dcp_world_size), 1
                )

            max_seqlen_pad = self._calc_padded_blocks(max_seq)
            block_kv_indices = self._create_block_kv_indices(
                bs,
                max_seqlen_pad,
                forward_batch.req_pool_indices,
                seq_lens,
                seq_lens.device,
            )

            self.forward_decode_metadata.block_kv_indices = block_kv_indices
            self.forward_decode_metadata.max_seq_len_k = int(max_seq)
            self.forward_decode_metadata.batch_size = bs

            forward_batch.decode_trtllm_mla_metadata = self.forward_decode_metadata
        else:
            return super().init_forward_metadata(forward_batch)

    def pad_draft_extend_query(
        self,
        q: torch.Tensor,
        padded_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor:
        """Pad draft extended query using Triton kernel."""
        return pad_draft_extend_query_triton(
            q,
            padded_q,
            seq_lens_q,
            cu_seqlens_q,
        )

    def unpad_draft_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
    ) -> torch.Tensor:
        """Unpad draft extended output using Triton kernel."""
        return unpad_draft_extend_output_triton(
            raw_out,
            cu_seqlens_q,
            seq_lens_q,
            sum_seq_lens_q,
            self.unpad_output_buffer,
        )

    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:
        """BMM1 scale q_scale * k_scale * softmax_scale. k_scale only
        applies when the KV cache stores FP8."""
        q_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
        else:
            if getattr(layer, "k_scale_float", None) is not None:
                logger.warning_once(
                    "Checkpoint has k_scale but KV cache dtype is not FP8. "
                    "Ignoring k_scale for BMM1 (k_scale=%.4f, kv_dtype=%s).",
                    layer.k_scale_float,
                    self.data_type,
                )
            k_scale = 1.0
        return q_scale * k_scale * layer.scaling

    def _decode_output_scale(self, layer: RadixAttention) -> float:
        """Dequant scale applied to attention output when KV is stored FP8
        (mirrors the tokenspeed decode kernel's output_scale)."""
        if self.data_type == torch.float8_e4m3fn:
            k_scale = getattr(layer, "k_scale_float", None)
            if k_scale is not None:
                return float(k_scale)
        return 1.0

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
        return_lse: bool = False,
        causal_mask: bool = True,
    ) -> torch.Tensor:
        """Hook for subclasses to swap the decode/spec-verify kernel."""
        if return_lse or not causal_mask:
            # flashinfer's trtllm-gen MLA decode kernel exposes neither an LSE
            # output nor a non-causal mode; DCP needs both. Use the
            # tokenspeed_mla backend for DCP.
            raise NotImplementedError(
                "DCP decode/verify requires return_lse / non-causal support "
                "from the decode kernel; the trtllm-gen MLA kernel provides "
                "neither. Use --attention-backend tokenspeed_mla for DCP."
            )

        # Scale computation for TRTLLM MLA kernel BMM1 operation:
        # The final BMM1 scale is computed as: q_scale * k_scale * softmax_scale
        # Scale components:
        # - q_scale: Query scaling factor (set to 1.0 for both FP16/FP8 paths)
        # - k_scale: Key scaling factor from model checkpoint. Only applied when KV cache
        #   stores FP8-quantized values, to compensate for the quantization scaling.
        #   For BF16/FP16 KV cache, k_scale must be 1.0 since values are unscaled.
        # - softmax_scale: Attention softmax scaling = 1/sqrt(head_dim), pre-computed as layer.scaling
        bmm1_scale = self._compute_decode_bmm1_scale(layer)
        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        extra_kwargs = {"backend": self.backend} if self.backend != "trtllm-gen" else {}
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            **extra_kwargs,
        )

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        max_q_len: int,
        seq_lens_kv: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
        out_buffer: torch.Tensor,
        o_sf_scale: float = 1.0,
    ):
        """Hook for subclasses to swap the ragged prefill kernel. Q/K/V arrive
        in model-native dtype; subclasses do any kernel-specific quantization.
        Returns the output tensor or (output, lse) if return_lse."""
        q_scale = k_scale = v_scale = 1.0
        if self.data_type == torch.float8_e4m3fn:
            q, k, v, k_scale, v_scale = _quantize_fp8_qkv(q, k, v, layer)
        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.workspace_buffer,
            batch_size=batch_size,
            window_left=-1,
            enable_pdl=False,
            max_q_len=max_q_len,
            bmm1_scale=q_scale * k_scale * layer.scaling,
            bmm2_scale=v_scale,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            seq_lens=seq_lens_kv,
            max_kv_len=max_kv_len,
            is_causal=is_causal,
            return_lse=return_lse,
            o_sf_scale=o_sf_scale,
            out=out_buffer,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
        )

    def forward_decode(
        self,
        q: torch.Tensor,  # q_nope
        k: torch.Tensor,  # k_nope
        v: torch.Tensor,  # not used in this backend
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MLA kernel."""
        merge_query = q_rope is not None
        if self.data_type == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

        # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            query = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            # For FP8 path, we already have the query and rope parts merged because of the quantize_and_rope_for_fp8 function
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            query = query.to(self.q_data_type) * llama_4_scaling
            query = query.to(self.data_type)

        # Ensure query has shape [bs, acc_q_len, num_q_heads, head_dim] when seq_len 1
        if query.dim() == 3:
            query = query.unsqueeze(1)

        # Prepare KV cache inline
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        # Get metadata
        metadata = (
            getattr(forward_batch, "decode_trtllm_mla_metadata", None)
            or self.forward_decode_metadata
        )

        # Backstop: metadata was built pre-pad (marked) and DP padding then
        # grew the batch. The marker path deliberately does not re-plan
        # post-pad (DSA can't rebuild on a padded batch, see #27091), so this
        # local re-plan catches the size mismatch.
        batch_size = getattr(metadata, "batch_size", None)
        if batch_size is not None and batch_size < forward_batch.batch_size:
            self.init_forward_metadata(forward_batch)
            metadata = forward_batch.decode_trtllm_mla_metadata

        if dcp_enabled() and forward_batch.forward_mode.is_decode():
            # q arrives all-gathered along heads (num_local_heads *
            # dcp_world_size, via attn_mqa_for_dcp_decode); attention runs over
            # this rank's KV shard only (rank-invariant block tables +
            # rank-local lens) and returns (out, lse) for the cross-rank merge
            # in cp_lse_ag_out_rs_mla. The LSE is base-2 with the softmax scale
            # folded — exactly what _correct_attn_cp_out_kernel consumes.
            raw_out, lse = self._run_decode_kernel(
                query=query,
                kv_cache=kv_cache,
                block_tables=metadata.block_kv_indices,
                seq_lens=metadata.dcp_local_seq_lens,
                max_seq_len=metadata.dcp_max_local_seq_len,
                layer=layer,
                return_lse=True,
            )
            output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            # lse: [B, 1, H] -> [B, H]
            return output, lse.view(-1, layer.tp_q_head_num)

        raw_out = self._run_decode_kernel(
            query=query,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
        )

        # Reshape output directly without slicing
        output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return output

    def _forward_target_verify_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: TRTLLMMLADecodeMetadata,
        kv_cache: torch.Tensor,
    ):
        """Two-phase DCP attention for speculative-decoding target verify.

        Phase (a): the decode kernel attends this rank's LOCAL share of the
        committed PREFIX KV (rank-local lens, rank-invariant block tables)
        with all q_len draft-block tokens. causal_mask=False because every
        block token attends the entire committed prefix. Returns a base-2 LSE
        with the softmax scale folded.

        Phase (b): the num_draft_tokens fresh K/V of the draft block (k /
        k_rope arguments, replicated on every rank) — but each rank only
        attends its RESIDUE-CLASS share of the block positions:
        rank r attends block position j iff (seq_len + j) % dcp_size == r,
        under causality q_i >= j. This partitions the block tokens exactly
        once across ranks, matching how the rank-filtered cache write shards
        them for future steps. Tiny [bs, q_len, H, <= q_len] masked attention
        in fp32, producing the same base-2 / scale-folded LSE convention
        (empty rows -> lse = -inf, out = 0, which
        _correct_attn_cp_out_kernel treats as no contribution).

        Phase (c): merge (a) + (b) locally per token in base-2 (max-subtracted
        log2-sum-exp2). The cross-rank merge is then done by the regular
        cp_lse_ag_out_rs_mla call in forward_mla.py — this method just returns
        one normal partial (out, lse).

        Phases (b)+(c) run as ONE fused Triton kernel per layer
        (dcp_verify_draft_merge); SGLANG_DCP_VERIFY_FUSED=0 selects the
        original unfused torch reference (dcp_verify_draft_merge_torch).
        Both are static-shape / no host reads (CUDA-graph capturable).
        """
        bs = forward_batch.batch_size
        draft = self.num_draft_tokens
        num_heads = layer.tp_q_head_num  # gathered heads under DCP

        # ---- Phase (a): local committed prefix via the decode kernel ----
        o_a, lse_a = self._run_decode_kernel(
            query=q,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=metadata.dcp_local_prefix_lens,
            max_seq_len=metadata.dcp_max_local_prefix_len,
            layer=layer,
            return_lse=True,
            causal_mask=False,
        )
        # o_a: [bs, draft, H, kv_lora_rank]; lse_a: [bs, draft, H] fp32,
        # base-2, +inf sentinel on empty rows (local prefix len == 0).

        # ---- Phases (b)+(c): fused residue-class block attn + base-2 merge
        softmax_scale = self._compute_decode_bmm1_scale(layer)
        output_scale = self._decode_output_scale(layer)

        merge_fn = (
            dcp_verify_draft_merge
            if self.dcp_verify_fused
            else dcp_verify_draft_merge_torch
        )
        o, merged_lse = merge_fn(
            q=q.view(bs, draft, num_heads, -1),
            k_latent=k.reshape(bs, draft, self.kv_lora_rank),
            k_rope=k_rope.reshape(bs, draft, self.qk_rope_head_dim),
            o_a=o_a.view(bs, draft, num_heads, self.kv_lora_rank),
            lse_a=lse_a.view(bs, draft, num_heads),
            seq_lens=forward_batch.seq_lens,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            dcp_rank=self.dcp_rank,
            dcp_world_size=self.dcp_world_size,
        )

        out = o.view(-1, num_heads * layer.v_head_dim)
        # merged_lse: [bs, draft, H] -> [tokens, H]; -inf marks fully-empty
        # local rows (the cross-rank merge kernel zeroes their contribution).
        return out, merged_lse.view(-1, num_heads)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if (
            self.forward_prefill_metadata is not None
            and self.forward_prefill_metadata.fallback_to_flashinfer_impl
        ):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
            )

        # TODO refactor to avoid code duplication
        merge_query = q_rope is not None
        if (
            self.data_type == torch.float8_e4m3fn
        ) and forward_batch.forward_mode.is_target_verify():
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert all(
                x is not None for x in [q_rope, k_rope, cos_sin_cache]
            ), "For FP8 path and using flashinfer.rope.mla_rope_quantize we need all of q_rope, k_rope and cos_sin_cache to be not None."
            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

        # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )

        # TODO refactor to avoid code duplication
        # Prepare query tensor inline
        if merge_query:
            # For FP16 path, we merge the query and rope parts into a single tensor
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q = q.to(self.q_data_type) * llama_4_scaling
            q = q.to(self.data_type)

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            metadata = (
                getattr(forward_batch, "decode_trtllm_mla_metadata", None)
                or self.forward_decode_metadata
            )

            # Backstop: metadata was built pre-pad (marked) and DP padding
            # then grew the batch. The marker path deliberately does not
            # re-plan post-pad (DSA can't rebuild on a padded batch, see
            # #27091), so this local re-plan catches the size mismatch.
            batch_size = getattr(metadata, "batch_size", None)
            if batch_size is not None and batch_size < forward_batch.batch_size:
                self.init_forward_metadata(forward_batch)
                metadata = forward_batch.decode_trtllm_mla_metadata

            # Ensure query has shape [bs, num_draft_tokens, num_q_heads, head_dim]
            bs = forward_batch.batch_size

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

            q = q.to(self.data_type)

            if forward_batch.forward_mode.is_target_verify():
                # For target_verify, all sequences have the same number of draft tokens
                q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                if dcp_enabled():
                    # Two-phase DCP verify: local prefix shard via the decode
                    # kernel + residue-class share of the fresh draft block in
                    # torch, merged locally in base-2. k / k_rope here are the
                    # draft block's fresh latent KV (computed replicated on
                    # every rank before the rank-filtered cache write).
                    return self._forward_target_verify_dcp(
                        q=q,
                        k=k,
                        k_rope=k_rope,
                        layer=layer,
                        forward_batch=forward_batch,
                        metadata=metadata,
                        kv_cache=kv_cache,
                    )
                max_seq_len = (
                    metadata.max_seq_len_k + forward_batch.spec_info.draft_token_num
                )
                needs_unpad = False
            else:
                # draft_extend: handle varying num_correct_drafts_per_req. If total_tokens % bs == 0,
                # we can directly reshape q; otherwise, pad to max_seq_len_q.
                total_tokens = q.shape[0]
                tokens_per_seq = total_tokens // bs if bs > 0 else 0
                can_direct_view = bs > 0 and (total_tokens % bs == 0)

                if can_direct_view:
                    max_seq_len = metadata.max_seq_len_k + tokens_per_seq
                    q = q.view(bs, tokens_per_seq, layer.tp_q_head_num, layer.head_dim)
                    needs_unpad = False
                else:
                    # Varying lengths: pad q to (bs, max_seq_len_q, ...)
                    actual_seq_lens_q = forward_batch.extend_seq_lens
                    actual_max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                    max_seq_len = metadata.max_seq_len_k + actual_max_seq_len_q

                    actual_cu_seqlens_q = torch.nn.functional.pad(
                        torch.cumsum(actual_seq_lens_q, dim=0, dtype=torch.int32),
                        (1, 0),
                    )

                    if self.padded_q_buffer is not None:
                        padded_q = self.padded_q_buffer[
                            :bs, :actual_max_seq_len_q, :, :
                        ].to(dtype=q.dtype)
                        padded_q.zero_()
                    else:
                        padded_q = torch.zeros(
                            (
                                bs,
                                actual_max_seq_len_q,
                                layer.tp_q_head_num,
                                layer.head_dim,
                            ),
                            dtype=q.dtype,
                            device=q.device,
                        )

                    q = self.pad_draft_extend_query(
                        q, padded_q, actual_seq_lens_q, actual_cu_seqlens_q
                    )
                    needs_unpad = True
                    unpad_seq_lens_q = actual_seq_lens_q
                    unpad_cu_seqlens_q = actual_cu_seqlens_q
                    unpad_sum_seq_lens_q = total_tokens

            assert kv_cache.dtype == self.data_type

            raw_out = self._run_decode_kernel(
                query=q,
                kv_cache=kv_cache,
                block_tables=metadata.block_kv_indices,
                seq_lens=metadata.seq_lens_k,
                max_seq_len=max_seq_len,
                layer=layer,
            )

            if needs_unpad:
                # Unpad the output for draft_extend mode with varying lengths
                # Use the actual values computed during padding, not from metadata
                output = self.unpad_draft_extend_output(
                    raw_out,
                    unpad_cu_seqlens_q,
                    unpad_seq_lens_q,
                    unpad_sum_seq_lens_q,
                )
                output = output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            else:
                output = raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            return output

        if k_rope is not None:
            k = torch.cat([k, k_rope], dim=-1)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)

        # When chunked prefix cache is enabled, dispatch to different path for ragged attention.
        if forward_batch.attn_attend_prefix_cache:
            # MHA for chunked prefix kv cache when running model with MLA
            assert forward_batch.prefix_chunk_idx is not None
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert q_rope is None
            assert k_rope is None
            chunk_idx = forward_batch.prefix_chunk_idx

            out = torch.empty(
                q.shape[0],
                layer.tp_q_head_num,
                layer.v_head_dim,
                dtype=self.q_data_type,
                device=q.device,
            )
            result = self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=forward_batch.prefix_chunk_seq_lens[chunk_idx],
                cum_seq_lens_kv=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                max_kv_len=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                is_causal=False,
                return_lse=True,
                out_buffer=out,
                o_sf_scale=-1.0,
            )

            # The TRT-LLM ragged attention cubin kernel does not correctly
            # handle rows with kv_len == 0: it leaves stale data in the
            # workspace softmaxStats buffer and may produce non-zero output
            # for those rows.  Fix up by forcing out=0 and lse=-inf for
            # zero-KV rows so that downstream merge_state ignores them.
            # Skip entirely when this chunk has no zero-KV rows (pure CPU
            # check, precomputed in prepare_chunked_prefix_cache_info).
            if forward_batch.prefix_chunk_has_zero_kv[chunk_idx]:
                out_tensor, lse_tensor = result
                fixup_zero_kv_rows(
                    out_tensor,
                    lse_tensor,
                    forward_batch.prefix_chunk_seq_lens[chunk_idx],
                    self.forward_prefill_metadata.cum_seq_lens,
                    self.forward_prefill_metadata.max_seq_len,
                )

            return result
        else:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self.q_data_type,
            )
            return self._run_prefill_kernel(
                q=q,
                k=k,
                v=v,
                layer=layer,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_prefill_metadata.cum_seq_lens,
                max_q_len=self.forward_prefill_metadata.max_seq_len,
                seq_lens_kv=self.forward_prefill_metadata.seq_lens,
                cum_seq_lens_kv=self.forward_prefill_metadata.cum_seq_lens,
                max_kv_len=self.forward_prefill_metadata.max_seq_len,
                is_causal=True,
                return_lse=forward_batch.mha_return_lse,
                out_buffer=out,
                o_sf_scale=1.0,
            )


class TRTLLMMLAMultiStepDraftBackend(FlashInferMLAMultiStepDraftBackend):
    """Multi-step draft backend for TRT-LLM MLA used by EAGLE."""

    # Per-step draft decode never reads seq_lens_cpu / seq_lens_sum; opt out so
    # decide_needs_cpu_seq_lens' OR over the backends stays False.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        backend: str = "trtllm-gen",
    ):
        super().__init__(model_runner, topk, speculative_num_steps)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
                backend=backend,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        if in_capture:
            return super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )
        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_out_graph(inner_fb)
