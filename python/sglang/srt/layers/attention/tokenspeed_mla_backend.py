# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

"""Attention backend for the tokenspeed-mla CuTe DSL kernels on Blackwell.

Subclasses :class:`TRTLLMMLABackend` and overrides only ``_run_decode_kernel``
and ``_run_prefill_kernel``. All metadata, KV-cache layout, CUDA-graph
plumbing, FP8 quantize/rope, draft-extend padding, and chunked-prefix
dispatch are inherited unchanged from the parent.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.fp8_quantize import fp8_quantize
from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.utils import mla_quantize_and_rope_for_fp8
from sglang.srt.utils import is_flashinfer_available, is_tokenspeed_mla_available

if is_flashinfer_available():
    import flashinfer.rope as _flashinfer_rope

if is_tokenspeed_mla_available():
    import tokenspeed_mla

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

logger = logging.getLogger(__name__)


# Workspace upper bound for tokenspeed_mla_decode:
#   num_sms * num_heads * max_q_len * (kv_lora_rank + 1) * sizeof(float32)
# MAX_Q_LEN=8 covers EAGLE3 num_draft_tokens=4 plus headroom.
_TOKENSPEED_MAX_Q_LEN = 8

_g_tokenspeed_workspace: dict[torch.device, torch.Tensor] = {}

# Separate workspace for the padded-extend path (q_len buckets > 8). Kept
# apart from `_g_tokenspeed_workspace` on purpose: the decode workspace may be
# baked into captured decode CUDA graphs, so it must never be reallocated;
# this one is only used by eagerly-launched extend kernels and may grow.
_g_tokenspeed_extend_workspace: dict[torch.device, torch.Tensor] = {}


def _get_tokenspeed_workspace(
    device: torch.device, num_heads: int, kv_lora_rank: int
) -> torch.Tensor:
    needed = (
        tokenspeed_mla.get_num_sm(device)
        * num_heads
        * _TOKENSPEED_MAX_Q_LEN
        * (kv_lora_rank + 1)
        * 4
    )
    existing = _g_tokenspeed_workspace.get(device)
    if existing is None or existing.numel() < needed:
        _g_tokenspeed_workspace[device] = torch.empty(
            needed, dtype=torch.int8, device=device
        )
    return _g_tokenspeed_workspace[device]


def _tokenspeed_extend_workspace_size(
    device: torch.device,
    batch_size: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_seq_len: int,
) -> int:
    """Exact workspace bytes tokenspeed_mla_decode will demand for this shape.

    Mirrors the fold/tiler/split-kv selection inside tokenspeed_mla_decode so
    the assert there can never fire. All helpers are cached host-side math.
    """
    from tokenspeed_mla.mla_decode import _get_split_kv_and_workspace_size
    from tokenspeed_mla.mla_helpers import (
        get_mla_decode_fold_sq_factor,
        select_mla_decode_tilers,
    )

    cc = torch.cuda.get_device_capability(device)
    mma_qk_tiler_mn, _ = select_mla_decode_tilers(
        num_heads, q_len, is_fp8=True, compute_capability=cc
    )
    fold = get_mla_decode_fold_sq_factor(num_heads, q_len, mma_qk_tiler_mn[0])
    _, workspace_size = _get_split_kv_and_workspace_size(
        batch_size,
        q_len // fold,
        num_heads * fold,
        kv_lora_rank,
        tokenspeed_mla.get_num_sm(device),
        max_seq_len,
        torch.float8_e4m3fn,
        mma_qk_tiler_mn,
    )
    return workspace_size


def _get_tokenspeed_extend_workspace(
    device: torch.device, needed: int
) -> torch.Tensor:
    existing = _g_tokenspeed_extend_workspace.get(device)
    if existing is None or existing.numel() < max(needed, 1):
        _g_tokenspeed_extend_workspace[device] = torch.empty(
            max(needed, 1), dtype=torch.int8, device=device
        )
    return _g_tokenspeed_extend_workspace[device]


@dataclass
class TokenspeedPaddedExtendMetadata:
    """Metadata for the padded-extend path (absorbed-MLA extends run on the
    paged tokenspeed decode kernel at a bucketed uniform q_len)."""

    q_len_bucket: int
    # Real (unpadded) per-request query lengths / cumulative offsets.
    seq_lens_q: torch.Tensor  # [bs] int32, device
    cu_seqlens_q: torch.Tensor  # [bs+1] int32, device
    sum_seq_lens_q: int
    # Padded KV lengths: seq_lens - extend_seq_lens + q_len_bucket. Same
    # convention as the draft_extend path: tail-padded query rows get causal
    # bounds past the real KV (garbage, discarded); real row j of request i
    # gets k_bound = prefix_i + j + 1, i.e. exact extend semantics.
    seq_lens_k: torch.Tensor  # [bs] int32, device
    block_kv_indices: torch.Tensor  # [bs, max_blocks] int32, device
    max_seq_len_k: int  # max over requests of padded KV length


# TODO(Qiaolin-Yu): Merge this attention backend into trtllm_mla_backend.py
# once the same CuteDSL kernels in flashinfer_trtllm are stable
# and there is no performance gap compared to this backend.
class TokenspeedMLABackend(TRTLLMMLABackend):
    """tokenspeed-mla CuTe DSL attention backend (Blackwell SM100, FP8 KV)."""

    def __init__(
        self,
        model_runner: "ModelRunner",
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        if self.data_type != torch.float8_e4m3fn:
            raise ValueError(
                "tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got data_type={self.data_type}."
            )
        if self.page_size not in (32, 64):
            raise ValueError(
                "tokenspeed_mla backend requires page_size in {32, 64}, "
                f"got page_size={self.page_size}."
            )

        # Padded-extend path: absorbed-MLA extends (piecewise CUDA graphs, or
        # disable_chunked_prefix_cache) run on the paged fp8 tokenspeed decode
        # kernel at a bucketed uniform q_len instead of falling back to the
        # flashinfer MLA impl (which casts the WHOLE per-layer KV pool
        # fp8->bf16 on every forward — linear in pool size).
        self.padded_extend_enabled = (
            envs.SGLANG_TOKENSPEED_PADDED_EXTEND.get()
            and not skip_prefill
            and is_tokenspeed_mla_available()
        )
        self.extend_q_len_buckets: list[int] = []
        self.forward_padded_extend_metadata: Optional[
            TokenspeedPaddedExtendMetadata
        ] = None
        self._warned_adhoc_buckets: set[int] = set()
        if self.padded_extend_enabled:
            self.extend_q_len_buckets = sorted(
                int(tok)
                for tok in envs.SGLANG_TOKENSPEED_EXTEND_BUCKETS.get().split(",")
                if tok.strip()
            )
            assert self.extend_q_len_buckets, (
                "SGLANG_TOKENSPEED_PADDED_EXTEND=1 requires a non-empty "
                "SGLANG_TOKENSPEED_EXTEND_BUCKETS list"
            )
            assert all(b > 0 and b % 8 == 0 for b in self.extend_q_len_buckets), (
                "SGLANG_TOKENSPEED_EXTEND_BUCKETS must be positive multiples "
                f"of 8, got {self.extend_q_len_buckets}"
            )
            logger.info(
                "[tokenspeed-padded-extend] ENABLED q_len buckets=%s",
                self.extend_q_len_buckets,
            )
        elif envs.SGLANG_TOKENSPEED_PADDED_EXTEND.get():
            logger.info(
                "[tokenspeed-padded-extend] requested but inactive "
                "(skip_prefill=%s, tokenspeed available=%s)",
                skip_prefill,
                is_tokenspeed_mla_available(),
            )

        self._tokenspeed_workspace: Optional[torch.Tensor] = None
        if is_tokenspeed_mla_available():
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                self.device, self.num_q_heads, self.kv_lora_rank
            )

            # Pre-JIT the prefill kernel variants. Each cute.compile takes 1-2
            # min; without warm-up the first request trips the 300 s scheduler
            # watchdog.
            _compile_prefill_kernel = tokenspeed_mla.mla_prefill._compile_prefill_kernel
            _compiled_kernels = tokenspeed_mla.mla_prefill._compiled_kernels
            head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            enable_ex2_emulation = tokenspeed_mla.mla_prefill._enable_ex2_emulation()
            use_pdl = is_arch_support_pdl()
            for is_causal in (True, False):
                for return_lse in (True, False):
                    # Non-causal is only entered from the chunked-prefix
                    # branch, which always asks for the LSE.
                    if is_causal is False and return_lse is False:
                        continue
                    # Runtime feeds fp8_e4m3fn q/k/v
                    config = (
                        torch.float8_e4m3fn,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl,
                        enable_ex2_emulation,
                    )
                    if config in _compiled_kernels:
                        continue
                    _compiled_kernels[config] = _compile_prefill_kernel(
                        torch.float8_e4m3fn,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl=use_pdl,
                        enable_ex2_emulation=enable_ex2_emulation,
                    )

            if self.padded_extend_enabled:
                self._precompile_padded_extend_kernels()

    def _fused_rope_fp8_quantize(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        is_neox: bool,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RoPE + FP8 quantize that also packs nope+pe along the last
        dim, so FMHA consumes contig FP8 Q/K without an extra concat or cast.
        """
        num_heads = q_nope.shape[1]
        seq_len = q_nope.shape[0]
        q_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_nope.device,
        )
        k_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=k_nope.device,
        )
        if seq_len == 0:
            return q_fp8, k_fp8

        # Broadcast the shared latent k_pe across heads — RoPE is position-only
        # so per-head outputs are identical, and the cache write below reuses
        # head 0.
        if k_pe.dim() == 3 and k_pe.shape[1] == 1:
            k_pe_expanded = k_pe.expand(-1, num_heads, -1)
        else:
            k_pe_expanded = k_pe

        _flashinfer_rope.mla_rope_quantize_fp8(
            q_rope=q_pe,
            k_rope=k_pe_expanded,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=is_neox,
            quantize_dtype=torch.float8_e4m3fn,
            q_rope_out=q_fp8[..., qk_nope_head_dim:],
            k_rope_out=k_fp8[..., qk_nope_head_dim:],
            q_nope_out=q_fp8[..., :qk_nope_head_dim],
            k_nope_out=k_fp8[..., :qk_nope_head_dim],
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=is_arch_support_pdl(),
        )
        return q_fp8, k_fp8

    def prepare_prefill_qkv(
        self,
        *,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        layer: "DeepseekV2AttentionMLA",
        forward_batch: "ForwardBatch",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build FP8 (Q, K, V) for the FMHA kernel and write FP8 KV cache."""
        kv = layer.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, layer.num_local_heads, layer.qk_nope_head_dim + layer.v_head_dim
        )
        k_nope = kv[..., : layer.qk_nope_head_dim]
        v_bf16 = kv[..., layer.qk_nope_head_dim :]
        q_nope = q[..., : layer.qk_nope_head_dim]

        q_fp8, k_fp8 = self._fused_rope_fp8_quantize(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            cos_sin_cache=layer.rotary_emb.cos_sin_cache,
            positions=positions,
            is_neox=getattr(layer.rotary_emb, "is_neox_style", True),
            qk_nope_head_dim=layer.qk_nope_head_dim,
            qk_rope_head_dim=layer.qk_rope_head_dim,
        )
        v_fp8 = fp8_quantize(v_bf16, enable_pdl=is_arch_support_pdl())

        # k_pe is shared across heads (RoPE is position-only), so head 0
        # reproduces the original [tokens, 1, qk_rope] latent layout.
        kv_a_fp8 = fp8_quantize(kv_a, enable_pdl=is_arch_support_pdl())
        k_pe_fp8 = k_fp8[:, 0:1, layer.qk_nope_head_dim :]
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            kv_a_fp8.unsqueeze(1),
            k_pe_fp8,
        )
        return q_fp8, k_fp8, v_fp8

    def pack_prefix_chunk_kv(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack strided ``k_nope``+``k_pe`` into contig FP8 K and quantize
        strided ``v`` into contig FP8 V in a single kernel.
        """
        return mla_kv_pack_quantize_fp8(
            k_nope, k_pe, v, enable_pdl=is_arch_support_pdl()
        )

    # ------------------------------------------------------------------
    # Padded-extend path (SGLANG_TOKENSPEED_PADDED_EXTEND=1)
    # ------------------------------------------------------------------

    def _precompile_padded_extend_kernels(self) -> None:
        """JIT-compile the decode-kernel variants the extend buckets will hit.

        `seq_len_q` (and `is_workspace_size_zero`) are compile-time constants
        of the CuteDSL kernel; CuteDSL 4.5.1 has no disk cache, so an
        un-warmed bucket would stall the first matching request for 1-2 min.
        The workspace-zero flag depends on the runtime (batch, kv_len) via the
        wave-aware split heuristic, so evaluate the heuristic over the
        serving envelope and compile the union of variants it selects.
        """
        from tokenspeed_mla.mla_decode import _get_compiled_mla_kernel
        from tokenspeed_mla.mla_helpers import (
            get_mla_decode_fold_sq_factor,
            select_mla_decode_tilers,
        )

        cc = torch.cuda.get_device_capability(self.device)
        use_pdl = is_arch_support_pdl()
        seq_len_grid = sorted(
            {4096, 16384, 65536, self.max_context_len, max(self.max_context_len, 1)}
        )
        bs_grid = (1, 2, 3, 4, 6, 8, 12, 16)
        for q_len in self.extend_q_len_buckets:
            ws_variants = set()
            for bs in bs_grid:
                for max_seq in seq_len_grid:
                    ws_variants.add(
                        _tokenspeed_extend_workspace_size(
                            self.device,
                            bs,
                            q_len,
                            self.num_q_heads,
                            self.kv_lora_rank,
                            max_seq,
                        )
                        == 0
                    )
            mma_qk_tiler_mn, _ = select_mla_decode_tilers(
                self.num_q_heads, q_len, is_fp8=True, compute_capability=cc
            )
            fold = get_mla_decode_fold_sq_factor(
                self.num_q_heads, q_len, mma_qk_tiler_mn[0]
            )
            for ws_zero in sorted(ws_variants):
                logger.info(
                    "[tokenspeed-padded-extend] pre-compiling decode kernel "
                    "q_len=%d fold=%d ws_zero=%s (CuteDSL JIT, ~1-2 min)",
                    q_len,
                    fold,
                    ws_zero,
                )
                _get_compiled_mla_kernel(
                    torch_dtype=torch.float8_e4m3fn,
                    page_size=self.page_size,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    is_persistent=False,
                    is_var_seq=True,
                    is_var_split_kv=False,
                    skip_correction_threshold=0.0,
                    is_workspace_size_zero=ws_zero,
                    fold_sq_factor=fold,
                    causal_mask=True,
                    num_heads=self.num_q_heads,
                    seq_len_q=q_len,
                    cp_world=1,
                    use_pdl=use_pdl,
                    return_lse=False,
                    compute_capability=cc,
                )

    def _pick_extend_q_len_bucket(self, max_q_len: int) -> int:
        for bucket in self.extend_q_len_buckets:
            if bucket >= max_q_len:
                return bucket
        bucket = -(-max_q_len // 8) * 8
        if bucket not in self._warned_adhoc_buckets:
            self._warned_adhoc_buckets.add(bucket)
            logger.warning(
                "[tokenspeed-padded-extend] extend q_len %d exceeds largest "
                "configured bucket %d; ad-hoc bucket %d will JIT-compile on "
                "first use (~1-2 min stall). Extend "
                "SGLANG_TOKENSPEED_EXTEND_BUCKETS to cover it.",
                max_q_len,
                self.extend_q_len_buckets[-1],
                bucket,
            )
        return bucket

    def _padded_extend_applies(self, forward_batch: "ForwardBatch") -> bool:
        """True iff this batch dispatches to absorbed-MLA extend, i.e. the
        exact condition under which the parent would set
        fallback_to_flashinfer_impl (mirrors handle_attention_tokenspeed_mla
        + TRTLLMMLABackend.init_forward_metadata)."""
        if not self.padded_extend_enabled:
            return False
        fm = forward_batch.forward_mode
        if (
            not fm.is_extend()
            or fm.is_target_verify()
            or fm.is_draft_extend(include_v2=True)
        ):
            return False
        has_prefix = any(forward_batch.extend_prefix_lens_cpu)
        return (
            self.disable_chunked_prefix_cache and has_prefix
        ) or is_in_piecewise_cuda_graph()

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        if self._padded_extend_applies(forward_batch):
            self._init_padded_extend_metadata(forward_batch)
            return
        self.forward_padded_extend_metadata = None
        return super().init_forward_metadata(forward_batch)

    def _init_padded_extend_metadata(self, forward_batch: "ForwardBatch"):
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        extend_seq_lens = forward_batch.extend_seq_lens
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        max_q_len = int(max(extend_seq_lens_cpu))
        q_len_bucket = self._pick_extend_q_len_bucket(max_q_len)

        seq_lens_q = extend_seq_lens.to(torch.int32)
        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        sum_seq_lens_q = int(sum(extend_seq_lens_cpu))

        # draft_extend convention: pretend every request has q_len_bucket new
        # tokens so tail-padded rows get valid (garbage, discarded) causal
        # bounds while real rows attend exactly prefix + earlier new + self.
        seq_lens_k = (forward_batch.extend_prefix_lens + q_len_bucket).to(torch.int32)
        max_seq_len_k = int(max(forward_batch.extend_prefix_lens_cpu)) + q_len_bucket

        max_blocks = self._calc_padded_blocks(max_seq_len_k)
        block_kv_indices = self._create_block_kv_indices(
            bs,
            max_blocks,
            forward_batch.req_pool_indices,
            seq_lens_k,
            device,
        )

        # Parent extend paths must not see stale prefill metadata.
        self.forward_prefill_metadata = None
        self.forward_padded_extend_metadata = TokenspeedPaddedExtendMetadata(
            q_len_bucket=q_len_bucket,
            seq_lens_q=seq_lens_q,
            cu_seqlens_q=cu_seqlens_q,
            sum_seq_lens_q=sum_seq_lens_q,
            seq_lens_k=seq_lens_k,
            block_kv_indices=block_kv_indices,
            max_seq_len_k=max_seq_len_k,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fm = forward_batch.forward_mode
        metadata = self.forward_padded_extend_metadata
        if (
            metadata is None
            or not fm.is_extend()
            or fm.is_target_verify()
            or fm.is_draft_extend(include_v2=True)
        ):
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        # Padded-extend: fused rope + fp8 quantize (identical numerics to the
        # decode / target-verify paths), write fp8 KV, run the paged decode
        # kernel at the bucketed q_len, then unpad.
        assert self.data_type == torch.float8_e4m3fn
        assert all(x is not None for x in (q_rope, k_rope, cos_sin_cache)), (
            "padded tokenspeed extend requires the unrotated q/k fused-rope "
            "path (_fuse_rope_for_trtllm_mla must be True for this batch)"
        )
        assert llama_4_scaling is None, "llama_4_scaling unsupported here"

        num_tokens = q.shape[0]
        positions = forward_batch.positions[:num_tokens]
        q_fp8, k_fp8, k_rope_fp8 = mla_quantize_and_rope_for_fp8(
            q,
            q_rope,
            k.squeeze(1),
            k_rope.squeeze(1),
            positions,
            cos_sin_cache,
            is_neox,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        )

        if save_kv_cache:
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k_fp8, k_rope_fp8
            )

        bs = forward_batch.batch_size
        q_fp8 = q_fp8.view(-1, layer.tp_q_head_num, layer.head_dim)
        # Zero-init so pad rows hold valid fp8 values (uninitialized bytes can
        # decode to NaN); their outputs are discarded by the unpad below.
        padded_q = torch.zeros(
            (bs, metadata.q_len_bucket, layer.tp_q_head_num, layer.head_dim),
            dtype=q_fp8.dtype,
            device=q_fp8.device,
        )
        self.pad_draft_extend_query(
            q_fp8, padded_q, metadata.seq_lens_q, metadata.cu_seqlens_q
        )

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim).unsqueeze(1)

        raw_out = self._run_decode_kernel(
            query=padded_q,
            kv_cache=kv_cache,
            block_tables=metadata.block_kv_indices,
            seq_lens=metadata.seq_lens_k,
            max_seq_len=metadata.max_seq_len_k,
            layer=layer,
        )

        output = self._unpad_padded_extend_output(
            raw_out,
            metadata.cu_seqlens_q,
            metadata.seq_lens_q,
            metadata.sum_seq_lens_q,
        )
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _unpad_padded_extend_output(
        self,
        raw_out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seq_lens_q: torch.Tensor,
        sum_seq_lens_q: int,
    ) -> torch.Tensor:
        """Like unpad_draft_extend_output, but never routes through the small
        preallocated unpad_output_buffer (sized for draft tokens)."""
        saved = self.unpad_output_buffer
        self.unpad_output_buffer = None
        try:
            return self.unpad_draft_extend_output(
                raw_out, cu_seqlens_q, seq_lens_q, sum_seq_lens_q
            )
        finally:
            self.unpad_output_buffer = saved

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: "RadixAttention",
    ) -> torch.Tensor:
        k_scale = getattr(layer, "k_scale_float", None)
        if k_scale is None:
            k_scale = 1.0
        softmax_scale = float(layer.scaling) * float(k_scale)
        output_scale = float(k_scale)

        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        workspace = self._tokenspeed_workspace
        if query.shape[1] > _TOKENSPEED_MAX_Q_LEN:
            # Padded-extend buckets exceed the decode workspace sizing; use
            # the (growable, never graph-captured) extend workspace, sized by
            # the exact demand of this shape.
            needed = _tokenspeed_extend_workspace_size(
                query.device,
                query.shape[0],
                query.shape[1],
                self.num_q_heads,
                self.kv_lora_rank,
                int(max_seq_len),
            )
            workspace = _get_tokenspeed_extend_workspace(query.device, needed)
        return tokenspeed_mla.tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=int(max_seq_len),
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            enable_pdl=is_arch_support_pdl(),
        )

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
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
    ):  # Q/K/V arrive already in FP8 via the model-side fused path
        # (prepare_prefill_qkv / pack_prefix_chunk_kv); no quantize here.
        return tokenspeed_mla.tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens_kv,
            cum_seq_lens=cum_seq_lens_kv,
            max_seq_len=int(max_kv_len),
            batch_size=int(batch_size),
            softmax_scale=float(layer.scaling),
            is_causal=is_causal,
            return_lse=return_lse,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=int(max_q_len),
            enable_pdl=is_arch_support_pdl(),
        )


class TokenspeedMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):
    """Multi-step draft backend for tokenspeed_mla used by EAGLE."""

    def __init__(
        self, model_runner: "ModelRunner", topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # Parent populates self.attn_backends with TRT-LLM instances; replace
        # them with tokenspeed instances sharing the parent's index buffers.
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TokenspeedMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
