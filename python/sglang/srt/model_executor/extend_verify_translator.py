# Copyright 2026 SGLang Team
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
"""Extend-as-virtual-verify translator.

Runs qualifying small-M EXTEND forwards through the EXISTING monolithic
TARGET_VERIFY CUDA graphs (whole-forward, attention in-graph) instead of the
fully-eager extend path.

Identity used: an absorbed-MLA extend of M chain-causal new tokens over a
prefix of length P is mathematically identical to a target-verify forward of
M "draft" tokens. With draft block size B (= speculative_num_draft_tokens),
an extend req is decomposed into v = ceil(M/B) *virtual verify requests* that
all share the real request's req_to_token row (page table):

  - front-pad p = v*B - M rows at the START of the req's token segment
    (pad rows write KV to reserved slot 0, same convention as CUDA-graph
    batch padding; their outputs are discarded)
  - virtual req r (0-based) has pre-draft seq_len = P - p + B*r, so the
    backend's verify adjustment (+B) gives kv_len_r = P - p + B*(r+1)
  - the verify kernel places q row i of req r at position kv_len_r - B + i
    = P - p + B*r + i; real token j (global padded row p + j) therefore sits
    at position P + j with causal window P + j + 1: exactly extend semantics
    (full prefix + earlier new tokens + self). Pad rows (rows < p of req 0)
    attend only to prefix KV; garbage out, sliced away.

Requires: DFlash (chain-causal verify, no tree mask on trtllm/tokenspeed
backends), graphs captured with CaptureHiddenMode.FULL (aux hidden for the
draft), P >= p per request.

Gains: one CUDA-graph replay per extend forward (zero per-layer host work,
kills cross-rank eager-dispatch skew), tokenspeed attention stays in-graph,
ZERO new device memory (reuses the existing verify graphs and buffers).

Enable with SGLANG_EXTEND_VERIFY_GRAPH=1 (per-hit logs with
SGLANG_EXTEND_VERIFY_GRAPH_LOG=1).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes")


class ExtendVerifyGraphTranslator:
    """Translate qualifying EXTEND batches into virtual TARGET_VERIFY batches
    and replay them on the existing decode/verify CUDA graphs."""

    @staticmethod
    def maybe_create(model_runner: "ModelRunner") -> Optional[
        "ExtendVerifyGraphTranslator"
    ]:
        if not _env_bool("SGLANG_EXTEND_VERIFY_GRAPH"):
            return None
        reason = ExtendVerifyGraphTranslator._unsupported_reason(model_runner)
        if reason is not None:
            log_info_on_rank0(
                logger,
                f"[extend-verify-graph] lever requested but DISABLED: {reason}",
            )
            return None
        translator = ExtendVerifyGraphTranslator(model_runner)
        log_info_on_rank0(
            logger,
            "[extend-verify-graph] ENABLED: extends with <= "
            f"{translator.max_tokens} tokens replay TARGET_VERIFY graphs "
            f"(block={translator.block}, capture_bs={translator.capture_bs}).",
        )
        return translator

    @staticmethod
    def _unsupported_reason(model_runner: "ModelRunner") -> Optional[str]:
        if model_runner.is_draft_worker:
            return "draft worker"
        runner = model_runner.graph_runner
        if runner is None:
            return "no CUDA graph runner"
        if runner.capture_forward_mode == ForwardMode.TARGET_VERIFY:
            # Verify graphs must be chain-causal (linear drafts): DFlash only.
            if not model_runner.spec_algorithm.is_dflash():
                return (
                    "TARGET_VERIFY graphs require DFLASH (chain-causal verify); "
                    f"got {model_runner.spec_algorithm}"
                )
            if runner.capture_hidden_mode != CaptureHiddenMode.FULL:
                return "graphs not captured with CaptureHiddenMode.FULL"
        elif runner.capture_forward_mode == ForwardMode.DECODE:
            # Non-speculative: decode graphs are single-token chain-causal by
            # construction (block=1 decomposition).
            if not model_runner.spec_algorithm.is_none():
                return "DECODE-mode graphs with a spec algorithm are unsupported"
            if runner.num_tokens_per_bs != 1:
                return "DECODE-mode graphs must have num_tokens_per_bs == 1"
        else:
            return f"unsupported capture mode {runner.capture_forward_mode}"
        if getattr(runner, "disable_padding", False):
            return "graph runner has padding disabled"
        return None

    def __init__(self, model_runner: "ModelRunner"):
        self.model_runner = model_runner
        self.runner = model_runner.graph_runner
        self.capture_mode = self.runner.capture_forward_mode
        self.is_verify_mode = self.capture_mode == ForwardMode.TARGET_VERIFY
        self.block = int(self.runner.num_tokens_per_bs)
        self.capture_bs = list(self.runner.capture_bs)
        self.max_bs = max(self.capture_bs)
        self.max_tokens = self.max_bs * self.block
        self.device = model_runner.device
        self.log_hits = _env_bool("SGLANG_EXTEND_VERIFY_GRAPH_LOG")

        if self.is_verify_mode:
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput

            # Static spec-info stub, mirrors what get_spec_info() built at
            # capture time (draft_token/positions are unused at replay; the
            # trtllm-family kernels use their built-in causal path, no custom
            # mask).
            self._spec_info_stub = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=self.block,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            self.required_hidden_mode = CaptureHiddenMode.FULL
            # Verify convention: seq_lens are PRE-draft; backend adds +block.
            self._seq_len_offset = 0
        else:
            self._spec_info_stub = None
            self.required_hidden_mode = CaptureHiddenMode.NULL
            # Decode convention: seq_lens INCLUDE the current token.
            self._seq_len_offset = self.block  # == 1

        self.num_hits = 0
        self.num_rejects = 0

    # ------------------------------------------------------------------ gate

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        reason = self._reject_reason(forward_batch)
        if reason is None:
            return True
        self.num_rejects += 1
        if self.log_hits:
            logger.info(
                "[extend-verify-graph] reject (bs=%d, extend_lens=%s): %s",
                forward_batch.batch_size,
                forward_batch.extend_seq_lens_cpu,
                reason,
            )
        return False

    def _reject_reason(self, forward_batch: ForwardBatch) -> Optional[str]:
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return f"forward_mode {forward_batch.forward_mode}"
        if forward_batch.return_logprob:
            return "return_logprob"
        if (
            forward_batch.input_embeds is not None
            or forward_batch.replace_embeds is not None
        ):
            return "input_embeds/replace_embeds"
        # mm_inputs is a per-req list on multimodal-arch models (KimiK25
        # wrapper) — all-None means a pure text batch.
        if forward_batch.mm_inputs is not None and any(
            x is not None for x in forward_batch.mm_inputs
        ):
            return "multimodal inputs"
        if forward_batch.capture_hidden_mode != self.required_hidden_mode:
            # Verify mode: DFlash target extends always request FULL aux-hidden
            # capture; anything else would mismatch the captured graphs.
            # Decode mode: only NULL is supported.
            return (
                f"capture_hidden_mode {forward_batch.capture_hidden_mode} != "
                f"{self.required_hidden_mode}"
            )
        extend_lens = forward_batch.extend_seq_lens_cpu
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        if extend_lens is None or prefix_lens is None:
            return "missing extend/prefix lens"
        if len(extend_lens) != forward_batch.batch_size:
            return "extend_lens/batch_size mismatch"

        block = self.block
        total_virtual = 0
        for m, p in zip(extend_lens, prefix_lens):
            m = int(m)
            if m <= 0:
                return "empty extend req"
            v = -(-m // block)
            pad = v * block - m
            if int(p) < pad:
                # Front-padding borrows pad positions from the prefix.
                return f"prefix {int(p)} < pad {pad}"
            total_virtual += v
        if total_virtual > self.max_bs:
            return f"virtual bs {total_virtual} > max {self.max_bs}"
        return None

    # -------------------------------------------------------------- translate

    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        block = self.block
        device = forward_batch.input_ids.device
        extend_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        prefix_lens = [int(x) for x in forward_batch.extend_prefix_lens_cpu]
        bs_real = forward_batch.batch_size

        input_ids_pieces = []
        positions_pieces = []
        out_cache_loc_pieces = []
        req_pool_pieces = []
        seq_lens_list = []  # python ints, used for both cpu+gpu tensors
        real_rows = []  # global padded row idx of each real token, in order
        last_rows = []  # global padded row idx of last real token per req

        tok_off = 0  # offset into the original flat token axis
        row_off = 0  # offset into the virtual padded token axis
        for i in range(bs_real):
            m, p_len = extend_lens[i], prefix_lens[i]
            v = -(-m // block)
            pad = v * block - m
            n = v * block

            ids = forward_batch.input_ids[tok_off : tok_off + m]
            locs = forward_batch.out_cache_loc[tok_off : tok_off + m]
            if pad:
                input_ids_pieces.append(ids.new_zeros((pad,)))
                out_cache_loc_pieces.append(locs.new_zeros((pad,)))
            input_ids_pieces.append(ids)
            out_cache_loc_pieces.append(locs)

            # Row j of this req's padded segment sits at position
            # (P - pad) + j; real token k (row pad+k) at position P + k.
            positions_pieces.append(
                torch.arange(
                    p_len - pad, p_len - pad + n, dtype=torch.int64, device=device
                )
            )
            req_pool_pieces.append(
                forward_batch.req_pool_indices[i : i + 1].expand(v)
            )
            # Verify convention: pre-draft lens (backend adds +block).
            # Decode convention: lens include the current token (+1).
            seq_lens_list.extend(
                p_len - pad + block * r + self._seq_len_offset for r in range(v)
            )

            real_rows.extend(range(row_off + pad, row_off + n))
            last_rows.append(row_off + n - 1)
            tok_off += m
            row_off += n

        bs_virtual = row_off // block
        seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64)
        seq_lens_gpu = seq_lens_cpu.to(device, non_blocking=True)
        translated = dataclasses.replace(
            forward_batch,
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs_virtual,
            input_ids=torch.cat(input_ids_pieces),
            positions=torch.cat(positions_pieces),
            out_cache_loc=torch.cat(out_cache_loc_pieces),
            req_pool_indices=torch.cat(req_pool_pieces),
            seq_lens=seq_lens_gpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(sum(seq_lens_list)),
            orig_seq_lens=seq_lens_gpu,
            spec_info=self._spec_info_stub,
            capture_hidden_mode=self.required_hidden_mode,
            # Clear extend-shaped metadata so any accidental consumer fails
            # loudly instead of silently reading mismatched shapes.
            extend_num_tokens=bs_virtual * block,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_start_loc=None,
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            extend_logprob_start_lens_cpu=None,
        )

        # Graph-coverage assert (footgun: silently measuring eager). NOTE: do
        # NOT use runner.can_run() here — it caps real batches at
        # native_max_bs by design; translated batches may use the full
        # capture list (extra SGLANG_EXTEND_GRAPH_BS buckets).
        if bs_virtual > self.max_bs:
            raise RuntimeError(
                f"[extend-verify-graph] bs_virtual={bs_virtual} > max capture "
                f"bucket {self.max_bs} — can_run() gate out of sync."
            )
        if self.runner.capture_hidden_mode != self.required_hidden_mode:
            raise RuntimeError(
                "[extend-verify-graph] runner capture_hidden_mode "
                f"{self.runner.capture_hidden_mode} != {self.required_hidden_mode} "
                "(recaptured under a different mode?)"
            )

        out = self.runner.replay(translated)
        assert isinstance(out, LogitsProcessorOutput)

        self.num_hits += 1
        if self.log_hits:
            logger.info(
                "[extend-verify-graph] hit #%d: bs=%d tokens=%s -> virtual bs=%d "
                "(padded to graph bs=%d)",
                self.num_hits,
                bs_real,
                extend_lens,
                bs_virtual,
                self.runner.bs,
            )

        # Un-translate: extend contract is next-token logits per request plus
        # (for DFlash) aux hidden states aligned 1:1 with out_cache_loc rows.
        last_rows_t = torch.tensor(last_rows, dtype=torch.int64, device=device)
        next_token_logits = out.next_token_logits.index_select(0, last_rows_t)
        hidden_states = None
        if out.hidden_states is not None:
            real_rows_t = torch.tensor(real_rows, dtype=torch.int64, device=device)
            hidden_states = out.hidden_states.index_select(0, real_rows_t)
        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=hidden_states,
        )
