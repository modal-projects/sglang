"""DFlashWorkerV2 with the ROCm fast paths: packed draft argmax, fused draft-KV
store, fused post-verify bookkeeping, and graphed draft prep / draft-KV tail."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.k3_dflash_kv_store import rope_store_prefix_valid
from sglang.kernels.ops.speculative.k3_dflash_post import (
    compact_draft_lens,
    mamba_track_steps,
    verify_argmax,
)
from sglang.srt.layers import layernorm as _ln
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_parallel, mamba_track_grid
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.k3_dflash_prep_graph import K3DFlashPrepGraph
from sglang.srt.speculative.spec_utils import draft_pp_context, draft_tp_context

logger = logging.getLogger(__name__)

_TAIL_GRAPH_MAX_BS = 64
_TAIL_GRAPH_MAX_SIGNATURES = 64
_TAIL_GRAPH_WARMUP_STEPS = 3


class DFlashWorkerV2ROCm(DFlashWorkerV2):
    _packed_draft_argmax = True
    # Publish the new seq lens (the event the scheduler host-waits on) before
    # the mamba-state commit so the commit overlaps the scheduler's host work.
    _publish_before_mamba_commit = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prep_graph: Optional[K3DFlashPrepGraph] = None
        self._tail_graph_on = (
            self.lilicorr is None and not get_parallel().attn_dp_enabled
        )
        self._tail_graphs: dict = {}
        self._tail_graph_pool = None
        self._tail_commit_lens: Optional[torch.Tensor] = None

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        self._prep_graph = None
        if self._draft_sampler is not None:
            with draft_pp_context(), draft_tp_context(self.draft_parallel_layout):
                self._prep_graph = K3DFlashPrepGraph.maybe_build(self)

    def _draft_block_eager(self, *, batch, draft_input, bs, block_size, **kwargs):
        out = None
        if self._prep_graph is not None:
            out = self._prep_graph.run(batch=batch, draft_input=draft_input, bs=bs)
        if out is None:
            return super()._draft_block_eager(
                batch=batch,
                draft_input=draft_input,
                bs=bs,
                block_size=block_size,
                **kwargs,
            )
        kv_loc_plan = self._plan_block_kv_locs(
            batch=batch,
            draft_input=draft_input,
            prefix_lens=out[5],
            verify_out_cache_loc=out[2],
            block_size=block_size,
        )
        return (*out, kv_loc_plan)

    def _append_target_hidden_to_draft_kv_by_loc(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        extend_lens: Optional[torch.Tensor] = None,
    ) -> None:
        if (
            self._tail_graph_on
            and commit_lens is not None
            and cache_loc_2d is not None
            and self._replay_tail_graph(
                target_hidden, cache_loc, cache_loc_2d, positions, commit_lens
            )
        ):
            return
        super()._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
            extend_lens=extend_lens,
        )

    def _replay_tail_graph(
        self, target_hidden, cache_loc, cache_loc_2d, positions, commit_lens
    ) -> bool:
        """Replay the post-verify draft-KV append as one graph per input
        signature, captured after a few eager steps. Every input but
        ``commit_lens`` (staged into a static buffer) has a stable address per
        batch size. False: the caller runs the eager append."""
        bs = cache_loc_2d.shape[0]
        if not 0 < bs <= _TAIL_GRAPH_MAX_BS:
            return False
        sig = (
            target_hidden.data_ptr(),
            tuple(target_hidden.shape),
            tuple(target_hidden.stride()),
            target_hidden.dtype,
            cache_loc.data_ptr(),
            cache_loc_2d.data_ptr(),
            tuple(cache_loc_2d.shape),
            positions.data_ptr(),
            positions.dtype,
            cache_loc.dtype,
            cache_loc_2d.dtype,
        )
        st = self._tail_graphs.get(sig)
        if st is None:
            if len(self._tail_graphs) >= _TAIL_GRAPH_MAX_SIGNATURES:
                return False
            st = self._tail_graphs[sig] = {"warm": 0, "graph": None}
        if self._tail_commit_lens is None:
            self._tail_commit_lens = torch.zeros(
                (_TAIL_GRAPH_MAX_BS,), dtype=torch.int32, device=target_hidden.device
            )
        commit_static = self._tail_commit_lens[:bs]
        if st["graph"] is None:
            st["warm"] += 1
            if st["warm"] <= _TAIL_GRAPH_WARMUP_STEPS or st.get("failed"):
                return False
        commit_static.copy_(commit_lens)
        if st["graph"] is None:
            if self._tail_graph_pool is None:
                self._tail_graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            try:
                torch.cuda.synchronize()
                with torch.cuda.graph(
                    graph, pool=self._tail_graph_pool, capture_error_mode="thread_local"
                ):
                    super()._append_target_hidden_to_draft_kv_by_loc(
                        target_hidden=target_hidden,
                        cache_loc=cache_loc,
                        cache_loc_2d=cache_loc_2d,
                        positions=positions,
                        commit_lens=commit_static,
                    )
            except Exception:
                logger.warning(
                    "DFlash draft-KV tail graph capture failed for bs=%d; "
                    "running it eagerly.",
                    bs,
                    exc_info=True,
                )
                st["failed"] = True
                return False
            st["graph"] = graph
        st["graph"].replay()
        return True

    def _kv_fuse_store(self, attn, k, v, positions, cache_loc_2d, commit_lens) -> bool:
        """AITER k-norm on the strided K slice, then one bf16-exact RoPE +
        prefix-valid KV write. False (nothing written) if ineligible."""
        rope = getattr(attn, "rotary_emb", None)
        pool = self.draft_model_runner.token_to_kv_pool
        hd = int(attn.head_dim)
        norm = attn.k_norm
        if not (
            _ln._use_aiter
            and getattr(getattr(norm, "_forward_method", None), "__func__", None)
            is _ln.RMSNorm.forward_aiter
            and not getattr(norm, "_fused_pad_kernel", None)
            and not _ln.is_batch_invariant_mode_enabled()
            and norm.weight.dtype == torch.bfloat16
            and int(attn.num_kv_heads) == 1
            and k.dtype == v.dtype == torch.bfloat16
            and k.dim() == 2
            and k.shape[1] == hd
            and k.stride(1) == 1
            and v.stride(1) == 1
            and getattr(rope, "use_fallback_kernel", False)
            and getattr(rope, "is_neox_style", False)
            and int(getattr(rope, "rotary_dim", -1)) == hd
            and int(getattr(rope, "head_size", -1)) == hd
            and getattr(pool, "store_dtype", None) == torch.bfloat16
            and getattr(pool, "dtype", None) == torch.bfloat16
            and not getattr(pool, "use_hnd", False)
        ):
            return False
        k_normed = _ln.rms_norm(k, norm.weight.data, norm.variance_epsilon)
        rope.cos_sin_cache = rope.cos_sin_cache.to(k.device)
        layer = attn.attn.layer_id - pool.start_layer
        rope_store_prefix_valid(
            k_normed.view(-1, 1, hd),
            v.view(-1, 1, hd),
            positions,
            rope.cos_sin_cache_as(torch.bfloat16),
            cache_loc_2d,
            commit_lens,
            pool.k_buffer[layer],
            pool.v_buffer[layer],
        )
        return True

    def _mamba_commit_steps(
        self,
        *,
        batch: ScheduleBatch,
        commit_lens: torch.Tensor,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        has_track = batch.mamba_track_indices is not None
        return mamba_track_steps(
            commit_lens,
            seq_lens_pre_verify,
            seq_lens_post_verify,
            mamba_track_grid(batch.tree_cache.page_size) if has_track else 1,
            has_track,
        )

    def _verify_argmax(self, next_token_logits: torch.Tensor) -> torch.Tensor:
        if (
            next_token_logits.dtype == torch.float32
            and next_token_logits.dim() == 2
            and next_token_logits.stride(1) == 1
        ):
            return verify_argmax(next_token_logits)
        return super()._verify_argmax(next_token_logits)

    def _compact_draft_prefix_lens(
        self, prefix_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return compact_draft_lens(
            prefix_lens, int(self.draft_window_size), self.page_size
        )
