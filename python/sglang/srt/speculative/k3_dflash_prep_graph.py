"""One graph per draft batch size for the DFlash draft step's eager prep.

The prep (block ids / positions / verify cache locs, noise embedding, compact
window rebuild, draft runner load_batch) reads only device tensors with stable
addresses per batch size, so it is captured fed by three static input rows
(bonus token, committed seq len, req pool index). The draft runner's host-side
replay state is snapshotted at capture and re-installed per replay.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional

import torch

from sglang.kernels.ops.memory.small_copy import try_small_copy
from sglang.kernels.ops.speculative.dflash import _prepare_dflash_draft_block_unchecked
from sglang.kernels.ops.speculative.k3_dflash_post import compact_draft_lens
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_worker_v2 import _resolve_dflash_embedding_module
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context

_MAX_BS = 16


class K3DFlashPrepGraph:
    def __init__(self, worker) -> None:
        self.w = worker
        self.runner = worker.draft_model_runner.decode_cuda_graph_runner
        self.states: Dict[int, dict] = {}

    @classmethod
    def maybe_build(cls, worker) -> Optional[K3DFlashPrepGraph]:
        runner = getattr(worker.draft_model_runner, "decode_cuda_graph_runner", None)
        supported = (
            worker.use_compact_draft_cache
            and worker._draft_sampler is not None
            and worker.selector is None
            and worker.lilicorr is None
            and not worker._is_domino
            and not get_parallel().attn_dp_enabled
            and not enable_num_token_non_padded()
            and runner is not None
            and type(getattr(runner, "backend", None)).__name__
            in ("FullCudaGraphBackend", "BreakableCudaGraphBackend")
            and not getattr(runner, "ragged_verify_mode", False)
            and not getattr(runner, "enable_two_batch_overlap", False)
            and not getattr(runner, "enable_pdmux", False)
            and not getattr(runner, "require_mlp_tp_gather", False)
            and not getattr(runner, "dllm_uses_input_embeds", False)
            and worker.draft_model_runner.lora_manager is None
            and int(getattr(runner, "captured_req_width", -1)) == int(worker.block_size)
        )
        if not supported:
            return None
        pg = cls(worker)
        pg.capture_all()
        return pg if pg.states else None

    def capture_all(self) -> None:
        w = self.w
        bss = sorted({int(b) for b in self.runner.capture_bs if 0 < int(b) <= _MAX_BS})
        if not bss:
            return
        max_bs = max(bss)
        dev = w.device
        # Own references: the worker may later reallocate its buffers for a
        # larger eager batch; the graphs keep pointing at these.
        w._ensure_draft_block_buffers(max_bs)
        self.block_ids_buf = w._draft_block_ids_buf
        self.positions_buf = w._draft_block_positions_buf
        self.cache_loc_buf = w._draft_verify_out_cache_loc_buf
        self.seq_lens_cpu_buf = torch.empty((max_bs,), dtype=torch.int32)
        self.in_bonus = torch.zeros((max_bs,), dtype=torch.int64, device=dev)
        self.in_seq = torch.ones((max_bs,), dtype=torch.int64, device=dev)
        self.in_rpi = torch.zeros((max_bs,), dtype=torch.int64, device=dev)
        pool = torch.cuda.graph_pool_handle()

        stream = get_or_create_global_graph_capture_stream()
        with graph_capture(stream=stream) as ctx:
            for bs in reversed(bss):
                for _ in range(2):
                    torch.cuda.synchronize()
                    get_parallel().tp_group.barrier()
                    self._body(bs)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=ctx.stream):
                    self._body(bs)
                r = self.runner
                attn = r._replay_attn_backend()
                leaves = [attn] + list(getattr(attn, "attn_backend_list", None) or [])
                self.states[bs] = dict(
                    graph=g,
                    key=r._replay_graph_key,
                    raw_bs=r.raw_bs,
                    bs=r.bs,
                    raw_num_token=r.raw_num_token,
                    meta=[(b, getattr(b, "forward_metadata", None)) for b in leaves],
                )
        torch.cuda.synchronize()

    def _body(self, bs: int) -> None:
        """Mirrors DFlashWorkerV2._draft_block_eager up to the draft runner's
        load_batch, on ``bs`` rows of the static inputs."""
        w = self.w
        block_size = int(w.block_size)
        block_ids = self.block_ids_buf[:bs]
        positions_2d = self.positions_buf[:bs]
        cache_loc_2d = self.cache_loc_buf[:bs]
        prefix_lens = self.in_seq[:bs]
        rpi = self.in_rpi[:bs]

        _prepare_dflash_draft_block_unchecked(
            bonus_tokens=self.in_bonus[:bs],
            prefix_lens=prefix_lens,
            req_pool_indices=rpi,
            req_to_token=w.model_runner.req_to_token_pool.req_to_token,
            block_ids_out=block_ids,
            positions_out=positions_2d,
            cache_loc_out=cache_loc_2d,
            mask_token_id=int(w._mask_token_id),
        )
        with draft_tp_context(w.draft_parallel_layout):
            if w._full_embed_gpu is not None:
                noise = torch.nn.functional.embedding(block_ids, w._full_embed_gpu)
            else:
                target_model = w.target_worker.model_runner.model
                embed_module = unwrap_lora_layer(
                    _resolve_dflash_embedding_module(w.draft_model, target_model)
                )
                noise = embed_module(block_ids)
        if w._noise_embed_scale != 1.0:
            noise = noise * w._noise_embed_scale

        draft_prefix_lens, suffix_start = compact_draft_lens(
            prefix_lens, int(w.draft_window_size), w.page_size
        )
        w._rebuild_compact_draft_cache(
            req_pool_indices=rpi,
            prefix_lens=prefix_lens,
            draft_prefix_lens=draft_prefix_lens,
            verify_out_cache_loc_2d=cache_loc_2d,
            bs=bs,
            block_size=block_size,
            suffix_start=suffix_start,
        )
        # Host-side planning bound, read only at capture.
        seq_lens_cpu = self.seq_lens_cpu_buf[:bs]
        seq_lens_cpu.fill_(
            int(w.draft_window_size) + (w.page_size if w.page_size > 1 else 0)
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc_is_physical=True,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=rpi,
            seq_lens=draft_prefix_lens,
            out_cache_loc=cache_loc_2d.reshape(-1),
            seq_lens_sum=int(seq_lens_cpu.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            positions=positions_2d.reshape(-1),
            input_embeds=noise.view(-1, noise.shape[-1]),
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=w._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=None,
            global_num_token_non_padded_cpu=bs * block_size,
        )
        mr = w.draft_model_runner
        ctx = (
            contextlib.nullcontext()
            if has_forward_context()
            else forward_context(ForwardContext(attn_backend=mr.attn_backend))
        )
        with ctx, draft_tp_context(w.draft_parallel_layout):
            if not self.runner.can_run_graph(fb):
                raise RuntimeError(f"draft graph cannot run bs={bs}")
            self.runner.load_batch(fb)

    def run(self, *, batch, draft_input, bs: int):
        """Prep graph + draft graph replay; None if ``bs`` was not captured."""
        st = self.states.get(bs)
        if st is None:
            return None
        dsts = [self.in_bonus[:bs], self.in_seq[:bs], self.in_rpi[:bs]]
        srcs = [
            draft_input.bonus_tokens.reshape(-1),
            batch.seq_lens.reshape(-1),
            batch.req_pool_indices.reshape(-1),
        ]
        if not try_small_copy(dsts, srcs):
            for d, s in zip(dsts, srcs):
                d.copy_(s)
        st["graph"].replay()
        r = self.runner
        for b, md in st["meta"]:
            if md is not None:
                b.forward_metadata = md
        r.raw_bs = st["raw_bs"]
        r.bs = st["bs"]
        r.raw_num_token = st["raw_num_token"]
        r._replay_graph_key = st["key"]
        with r.backend.replay_session():
            r.backend.replay(st["key"], None)

        block_size = int(self.w.block_size)
        draft_next = self.w._draft_sampler.out[: bs * (block_size - 1)].view(
            bs, block_size - 1
        )
        cache_2d = self.cache_loc_buf[:bs]
        return (
            self.block_ids_buf[:bs],
            self.positions_buf[:bs].reshape(-1),
            cache_2d.reshape(-1),
            cache_2d,
            draft_next,
            batch.seq_lens,
        )
