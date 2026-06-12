import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import (
    DFlashDraftInputV2,
    _get_overlap_plan_stream,
)
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    generate_dflash_linear_token_bitmask,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.triton_ops.dflash_accept_bonus import (
    _compute_dflash_accept_bonus_triton_unchecked,
)
from sglang.srt.speculative.triton_ops.dflash_prepare_block import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import is_pin_memory_available

logger = logging.getLogger(__name__)


_DFLASH_GRAMMAR_MASK_RING_SIZE = 3


@dataclass
class _DFlashGrammarMaskSlot:
    cpu_buf: Optional[torch.Tensor] = None
    cpu_cap: int = 0
    gpu_buf: Optional[torch.Tensor] = None
    gpu_cap: int = 0
    gpu_device: Optional[torch.device] = None
    cache_key: Optional[tuple] = None
    pin_memory: Optional[bool] = None
    h2d_done: Optional[object] = None
    apply_done: Optional[object] = None


class DFlashWorkerV2(DFlashWorker):
    """DFLASH speculative decoding worker (spec-v2 overlap scheduling).

    This is intentionally implemented as a *separate* worker from the existing
    spec-v1 `DFlashWorker` (non-overlap), to keep the v1 path stable and to
    minimize risk while bringing up overlap scheduling.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        supports_gpu_triton = is_cuda() or is_hip()
        self._use_triton_prepare_block = supports_gpu_triton
        self._use_triton_accept_bonus = supports_gpu_triton
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._accept_len_buf: Optional[torch.Tensor] = None
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []
        self._grammar_draft_tail_cpu_buf: Optional[torch.Tensor] = None
        self._grammar_draft_tail_cpu_cap: int = 0
        self._grammar_draft_tail_cols: int = -1
        self._grammar_draft_tail_pin_memory: Optional[bool] = None
        self._grammar_vocab_mask_slots = [
            _DFlashGrammarMaskSlot() for _ in range(_DFLASH_GRAMMAR_MASK_RING_SIZE)
        ]
        self._grammar_vocab_mask_slot: int = 0

    def _ensure_grammar_draft_tail_cpu_buffer(
        self, bs: int, block_size: int
    ) -> torch.Tensor:
        cols = max(int(block_size) - 1, 0)
        pin_memory = is_pin_memory_available(self.device)
        if (
            self._grammar_draft_tail_cpu_buf is not None
            and self._grammar_draft_tail_cpu_cap >= int(bs)
            and self._grammar_draft_tail_cols == cols
            and self._grammar_draft_tail_pin_memory == pin_memory
        ):
            return self._grammar_draft_tail_cpu_buf[:bs]

        new_cap = max(
            int(bs),
            (
                self._grammar_draft_tail_cpu_cap * 2
                if self._grammar_draft_tail_cpu_cap > 0
                else int(bs)
            ),
        )
        self._grammar_draft_tail_cpu_buf = torch.empty(
            (new_cap, cols),
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._grammar_draft_tail_cpu_cap = new_cap
        self._grammar_draft_tail_cols = cols
        self._grammar_draft_tail_pin_memory = pin_memory
        return self._grammar_draft_tail_cpu_buf[:bs]

    @staticmethod
    def _grammar_vocab_mask_key(grammar, vocab_size: int) -> tuple:
        inner_grammar = getattr(grammar, "grammar", None)
        return (
            type(grammar),
            type(inner_grammar) if inner_grammar is not None else None,
            int(vocab_size),
        )

    @staticmethod
    def _event_is_done(event) -> bool:
        return event is None or bool(event.query())

    @staticmethod
    def _clear_done_event(event):
        return None if event is None or bool(event.query()) else event

    def _next_grammar_vocab_mask_slot(self) -> _DFlashGrammarMaskSlot:
        slot_count = len(self._grammar_vocab_mask_slots)
        start = self._grammar_vocab_mask_slot
        for offset in range(slot_count):
            idx = (start + offset) % slot_count
            slot = self._grammar_vocab_mask_slots[idx]
            if self._event_is_done(slot.h2d_done):
                slot.h2d_done = None
                slot.apply_done = self._clear_done_event(slot.apply_done)
                self._grammar_vocab_mask_slot = (idx + 1) % slot_count
                return slot

        slot = self._grammar_vocab_mask_slots[start]
        slot.h2d_done.synchronize()
        slot.h2d_done = None
        slot.apply_done = self._clear_done_event(slot.apply_done)
        self._grammar_vocab_mask_slot = (start + 1) % slot_count
        return slot

    def _ensure_grammar_vocab_mask_slot(
        self,
        *,
        grammar,
        vocab_size: int,
        bs: int,
        block_size: int,
    ) -> Optional[_DFlashGrammarMaskSlot]:
        rows = int(bs) * int(block_size)
        pin_memory = is_pin_memory_available(self.device)
        key = self._grammar_vocab_mask_key(grammar, vocab_size)
        slot = self._next_grammar_vocab_mask_slot()

        if (
            slot.cpu_buf is not None
            and slot.cpu_cap >= rows
            and slot.cache_key == key
            and slot.pin_memory == pin_memory
        ):
            return slot

        if slot.apply_done is not None:
            slot.apply_done.synchronize()
            slot.apply_done = None

        new_cap = max(
            rows,
            (slot.cpu_cap * 2 if slot.cpu_cap > 0 and slot.cache_key == key else rows),
        )
        vocab_mask = grammar.allocate_vocab_mask(
            vocab_size=vocab_size,
            batch_size=new_cap,
            device="cpu",
        )
        if vocab_mask is None:
            return None
        if pin_memory and not vocab_mask.is_pinned():
            vocab_mask = vocab_mask.pin_memory()

        slot.cpu_buf = vocab_mask
        slot.cpu_cap = int(vocab_mask.shape[0])
        slot.gpu_buf = None
        slot.gpu_cap = 0
        slot.gpu_device = None
        slot.cache_key = key
        slot.pin_memory = pin_memory
        return slot

    def _copy_grammar_vocab_mask_to_device(
        self,
        *,
        slot: _DFlashGrammarMaskSlot,
        cpu_vocab_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, Optional[object]]:
        rows = int(cpu_vocab_mask.shape[0])
        need_new_gpu_buf = (
            slot.gpu_buf is None
            or slot.gpu_cap < rows
            or slot.gpu_buf.dtype != cpu_vocab_mask.dtype
            or slot.gpu_buf.shape[1:] != cpu_vocab_mask.shape[1:]
            or slot.gpu_device != device
        )
        if need_new_gpu_buf:
            if slot.apply_done is not None:
                slot.apply_done.synchronize()
                slot.apply_done = None
            slot.gpu_buf = torch.empty(
                (slot.cpu_cap, *cpu_vocab_mask.shape[1:]),
                dtype=cpu_vocab_mask.dtype,
                device=device,
            )
            slot.gpu_cap = int(slot.gpu_buf.shape[0])
            slot.gpu_device = device

        gpu_vocab_mask = slot.gpu_buf[:rows]
        copy_stream, copy_stream_ctx = _get_overlap_plan_stream(device)
        current_stream = torch.get_device_module(device).current_stream()
        if copy_stream is not None:
            with copy_stream_ctx:
                if slot.apply_done is not None:
                    copy_stream.wait_event(slot.apply_done)
                    slot.apply_done = None
                gpu_vocab_mask.copy_(cpu_vocab_mask, non_blocking=True)
                ready = torch.get_device_module(device).Event()
                ready.record()
                slot.h2d_done = ready
            return gpu_vocab_mask, ready

        if slot.apply_done is not None:
            current_stream.wait_event(slot.apply_done)
            slot.apply_done = None
        gpu_vocab_mask.copy_(cpu_vocab_mask, non_blocking=True)
        ready = torch.get_device_module(device).Event()
        ready.record()
        slot.h2d_done = ready
        return gpu_vocab_mask, None

    def _ensure_accept_bonus_buffers(self, bs: int) -> None:
        if self._accept_bonus_buffer_cap >= int(bs):
            return

        new_cap = max(
            int(bs),
            (
                self._accept_bonus_buffer_cap * 2
                if self._accept_bonus_buffer_cap > 0
                else int(bs)
            ),
        )
        device = self.device
        block_size = int(self.block_size)
        self._accept_len_buf = torch.empty((new_cap,), dtype=torch.int32, device=device)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._new_seq_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._accept_bonus_buffer_cap = new_cap

    def _next_accept_bonus_buffers(self, bs: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_accept_bonus_buffers(bs)
        assert self._accept_len_buf is not None
        slot = self._accept_bonus_buffer_slot
        self._accept_bonus_buffer_slot = (slot + 1) % 2
        return (
            self._accept_len_buf[:bs],
            self._commit_lens_bufs[slot][:bs],
            self._bonus_id_bufs[slot][:bs],
            self._out_tokens_bufs[slot][:bs],
            self._new_seq_lens_bufs[slot][:bs],
        )

    def _validate_phase1_sampling_support(
        self, model_worker_batch: ScheduleBatch
    ) -> None:
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is None or sampling_info.is_all_greedy:
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and self.tp_rank == 0
        ):
            logger.warning(
                "DFLASH non-greedy verification is unavailable on this build/device; "
                "falling back to greedy argmax verification."
            )
            self._warned_sampling_fallback = True

    def _make_next_draft_input_prefill(
        self,
        *,
        verified_id: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _make_next_draft_input_decode(
        self,
        *,
        verified_id: torch.Tensor,
        new_seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(new_seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(model_worker_batch)

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            if (
                model_worker_batch.extend_lens is None
                or model_worker_batch.prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_lens / prefix_lens to be populated in extend mode, "
                    "but got None."
                )
            extend_lens = tuple(model_worker_batch.extend_lens)
            prefix_lens = tuple(model_worker_batch.prefix_lens)

            ctx_lens = torch.tensor(
                extend_lens,
                dtype=torch.int32,
                device=model_worker_batch.seq_lens.device,
            )
            draft_seq_lens = torch.tensor(
                prefix_lens,
                dtype=torch.int32,
                device=model_worker_batch.seq_lens.device,
            )
            num_extend_tokens = int(sum(extend_lens))

            # Target prefill: capture DFlash aux hidden states for prompt tokens.
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            logits_output, next_token_ids = (
                batch_output.logits_output,
                batch_output.next_token_ids,
            )
            batch_output.new_seq_lens = model_worker_batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )
            if (
                model_worker_batch.extend_lens is None
                or model_worker_batch.prefix_lens is None
                or tuple(model_worker_batch.extend_lens) != extend_lens
                or tuple(model_worker_batch.prefix_lens) != prefix_lens
            ):
                raise RuntimeError(
                    "DFLASH prefill expected extend_lens / prefix_lens to remain "
                    "stable across target prefill."
                )

            # Materialize prompt tokens into the draft KV cache immediately. This is required
            # for radix cache safety (the scheduler may update radix after prefill returns).
            device = next_token_ids.device
            expected_device = torch.device(self.device)
            if device.type != expected_device.type or (
                expected_device.index is not None
                and device.index != expected_device.index
            ):
                raise RuntimeError(
                    f"DFLASH prefill expected target outputs on {self.device}, got {device}."
                )

            if model_worker_batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH prefill expected out_cache_loc, but got None."
                )
            positions, _ = compute_position(
                self.model_runner.server_args.attention_backend,
                draft_seq_lens,
                ctx_lens,
                num_extend_tokens,
            )
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=logits_output.hidden_states,
                cache_loc=model_worker_batch.out_cache_loc,
                positions=positions,
            )

            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            batch_output.next_draft_input = self._make_next_draft_input_prefill(
                verified_id=next_token_ids,
                seq_lens=model_worker_batch.seq_lens,
                cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            batch_output.next_draft_input.verify_done = verify_done
            return batch_output

        # Decode / target-verify stage.
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DFlashDraftInputV2.create_idle_input(
                device=self.device
            )

        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if model_worker_batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                verified_id=torch.empty((0,), device=self.device, dtype=torch.int32),
                new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=int(self.block_size),
            )

        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(model_worker_batch.seq_lens)
        device = self.device

        # --- 1) Draft a fixed block with the draft model.
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DFLASH requires the target model to expose `lm_head` with `weight`."
            )

        block_size = int(self.block_size)
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = model_worker_batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    verified_id=draft_input.verified_id.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=model_worker_batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.verified_id)
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets,
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=model_worker_batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=device,
                )
                verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.verified_id)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            # Non-windowed path uses the shared overallocated mapping directly.
            # Backend planning only needs a safe upper bound for the committed
            # prefix lengths, not the full allocator reservation length.
            draft_seq_lens = prefix_lens
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif model_worker_batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
                draft_seq_lens_sum = (
                    int(model_worker_batch.seq_lens_sum)
                    if model_worker_batch.seq_lens_sum is not None
                    else int(model_worker_batch.seq_lens_cpu.sum())
                )
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, int(self.block_size) - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        grammar_draft_tail_cpu = None
        grammar_tail_copy_done = None
        if model_worker_batch.has_grammar:
            grammar_draft_tail_cpu = self._ensure_grammar_draft_tail_cpu_buffer(
                bs, block_size
            )
            if block_size > 1:
                copy_stream, copy_stream_ctx = _get_overlap_plan_stream(device)
                if copy_stream is not None:
                    current_stream = torch.get_device_module(device).current_stream()
                    with copy_stream_ctx:
                        copy_stream.wait_stream(current_stream)
                        grammar_draft_tail_cpu.copy_(
                            draft_tokens[:, 1:], non_blocking=True
                        )
                        grammar_tail_copy_done = torch.get_device_module(device).Event()
                        grammar_tail_copy_done.record()
                else:
                    grammar_draft_tail_cpu.copy_(draft_tokens[:, 1:], non_blocking=True)
                    grammar_tail_copy_done = torch.get_device_module(device).Event()
                    grammar_tail_copy_done.record()

        # --- 2) Target verify.
        # TARGET_VERIFY uses standard causal masking; custom masks are unnecessary here.
        custom_mask = None

        verify_input_ids = draft_tokens.reshape(-1)
        verify_input = DFlashVerifyInput(
            draft_token=verify_input_ids,
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        model_worker_batch.out_cache_loc = verify_out_cache_loc
        sampling_info = model_worker_batch.sampling_info

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            model_worker_batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
        seq_lens_sum_backup = model_worker_batch.seq_lens_sum
        if draft_input.planning_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.planning_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.planning_seq_lens_sum)
        elif draft_input.reserved_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_v2_verify(
            model_worker_batch, self.target_worker
        )
        model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
        model_worker_batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph
        dflash_vocab_mask = None
        dflash_mask_grammar = None
        dflash_vocab_mask_ready = None
        dflash_vocab_mask_slot = None
        if model_worker_batch.has_grammar:
            if grammar_tail_copy_done is not None:
                grammar_tail_copy_done.synchronize()
            if grammar_draft_tail_cpu is None:
                raise RuntimeError(
                    "DFLASH grammar decoding expected draft-tail CPU buffer."
                )
            vocab_size = (
                int(sampling_info.vocab_size)
                if sampling_info is not None
                else int(self.target_worker.model_runner.model_config.vocab_size)
            )
            first_grammar = next(
                (req.grammar for req in model_worker_batch.reqs if req.grammar),
                None,
            )
            dflash_vocab_mask_slot = (
                self._ensure_grammar_vocab_mask_slot(
                    grammar=first_grammar,
                    vocab_size=vocab_size,
                    bs=bs,
                    block_size=block_size,
                )
                if first_grammar is not None
                else None
            )
            vocab_mask_buf = (
                dflash_vocab_mask_slot.cpu_buf[: bs * block_size]
                if dflash_vocab_mask_slot is not None
                and dflash_vocab_mask_slot.cpu_buf is not None
                else None
            )
            (
                dflash_vocab_mask,
                dflash_mask_grammar,
            ) = generate_dflash_linear_token_bitmask(
                reqs=model_worker_batch.reqs,
                draft_token_tail_cpu=grammar_draft_tail_cpu,
                vocab_size=vocab_size,
                vocab_mask_buf=vocab_mask_buf,
            )
            if dflash_vocab_mask is not None:
                assert dflash_mask_grammar is not None
                logits_device = logits_output.next_token_logits.device
                if (
                    dflash_vocab_mask_slot is not None
                    and dflash_vocab_mask.device.type == "cpu"
                ):
                    (
                        dflash_vocab_mask,
                        dflash_vocab_mask_ready,
                    ) = self._copy_grammar_vocab_mask_to_device(
                        slot=dflash_vocab_mask_slot,
                        cpu_vocab_mask=dflash_vocab_mask,
                        device=logits_device,
                    )
                else:
                    dflash_vocab_mask = dflash_mask_grammar.move_vocab_mask(
                        dflash_vocab_mask,
                        logits_device,
                    )
            if sampling_info is not None:
                # The regular grammar mask has one row per request. DFLASH verify
                # needs one row per block position, so never broadcast the regular
                # mask over the fixed verify block.
                sampling_info.vocab_mask = None

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(self.block_size),
            )
        if dflash_vocab_mask is not None:
            assert dflash_mask_grammar is not None
            if dflash_vocab_mask_ready is not None:
                torch.get_device_module(
                    logits_output.next_token_logits.device
                ).current_stream().wait_event(dflash_vocab_mask_ready)
            dflash_mask_grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits,
                vocab_mask=dflash_vocab_mask,
            )
            if dflash_vocab_mask_slot is not None:
                apply_done = torch.get_device_module(
                    logits_output.next_token_logits.device
                ).Event()
                apply_done.record()
                dflash_vocab_mask_slot.apply_done = apply_done

        candidates = draft_tokens
        new_seq_lens = None
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            commit_lens = accept_len.to(torch.int32) + 1  # [bs]
            out_tokens = torch.empty(
                (bs, int(self.block_size)), dtype=torch.int64, device=device
            )
            if int(self.block_size) > 1:
                out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
            out_tokens[:, int(self.block_size) - 1].fill_(0)
            out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, int(self.block_size)
            )
            if self._use_triton_accept_bonus:
                try:
                    (
                        accept_len,
                        commit_lens,
                        bonus,
                        out_tokens,
                        new_seq_lens,
                    ) = self._next_accept_bonus_buffers(bs)
                    _compute_dflash_accept_bonus_triton_unchecked(
                        candidates=candidates,
                        target_top1=target_predict,
                        accept_lens_out=accept_len,
                        commit_lens_out=commit_lens,
                        bonus_ids_out=bonus,
                        out_tokens_out=out_tokens,
                        prefix_lens=prefix_lens,
                        new_seq_lens_out=new_seq_lens,
                    )
                except Exception as e:
                    self._use_triton_accept_bonus = False
                    logger.warning(
                        "DFLASH Triton accept/bonus failed; falling back to eager path: %s",
                        e,
                    )
                    accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                    commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                    out_tokens = torch.empty(
                        (bs, int(self.block_size)),
                        dtype=torch.int64,
                        device=device,
                    )
                    if int(self.block_size) > 1:
                        out_tokens[:, : int(self.block_size) - 1].copy_(
                            candidates[:, 1:]
                        )
                    out_tokens[:, int(self.block_size) - 1].fill_(0)
                    out_tokens.scatter_(
                        1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                    )
            else:
                accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                out_tokens = torch.empty(
                    (bs, int(self.block_size)), dtype=torch.int64, device=device
                )
                if int(self.block_size) > 1:
                    out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
                out_tokens[:, int(self.block_size) - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        # --- 3) Materialize committed verify-input tokens into draft KV cache.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, int(self.block_size), -1)

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_out_cache_loc,
            cache_loc_2d=verify_out_cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
        )

        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input_decode(
            verified_id=bonus,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.block_size),
        )
