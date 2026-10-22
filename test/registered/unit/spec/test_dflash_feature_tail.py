"""Feature ownership and checkpoint-window boundaries; no server or GPU."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.runtime_context import get_context, get_memory, get_spec
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_feature_tail import (
    CONTEXT_CHUNK_SIZE,
    DFlashContextCapture,
    DFlashContextState,
    DFlashContextWindow,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.dflash_features import capture_dflash_features_reference
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestDFlashFeatureTail(CustomTestCase):
    def test_snapshot_survives_forward_buffer_reuse_without_retaining_full_storage(
        self,
    ):
        backing = torch.arange(128 * 16, dtype=torch.float32).view(128, 16)
        source = backing[:, ::2].requires_grad_()
        expected = source[18:26].detach().clone()
        tail = DFlashContextWindow.capture(
            source, start_pos=0, end_pos=26, window_size=8
        )
        with torch.no_grad():
            backing.fill_(float("nan"))
        torch.testing.assert_close(tail.as_tensor(), expected, rtol=0, atol=0)
        self.assertEqual((tail.start_pos, tail.end_pos), (18, 26))
        self.assertFalse(tail.as_tensor().requires_grad)
        self.assertTrue(tail.as_tensor().is_contiguous())
        self.assertEqual(
            tail.as_tensor().untyped_storage().nbytes(),
            expected.numel() * expected.element_size(),
        )

    def test_chunked_prefill_and_interior_checkpoints_match_full_history(self):
        history = torch.arange(64 * 8, dtype=torch.float32).view(64, 8)
        for window in (4, 8, 16):
            previous = None
            snapshots = []
            for start, stop in zip((0, 5, 9, 40, 45), (5, 9, 40, 45, 64)):
                source = history[start:stop].clone()
                for end in (start, min(start + 2, stop), stop):
                    with self.subTest(window=window, start=start, end=end):
                        tail = DFlashContextWindow.capture(
                            source,
                            start_pos=start,
                            end_pos=end,
                            window_size=window,
                            prefix_tail=previous,
                        )
                        snapshots.append((end, tail))
                previous = tail
                source.fill_(-1)
            for end, tail in snapshots:
                torch.testing.assert_close(
                    tail.as_tensor(),
                    history[max(0, end - window) : end],
                    rtol=0,
                    atol=0,
                )

    def test_only_committed_verify_input_rows_enter_the_tail(self):
        history = torch.arange(48 * 8, dtype=torch.float32).view(48, 8)
        prefix = DFlashContextWindow.capture(
            history[:32], start_pos=0, end_pos=32, window_size=8
        )
        for num_commit_tokens in (1, 5, 16):
            with self.subTest(num_commit_tokens=num_commit_tokens):
                source = history[32:].clone()
                source[num_commit_tokens:].fill_(float("nan"))
                end = 32 + num_commit_tokens
                tail = DFlashContextWindow.capture(
                    source,
                    start_pos=32,
                    end_pos=end,
                    window_size=8,
                    prefix_tail=prefix,
                )
                torch.testing.assert_close(
                    tail.as_tensor(), history[end - 8 : end], rtol=0, atol=0
                )

    def test_missing_or_misaligned_history_cannot_form_a_complete_tail(self):
        history = torch.zeros(16, 8)
        for old_end, old_window in ((7, 8), (9, 8), (8, 2)):
            prefix = DFlashContextWindow.capture(
                history, start_pos=0, end_pos=old_end, window_size=old_window
            )
            with self.subTest(old_end=old_end, old_window=old_window):
                with self.assertRaisesRegex(ValueError, "Missing target feature"):
                    DFlashContextWindow.capture(
                        history[8:9],
                        start_pos=8,
                        end_pos=9,
                        window_size=8,
                        prefix_tail=prefix,
                    )
        with self.assertRaisesRegex(ValueError, "Missing target feature"):
            DFlashContextWindow.capture(
                history[8:9], start_pos=8, end_pos=9, window_size=8
            )
        # A complete new window needs no cached features.
        tail = DFlashContextWindow.capture(
            history[8:], start_pos=8, end_pos=16, window_size=8
        )
        self.assertEqual((tail.start_pos, tail.end_pos), (8, 16))

    def test_invalid_boundaries_and_incompatible_features_fail_before_copy(self):
        source = torch.zeros(4, 8)
        for start, end, window in ((-1, 2, 4), (2, 1, 4), (0, 5, 4), (0, 4, 0)):
            with self.subTest(start=start, end=end, window=window):
                with self.assertRaises(ValueError):
                    DFlashContextWindow.capture(
                        source, start_pos=start, end_pos=end, window_size=window
                    )
        prefix = DFlashContextWindow.capture(
            source, start_pos=0, end_pos=4, window_size=4
        )
        for new in (
            torch.zeros(1, 7),
            torch.zeros(1, 8, dtype=torch.bfloat16),
            torch.empty(1, 8, device="meta"),
        ):
            with self.subTest(shape=new.shape, dtype=new.dtype, device=new.device):
                with self.assertRaisesRegex(ValueError, "must match"):
                    DFlashContextWindow.capture(
                        new, start_pos=4, end_pos=5, window_size=4, prefix_tail=prefix
                    )

    def test_fc_update_reprojects_saved_raw_features(self):
        from sglang.srt.models.dflash import DFlashDraftModel

        model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4),
            num_context_features=3,
            quantizable=False,
            fc=nn.Linear(12, 4, bias=False),
            hidden_norm=nn.RMSNorm(4),
        )
        history = torch.arange(20 * 12, dtype=torch.float32).view(20, 12) / 100
        source = history.clone()
        tail = DFlashContextWindow.capture(
            source, start_pos=0, end_pos=16, window_size=8
        )
        with torch.no_grad():
            model.fc.weight.copy_(torch.arange(48).view(4, 12) / 50)
            old = DFlashDraftModel.project_target_hidden(model, tail.as_tensor())
            source.zero_()
            model.fc.weight[0].mul_(-1)
            rebuilt = DFlashDraftModel.project_target_hidden(model, tail.as_tensor())
            direct = DFlashDraftModel.project_target_hidden(model, history[8:16])
        self.assertFalse(torch.equal(old, rebuilt))
        torch.testing.assert_close(rebuilt, direct, rtol=0, atol=0)


class TestDFlashFeatureCapture(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "sglang.srt.speculative.dflash_feature_tail.capture_dflash_features",
                side_effect=capture_dflash_features_reference,
            )
        )
        override = get_context().override_server_args(
            device="cpu",
            _mamba_cache_chunk_size=4,
            mamba_track_interval=8,
            mamba_radix_cache_strategy="extra_buffer",
        )
        override.install()
        self.addCleanup(override.restore)
        self.history = torch.arange(256 * 6, dtype=torch.float32).view(256, 6)

    def req(self):
        params = SamplingParams(max_new_tokens=256)
        params.normalize(None)
        req = Req("req", "", array("q", [1]), params, vocab_size=128)
        req.kv.req_pool_idx = 0
        return req

    def prefill(self, req, start, end, checkpoint=None, window=8):
        source = self.history[start:end].clone()
        capture = DFlashContextCapture.prefill(
            reqs=[req],
            target_hidden=source,
            prefix_lens=[start],
            extend_lens=[end - start],
            checkpoints=[checkpoint],
            window_size=window,
        )
        source.fill_(float("nan"))
        return capture

    def decode(self, req, start, count, checkpoint=None, window=8):
        source = self.history[start : start + 16].clone()
        source[count:].fill_(float("nan"))
        capture = DFlashContextCapture.decode(
            reqs=[req],
            target_hidden=source.unsqueeze(0),
            start_positions=torch.tensor([start]),
            commit_lens=torch.tensor([count]),
            checkpoint_steps=torch.tensor([checkpoint[1] - start - 1])
            if checkpoint
            else None,
            checkpoint_indices=torch.tensor([3]) if checkpoint else None,
            checkpoint_slots=[checkpoint[0]] if checkpoint else None,
            window_size=window,
        )
        source.fill_(float("nan"))
        return capture

    def assert_tail(self, tail, end, window=8):
        self.assertIsNotNone(tail)
        self.assertEqual((tail.start_pos, tail.end_pos), (max(0, end - window), end))
        torch.testing.assert_close(
            tail.as_tensor(), self.history[max(0, end - window) : end], rtol=0, atol=0
        )

    def test_chunked_prefill_interior_snapshots_and_decode_rollover(self):
        for window in (4, 8, 32):
            with self.subTest(window=window):
                req = self.req()
                snapshots = []
                for start, end, boundary in (
                    (0, 5, 4),
                    (5, 13, 8),
                    (13, 93, 24),
                    (93, 98, 96),
                ):
                    capture = self.prefill(req, start, end, (0, boundary), window)
                    # Staging is bounded even for a long prefill with two disjoint windows.
                    staged = sum(x.shape[0] for _, x in capture.entries[0].slices)
                    self.assertLessEqual(staged, min(end - start, 2 * window))
                    capture.consume([req])
                    state = req.kv.dflash_context_state
                    self.assert_tail(state.snapshot(), end, window)
                    snapshots.append((state.checkpoint(0, boundary), boundary))
                end = 98
                for count in (1, 7, 16, 3, 9, 2, 16, 1):
                    capture = self.decode(req, end, count, (1, end + count), window)
                    end += count
                    capture.consume([req])
                    self.assert_tail(state.snapshot(), end, window)
                    self.assert_tail(state.checkpoint(1, end), end, window)
                for tail, boundary in snapshots:
                    self.assert_tail(tail, boundary, window)

    def test_decode_and_checkpoint_copy_only_new_committed_rows(self):
        class CountCopies(TorchDispatchMode):
            elements = 0
            concatenations = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.copy_.default:
                    self.elements += args[0].numel()
                if func is torch.ops.aten.cat.default:
                    self.concatenations += 1
                return func(*args, **(kwargs or {}))

        req = self.req()
        self.prefill(req, 0, 64, window=64).consume([req])
        state = req.kv.dflash_context_state
        start = 64
        for count in (3, 1, 9, 16, 2, 16, 16, 7):
            capture = self.decode(req, start, count, (0, start + count), window=64)
            with CountCopies() as copies:
                capture.consume([req])
            self.assertEqual(copies.elements, count * self.history.shape[1])
            self.assertEqual(copies.concatenations, 0)
            self.assertLessEqual(
                sum(c.buffer.numel() for c in state._chunks),
                2 * 64 * self.history.shape[1],
            )
            start += count
            self.assert_tail(state.snapshot(), start, 64)

    def test_overlapping_checkpoints_share_rows_without_mutating_old_or_branched_tails(
        self,
    ):
        self.history = torch.arange(8192 * 6, dtype=torch.float32).view(8192, 6)
        req = self.req()
        window = 2048
        self.prefill(req, 0, 4096, (0, 4096), window).consume([req])
        state = req.kv.dflash_context_state
        previous = state.checkpoint(0, 4096)
        for start in range(4096, 4672, 16):
            checkpoint = (1, 4608) if start + 16 == 4608 else None
            self.decode(req, start, 16, checkpoint, window).consume([req])
        latest, live = state.checkpoint(1, 4608), state.snapshot()
        storages = {
            part.untyped_storage().data_ptr(): part.untyped_storage().nbytes()
            for tail in (previous, latest, live)
            for part in tail.parts
        }
        # The union spans 2624 rows; only its last 256-row chunk is partial.
        self.assertEqual(sum(storages.values()), 2816 * 6 * 4)
        self.assertLess(sum(storages.values()), 3 * window * 6 * 4)
        for tail, end in ((previous, 4096), (latest, 4608), (live, 4672)):
            self.assert_tail(tail, end, window)

        branch = DFlashContextState.from_tail(latest, window)
        branch._append(torch.full((16, 6), -7.0), 4608)
        for start in range(4672, 7168, 16):
            self.decode(req, start, 16, window=window).consume([req])
        state.close()
        for tail, end in ((previous, 4096), (latest, 4608), (live, 4672)):
            self.assert_tail(tail, end, window)
        expected = torch.cat((self.history[2576:4608], torch.full((16, 6), -7.0)))
        torch.testing.assert_close(
            branch.snapshot().as_tensor(), expected, rtol=0, atol=0
        )

    def test_cache_hit_without_history_only_becomes_complete_after_a_new_window(self):
        req = self.req()
        self.prefill(req, 64, 67, (0, 66)).consume([req])
        state = req.kv.dflash_context_state
        self.assertIsNone(state.snapshot())
        self.assertIsNone(state.checkpoint(0, 66))
        self.decode(req, 67, 5, (1, 72)).consume([req])
        self.assert_tail(state.snapshot(), 72)
        self.assert_tail(state.checkpoint(1, 72), 72)
        self.assertIsNone(state.checkpoint(1, 71))

    def test_disjoint_and_branched_windows_obey_chunk_storage_bound(self):
        self.history = torch.arange(16384 * 6, dtype=torch.float32).view(16384, 6)
        for window in (1, 17, 255, 256, 257, 2048):
            with self.subTest(window=window):
                req = self.req()
                first = window + 17
                second = first + window + 1
                end = first + 3 * window + 5
                self.prefill(req, 0, first, (0, first), window).consume([req])
                self.prefill(req, first, end, (1, second), window).consume([req])
                state = req.kv.dflash_context_state
                bound = window + 2 * (min(window, CONTEXT_CHUNK_SIZE) - 1)

                def assert_storage(tail, position):
                    self.assert_tail(tail, position, window)
                    storages = {
                        part.untyped_storage().data_ptr(): part.untyped_storage().nbytes()
                        for part in tail.parts
                    }
                    self.assertLessEqual(sum(storages.values()), bound * 6 * 4)

                # All three windows are disjoint: the reserve cannot assume
                # that checkpoint spacing is shorter than the attention window.
                previous = state.checkpoint(0, first)
                latest = state.checkpoint(1, second)
                assert_storage(previous, first)
                assert_storage(latest, second)
                assert_storage(state.snapshot(), end)

                # Repeated forks at partial chunk boundaries must not accumulate
                # internal padding or allow either branch to overwrite old rows.
                for count in (1, 3, 15, 7, 16, 2, 9, 16):
                    tail = state.snapshot()
                    state = DFlashContextState.from_tail(tail, window)
                    state._append(self.history[end : end + count], end)
                    assert_storage(tail, end)
                    end += count
                    assert_storage(state.snapshot(), end)
                assert_storage(previous, first)
                assert_storage(latest, second)

    def test_finished_checkpoint_uses_its_own_tail_after_long_decode(self):
        from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent

        req = self.req()
        self.prefill(req, 0, 8, (0, 8)).consume([req])
        self.decode(req, 8, 16, (1, 16)).consume([req])
        for start in range(24, 120, 16):
            self.decode(req, start, 16).consume([req])
        state = req.kv.dflash_context_state
        self.assert_tail(state.snapshot(), 120)
        pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        pool.enable_mamba_extra_buffer_lazy = False
        pool.mamba_ping_pong_track_buffer_size = 2
        component = MambaComponent.__new__(MambaComponent)
        component.cache = SimpleNamespace(req_to_token_pool=pool)
        req.kv.mamba_last_track_idx = 1
        req.kv.mamba_last_track_seqlen = 16
        req.kv.mamba_prev_track_seqlen = 8
        # A finished key can name the preceding checkpoint, not the latest one.
        boundary, slot = component._select_finished_checkpoint(req, 15)
        self.assertEqual((boundary, slot), (8, 0))
        self.assert_tail(state.checkpoint(slot, boundary), 8)
        self.assertIsNone(component._select_finished_checkpoint(req, 7))

    def test_slot_replacement_rejects_pending_snapshot_but_keeps_active_rows(self):
        req = self.req()
        self.prefill(req, 0, 8, (0, 8)).consume([req])
        state = req.kv.dflash_context_state
        pending = self.decode(req, 8, 8, (0, 16))
        pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            (1, 2), dtype=torch.long
        )
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([3, 4])
        pool.set_mamba_ping_pong_slot(req, 0, 5)
        self.assertIsNone(state.checkpoint(0, 8))
        pending.consume([req])
        self.assertIsNone(state.checkpoint(0, 16))
        self.assert_tail(state.snapshot(), 16)
        self.decode(req, 16, 4, (0, 20)).consume([req])
        self.assert_tail(state.checkpoint(0, 20), 20)

    def test_overlapping_prefills_reusing_tracking_slot_keep_latest_snapshot(self):
        req = self.req()
        first = self.prefill(req, 0, 8, (0, 8))
        second = self.prefill(req, 8, 16, (0, 16))
        first.consume([req])
        state = req.kv.dflash_context_state
        self.assertIsNone(state.checkpoint(0, 8))
        self.assert_tail(state.snapshot(), 8)
        second.consume([req])
        self.assert_tail(state.checkpoint(0, 16), 16)
        self.assert_tail(state.snapshot(), 16)

    def test_release_and_recycled_request_row_cannot_resurrect_pending_features(self):
        req = self.req()
        self.prefill(req, 0, 8, (0, 8)).consume([req])
        old = req.kv.dflash_context_state
        pending = self.decode(req, 8, 8, (1, 16))
        req.kv.mark_kv_released()
        self.assertTrue(old.closed)
        self.assertEqual(old._chunks, [])
        self.assertIsNone(old.checkpoint(0, 8))
        req.kv = ReqKvInfo(req_pool_idx=0)
        self.prefill(req, 0, 4, (0, 4)).consume([req])
        pending.consume([req])
        self.assert_tail(req.kv.dflash_context_state.snapshot(), 4)

    def test_retracted_or_finished_requests_discard_inflight_captures(self):
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        for finished in (False, True):
            with self.subTest(finished=finished):
                req = self.req()
                self.prefill(req, 0, 8).consume([req])
                pending = self.decode(req, 8, 4)
                if finished:
                    req.finished_reason = FINISH_ABORT("cancelled")
                else:
                    req.is_retracted = True
                pending.consume([req])
                self.assert_tail(req.kv.dflash_context_state.snapshot(), 8)
                req.kv.mark_kv_released()
                self.assertIsNone(req.kv.dflash_context_state)

    def test_batched_rows_and_negative_tracking_slot_do_not_cross_requests(self):
        reqs = [self.req(), self.req()]
        hidden = torch.stack((self.history[:16], self.history[32:48]))
        DFlashContextCapture.decode(
            reqs=reqs,
            target_hidden=hidden,
            start_positions=torch.tensor([0, 32]),
            commit_lens=torch.tensor([5, 13]),
            checkpoint_steps=torch.tensor([3, 7]),
            checkpoint_indices=torch.tensor([3, -1]),
            checkpoint_slots=[0, 1],
            window_size=8,
        ).consume(reqs)
        self.assert_tail(reqs[0].kv.dflash_context_state.snapshot(), 5)
        self.assert_tail(reqs[0].kv.dflash_context_state.checkpoint(0, 4), 4)
        self.assert_tail(reqs[1].kv.dflash_context_state.snapshot(), 45)
        self.assertIsNone(reqs[1].kv.dflash_context_state.checkpoint(1, 40))

    def test_prefill_uses_actual_checkpoint_boundary_before_kernel_encoding(self):
        req = self.req()
        req.extend_end = 40
        req.kv.mamba_ping_pong_track_buffer = torch.tensor([3, 4])
        req.kv.mamba_next_track_idx = 0
        req.mamba_branching_seqlen = 12
        pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        pool.mamba_ping_pong_track_buffer_size = 2
        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=pool,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(mamba_chunk_size=4)
            ),
            tree_cache=SimpleNamespace(page_size=4),
        )
        plan = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        self.assertEqual(
            plan.track_seqlen, 13
        )  # Kernel encoding forces an interior h snapshot.
        self.assertEqual(plan.checkpoint, (0, 12))
        capture = self.prefill(req, 0, 40, plan.checkpoint)
        # Scheduling lookahead can advance mutable request bookkeeping.
        req.kv.mamba_last_track_seqlen = 48
        req.kv.mamba_last_track_idx = 1
        capture.consume([req])
        self.assert_tail(req.kv.dflash_context_state.checkpoint(0, 12), 12)
        self.assert_tail(req.kv.dflash_context_state.snapshot(), 40)

    def test_decode_checkpoint_uses_the_recurrent_commit_selection(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        # Stand in for device state copies with tensors recording which verify
        # row was selected. The worker's crossing and partial-commit math is real.
        selected = {}

        def commit(**kwargs):
            selected.update(kwargs)

        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._need_mamba_verify_commit = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(
                    update_mamba_state_after_mtp_verify=commit
                ),
                model=None,
            )
        )
        reqs = [self.req() for _ in range(3)]
        for req in reqs:
            self.prefill(req, 0, 7).consume([req])
        batch = ScheduleBatch(reqs=reqs, tree_cache=SimpleNamespace(page_size=4))
        batch.req_pool_indices = torch.arange(3)
        batch.mamba_track_indices = torch.tensor([3, 4, 5])
        starts, counts = torch.tensor([7, 7, 7]), torch.tensor([1, 6, 10])
        steps = worker._update_target_mamba_state_after_verify(
            batch=batch,
            seq_lens_pre_verify=starts,
            seq_lens_post_verify=starts + counts,
            commit_lens=counts,
        )
        torch.testing.assert_close(selected["last_correct_step_indices"], counts - 1)
        torch.testing.assert_close(steps, torch.tensor([0, 0, 8]))
        DFlashContextCapture.decode(
            reqs=reqs,
            target_hidden=self.history[7:23].expand(3, -1, -1),
            start_positions=starts,
            commit_lens=counts,
            checkpoint_steps=steps,
            checkpoint_indices=batch.mamba_track_indices,
            checkpoint_slots=[0, 0, 0],
            window_size=8,
        ).consume(reqs)
        for req, end, boundary in zip(reqs, (8, 13, 17), (8, 8, 16)):
            self.assert_tail(req.kv.dflash_context_state.snapshot(), end)
            self.assert_tail(
                req.kv.dflash_context_state.checkpoint(0, boundary), boundary
            )

    def test_worker_default_and_hicache_alone_do_not_enable_raw_capture(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._context_window = None
        worker.use_compact_draft_cache = False
        worker._draft_worker = SimpleNamespace(alloc_memory_pool=lambda **kwargs: None)
        for hicache in (False, True):
            with get_memory().override(enable_hierarchical_cache=hicache):
                worker.alloc_memory_pool()
                self.assertIsNone(worker._context_window)
        with get_spec().override(enable_dflash_swa_cache=True):
            with self.assertRaisesRegex(
                ValueError, "requires a separate draft SWA pool"
            ):
                worker.alloc_memory_pool()
        self.assertIsNone(worker._context_window)

    def test_worker_prefill_retains_raw_features_before_clearing_target_output(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        for window in (None, 8):
            with self.subTest(window=window):
                req = self.req()
                batch = ScheduleBatch(
                    reqs=[req],
                    forward_mode=ForwardMode.EXTEND,
                    sampling_info=SimpleNamespace(is_all_greedy=True),
                    prefix_lens=[0],
                    extend_lens=[40],
                    seq_lens=torch.tensor([40]),
                    out_cache_loc=torch.arange(40),
                    mamba_prefill_checkpoints=[(0, 12)],
                    tree_cache=SimpleNamespace(
                        dflash_checkpoint_pool=SimpleNamespace(
                            revision=0, window_size=window
                        )
                        if window
                        else None
                    ),
                )
                source = self.history[:40].clone()
                output = GenerationBatchResult(
                    logits_output=LogitsProcessorOutput(None, hidden_states=source),
                    next_token_ids=torch.tensor([42]),
                    kv_loc_plan=SimpleNamespace(
                        write_ids=lambda _: batch.out_cache_loc
                    ),
                )
                worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
                worker._context_window = window
                worker._draft_cache = (
                    SimpleNamespace(
                        prepare=lambda *a, **kw: None,
                        retain_prefill=lambda *a: None,
                    )
                    if window
                    else None
                )
                worker._project_context_rows = lambda hidden, *a, **kw: hidden.clone()
                # Inference, collectives, and KV projection are external to the
                # capture contract. Run the actual prefill orchestration around them.
                worker._target_worker = SimpleNamespace(
                    forward_batch_generation=lambda *a, **kw: output
                )
                worker._tp_sync = SimpleNamespace(sync=lambda *a: None)
                worker.model_runner = SimpleNamespace(
                    prefill_attention_backend_str="torch_native"
                )
                worker.draft_model_runner = SimpleNamespace(kv_index_translator=None)
                worker._append_target_hidden_to_draft_kv_by_loc = lambda **kw: None
                result = worker.forward_batch_generation(batch)
                source.fill_(float("nan"))
                self.assertIsNone(result.logits_output.hidden_states)
                if window is None:
                    self.assertIsNone(result.dflash_context_capture)
                    self.assertIsNone(req.kv.dflash_context_state)
                else:
                    result.dflash_context_capture.consume([req])
                    self.assert_tail(req.kv.dflash_context_state.snapshot(), 40)
                    self.assert_tail(req.kv.dflash_context_state.checkpoint(0, 12), 12)


if __name__ == "__main__":
    unittest.main()
