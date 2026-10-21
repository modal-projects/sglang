"""DFlash raw-tail transfers and independent recurrent-state residency on CUDA.

These exercise HiCache runtime ownership and stream ordering without a server
or model weights; they do not add or benchmark a transfer kernel.
"""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.pool_host.dflash import DFlashFeatureTailHostPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.speculative.dflash_feature_tail import (
    DFlashFeatureCapture,
    DFlashFeatureState,
    DFlashFeatureTail,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class TestDFlashFeatureTailHost(CustomTestCase):
    def pool(self, engine, window, width, capacity=1):
        pool = DFlashFeatureTailHostPool(
            capacity=capacity,
            window_size=window,
            feature_size=width,
            dtype=torch.bfloat16,
            transfer_engine=engine,
        )
        self.addCleanup(pool.destroy)
        return pool

    def test_reused_host_slot_waits_for_previous_reader_without_cpu_wait(self):
        window, width = 64, 128
        engine = L2TransferEngine("direct")
        pool = self.pool(engine, window, width)
        reader, writer = torch.cuda.Stream(), torch.cuda.Stream()
        first = DFlashFeatureTail(
            0, (torch.ones((window, width), dtype=torch.bfloat16, device="cuda"),)
        )
        second = DFlashFeatureTail(0, (torch.full_like(first.parts[0], 2),))

        # Warm every destination/copy allocation so cudaMalloc cannot conceal
        # a missing dependency between the old read and the replacement write.
        warm_handle = pool.backup(first)
        warm_copies = [pool.restore(warm_handle).as_tensor().clone() for _ in range(2)]
        pool.release(warm_handle)
        torch.cuda.synchronize()
        pool.poll()
        del warm_copies

        with (
            patch.object(
                torch.cuda.Event, "synchronize", side_effect=AssertionError("CPU wait")
            ),
            patch.object(
                torch.cuda.Stream, "synchronize", side_effect=AssertionError("CPU wait")
            ),
        ):
            handle = pool.backup(first)
            with torch.cuda.stream(engine.host_to_device_stream):
                torch.cuda._sleep(100_000_000)
            with torch.cuda.stream(reader):
                observed_first = pool.restore(handle).as_tensor().clone()
            pool.release(handle)
            self.assertEqual(pool.available_size(), 1)
            with torch.cuda.stream(writer):
                replacement = pool.backup(second)
                torch.testing.assert_close(replacement._indices, handle._indices)
                observed_second = pool.restore(replacement).as_tensor().clone()
                pool.release(replacement)
        reader.synchronize()
        writer.synchronize()
        torch.testing.assert_close(observed_first, first.as_tensor(), rtol=0, atol=0)
        torch.testing.assert_close(observed_second, second.as_tensor(), rtol=0, atol=0)

    def test_baseline_window_waits_for_producer_backup_and_load_without_cpu_wait(self):
        window, width = 2048, 12288
        engine = L2TransferEngine("direct")
        pool = self.pool(engine, window, width)
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        history = (
            torch.arange(2 * window, device="cuda", dtype=torch.float32)
            .to(torch.bfloat16)[:, None]
            .expand(-1, width)
            .contiguous()
        )
        source = torch.zeros_like(history)
        end = window + window // 2
        expected = history[end - window : end].cpu()

        # Warm copies and both window allocations so cudaMalloc cannot hide an
        # omitted stream dependency in the delayed-producer case below.
        warm_tail = DFlashFeatureTail.capture(
            source, start_pos=0, end_pos=end, window_size=window
        )
        warm_handle = pool.backup(warm_tail)
        warm_restored = pool.restore(warm_handle)
        warm_observed = warm_restored.as_tensor().clone()
        pool.release(warm_handle)
        torch.cuda.synchronize()
        pool.poll()
        del warm_tail, warm_restored, warm_observed

        with (
            patch.object(
                torch.cuda.Event, "synchronize", side_effect=AssertionError("CPU wait")
            ),
            patch.object(
                torch.cuda.Stream, "synchronize", side_effect=AssertionError("CPU wait")
            ),
        ):
            with torch.cuda.stream(producer):
                torch.cuda._sleep(100_000_000)
                source.copy_(history)
                tail = DFlashFeatureTail.capture(
                    source, start_pos=0, end_pos=end, window_size=window
                )
                handle = pool.backup(tail)
                source.fill_(-99)
            source_reference = weakref.ref(tail.parts[0])
            del tail
            with torch.cuda.stream(consumer):
                restored = pool.restore(handle)
                observed = restored.as_tensor().clone()
            pool.release(handle)
        consumer.synchronize()
        torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
        pool.poll()
        self.assertIsNone(source_reference())
        self.assertEqual(pool.available_size(), 1)
        self.assertEqual(restored.as_tensor().untyped_storage().nbytes(), 48 * 1024**2)
        self.assertEqual((restored.start_pos, restored.end_pos), (end - window, end))

    def test_state_only_tail_only_and_both_restore_with_shared_backup_stream(self):
        engine = L2TransferEngine("direct")
        tail_engine = L2TransferEngine("direct")
        tail_engine.device_to_host_stream = engine.device_to_host_stream
        tails = self.pool(tail_engine, window=8, width=32)
        states = MambaStateBufferInfo(
            temporal=torch.zeros((2, 4, 2, 8, 16), device="cuda", dtype=torch.float32),
            conv=(torch.zeros((2, 4, 3, 128), device="cuda", dtype=torch.bfloat16),),
        )
        state_host = MambaPoolHost(
            states,
            host_to_device_ratio=1,
            host_size=0,
            device_capacity=4,
            layout="page_first_direct",
        )
        self.addCleanup(state_host.destroy)
        self.addCleanup(torch.cuda.synchronize)
        state_indices = state_host.alloc(2)[1:]
        source_slot, destination_slot = torch.tensor([1]), torch.tensor([3])
        states.temporal[:, 1].fill_(13)
        states.conv[0][:, 1].fill_(29)
        state_copy = engine.submit_device_to_host(
            [L2Transfer(state_host, states, state_indices, source_slot)]
        )
        history = torch.arange(24 * 32, device="cuda", dtype=torch.bfloat16).view(
            24, 32
        )
        raw = DFlashFeatureTail.capture(history, start_pos=0, end_pos=20, window_size=8)
        handle = tails.backup(raw)
        raw_reference = weakref.ref(raw.parts[0])
        del raw
        state_copy.finish_event.wait(torch.cuda.current_stream())
        states.temporal[:, 1].fill_(-1)
        states.conv[0][:, 1].fill_(-1)
        torch.cuda.synchronize()
        tails.poll()
        self.assertIsNone(raw_reference())

        # The recurrent slot and feature slot deliberately have different IDs.
        for restore_state, restore_tail in ((True, False), (False, True), (True, True)):
            with self.subTest(state=restore_state, tail=restore_tail):
                states.temporal[:, 3].fill_(-1 if restore_state else 13)
                states.conv[0][:, 3].fill_(-1 if restore_state else 29)
                if restore_state:
                    copied = engine.submit_host_to_device(
                        [
                            L2Transfer(
                                state_host, states, state_indices, destination_slot
                            )
                        ],
                        transfer_layer_id_max=2,
                    )
                    copied.finish_event.wait(torch.cuda.current_stream())
                if restore_tail:
                    restored = tails.restore(handle)
                    torch.testing.assert_close(
                        restored.as_tensor(), history[12:20], rtol=0, atol=0
                    )
                    for part in restored.parts:
                        part.zero_()
                torch.testing.assert_close(
                    states.temporal[:, 3], torch.full_like(states.temporal[:, 3], 13)
                )
                torch.testing.assert_close(
                    states.conv[0][:, 3], torch.full_like(states.conv[0][:, 3], 29)
                )
        tails.release(handle)

    def test_active_ring_crosses_result_stream_and_exports_exact_checkpoint(self):
        window, width = 2048, 12288
        engine = L2TransferEngine("direct")
        pool = self.pool(engine, window, width)
        producer, copier, consumer = (torch.cuda.Stream() for _ in range(3))
        req = SimpleNamespace(
            kv=ReqKvInfo(req_pool_idx=0),
            finished=lambda: False,
            is_retracted=False,
        )
        history = (torch.arange(window + 16, device="cuda") % 127).to(torch.bfloat16)
        history = history[:, None].expand(-1, width).contiguous()
        source = torch.empty_like(history)
        producer.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(producer):
            source.copy_(history)
            prefill = DFlashFeatureCapture.prefill(
                reqs=[req],
                target_hidden=source[:window],
                prefix_lens=[0],
                extend_lens=[window],
                checkpoints=None,
                window_size=window,
            )
            source.fill_(-1)

        # The scheduler already waits for result completion before consuming a
        # capture. Use its real D2H holder path, with only metadata crossing.
        def readback(capture):
            with torch.cuda.stream(producer):
                result = GenerationBatchResult(
                    logits_output=LogitsProcessorOutput(None),
                    next_token_ids=torch.tensor([1], device="cuda"),
                    dflash_feature_capture=capture,
                    copy_done=torch.cuda.Event(),
                )
            copier.wait_stream(producer)
            with torch.cuda.stream(copier):
                result.copy_to_cpu(return_logprob=False)
            result.copy_done.synchronize()
            self.assertEqual(capture.metadata.device.type, "cpu")
            self.assertTrue(
                all(x.is_cuda for e in capture.entries for _, x in e.slices)
            )
            return result

        result = readback(prefill)
        with torch.cuda.stream(consumer):
            result.dflash_feature_capture.consume([req])
        state = req.kv.dflash_feature_state
        with torch.cuda.stream(producer):
            source.copy_(history)
            source[window + 9 :].fill_(float("nan"))
            # Raw rows and metadata must be owned before this graph output and
            # the commit buffers are overwritten by another forward.
            starts = torch.tensor([window], device="cuda")
            commits = torch.tensor([9], device="cuda")
            capture = DFlashFeatureCapture.decode(
                reqs=[req],
                target_hidden=source[window:].unsqueeze(0),
                start_positions=starts,
                commit_lens=commits,
                checkpoint_steps=torch.tensor([3], device="cuda"),
                checkpoint_indices=torch.tensor([7], device="cuda"),
                checkpoint_slots=[0],
                window_size=window,
            )
            source.fill_(-2)
            starts.fill_(-1)
            commits.fill_(-1)
        result = readback(capture)
        with (
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("scalar read")
            ),
            patch.object(
                torch.cuda.Event, "synchronize", side_effect=AssertionError("CPU wait")
            ),
            patch.object(
                torch.cuda.Stream, "synchronize", side_effect=AssertionError("CPU wait")
            ),
            torch.cuda.stream(consumer),
        ):
            result.dflash_feature_capture.consume([req])
            snapshot = state.checkpoint(0, window + 4)
            handle = pool.backup(snapshot)
            live = state.snapshot()
            req.kv.mark_kv_released()
            del snapshot
            restored = pool.restore(handle)
            observed = restored.as_tensor().clone()
        consumer.synchronize()
        torch.testing.assert_close(
            live.as_tensor(), history[9 : window + 9], rtol=0, atol=0
        )
        torch.testing.assert_close(observed, history[4 : window + 4], rtol=0, atol=0)
        self.assertEqual(live.as_tensor().untyped_storage().nbytes(), 48 * 1024**2)
        self.assertTrue(state.closed)
        self.assertEqual(state._chunks, [])
        self.assertIsNone(req.kv.dflash_feature_state)
        pool.release(handle)

    def test_pending_prefill_donation_restores_two_independent_active_rings(self):
        window, width = 2048, 12288
        engine = L2TransferEngine("direct")
        pool = self.pool(engine, window, width)
        producer, results, restore_stream = (torch.cuda.Stream() for _ in range(3))
        req = SimpleNamespace(
            kv=ReqKvInfo(req_pool_idx=0),
            finished=lambda: False,
            is_retracted=False,
        )
        history = (torch.arange(window + 16, device="cuda") % 127).to(torch.bfloat16)
        history = history[:, None].expand(-1, width).contiguous()
        source = torch.empty_like(history)
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            source.copy_(history)
            capture = DFlashFeatureCapture.prefill(
                reqs=[req],
                target_hidden=source,
                prefix_lens=[0],
                extend_lens=[window + 16],
                checkpoints=[(0, window)],
                window_size=window,
            )
            source.fill_(-1)
        state = req.kv.dflash_feature_state
        handle = state.donate_checkpoint(0, window, pool)
        self.assertFalse(pool.is_ready(handle))
        # The scheduler has already donated/replaced this physical tracking slot.
        state.invalidate_checkpoint(0)
        with (
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("scalar read")
            ),
            patch.object(
                torch.cuda.Event, "synchronize", side_effect=AssertionError("CPU wait")
            ),
            patch.object(
                torch.cuda.Stream, "synchronize", side_effect=AssertionError("CPU wait")
            ),
        ):
            with torch.cuda.stream(results):
                results.wait_stream(producer)
                capture.consume([req])
                live = state.snapshot()
                req.kv.mark_kv_released()
            self.assertTrue(pool.is_ready(handle))
            with torch.cuda.stream(restore_stream):
                first = DFlashFeatureState.from_tail(pool.restore(handle), window)
                second = DFlashFeatureState.from_tail(pool.restore(handle), window)
                checkpoint_copy = second.snapshot()
            # A restored active ring crosses into result processing and extends
            # independently; eviction must fence both host reads.
            with torch.cuda.stream(results):
                results.wait_stream(restore_stream)
                first._append(history[window : window + 16], window)
                continued = first.snapshot()
                first.close()
            pool.release(handle)
        results.synchronize()
        torch.testing.assert_close(
            checkpoint_copy.as_tensor(), history[:window], rtol=0, atol=0
        )
        torch.testing.assert_close(live.as_tensor(), history[16:], rtol=0, atol=0)
        torch.testing.assert_close(continued.as_tensor(), history[16:], rtol=0, atol=0)
        pool.poll()
        self.assertEqual(pool.available_size(), 1)
        self.assertIsNone(req.kv.dflash_feature_state)


if __name__ == "__main__":
    unittest.main()
