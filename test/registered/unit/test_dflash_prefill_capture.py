"""Real CUDA copies and independent-process checks of the prefill capture ABI."""

import argparse
import fcntl
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.prefill_capture import (
    COPYING,
    FREE,
    READING,
    READY,
    SLOT_HEADER,
    PrefillCapture,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class _Request:
    def __init__(self, rid, length):
        self.rid = rid
        self.origin_input_ids = list(range(length))
        self.output_ids = []
        self.is_retracted = False
        self.multimodal_inputs = None
        self.done = False

    def finished(self):
        return self.done


class TestCaptureArgs(unittest.TestCase):
    def test_cli_and_validation(self):
        from sglang.srt.arg_groups.speculative_hook import _validate_speculative_capture
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                "target",
                "--speculative-capture-path",
                "/dev/shm/tail",
                "--speculative-capture-window",
                "256",
                "--speculative-capture-slots",
                "4",
                "--speculative-capture-verify-slots",
                "64",
                "--speculative-capture-sample-rate",
                "0.5",
            ]
        )
        self.assertEqual(args.speculative_capture_window, 256)
        self.assertEqual(args.speculative_capture_slots, 4)
        self.assertEqual(args.speculative_capture_mode, "kl")
        self.assertEqual(args.speculative_capture_verify_slots, 64)
        self.assertEqual(args.speculative_capture_sample_rate, 0.5)
        cfg = SimpleNamespace(**vars(args))
        cfg.speculative_algorithm = "DFLASH"
        cfg.device = "cuda"
        for tp_size in (1, 2, 4, 8):
            cfg.tp_size = tp_size
            _validate_speculative_capture(cfg)
        for key, value in [
            ("pp_size", 2),
            ("dp_size", 2),
            ("device", "cpu"),
            ("speculative_algorithm", "EAGLE"),
            ("speculative_capture_slots", 0),
            ("speculative_capture_mode", "unknown"),
            ("speculative_capture_verify_slots", 0),
            ("speculative_capture_window", -1),
            ("speculative_capture_sample_rate", 1.1),
            ("speculative_capture_sample_rate", float("nan")),
        ]:
            with self.subTest(key=key, value=value):
                invalid = SimpleNamespace(**vars(cfg))
                setattr(invalid, key, value)
                with self.assertRaises(ValueError):
                    _validate_speculative_capture(invalid)
        # Disabled capture should add no restrictions to other engines/platforms.
        _validate_speculative_capture(SimpleNamespace(speculative_capture_path=None))

    def test_disabled_capture_does_not_require_rank_or_model_state(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        worker = SimpleNamespace()
        with mock.patch(
            "sglang.srt.speculative.dflash_worker_v2.get_spec",
            return_value=SimpleNamespace(speculative_capture_path=None),
        ):
            DFlashWorkerV2._init_prefill_capture(worker)
        self.assertIsNone(worker._prefill_capture)

    def test_weight_update_invalidates_only_target_capture_namespace(self):
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        capture = mock.Mock()
        updater = SimpleNamespace(
            scheduler=SimpleNamespace(
                model_worker=SimpleNamespace(_prefill_capture=capture)
            ),
            metrics_collector=None,
        )
        with SchedulerWeightUpdaterManager._observe_weight_load(
            updater, "tensor", changes_target=False
        ):
            capture.invalidate_target.assert_not_called()
        with self.assertRaises(RuntimeError):
            with SchedulerWeightUpdaterManager._observe_weight_load(updater, "disk"):
                capture.invalidate_target.assert_called_once()
                raise RuntimeError("partially failed weight load")


class TestTeacherState(unittest.TestCase):
    def test_teacher_is_normalized_unpruned_input_not_aux_or_pre_norm(self):
        from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.capture_target_hidden_states = True
        head = torch.tensor([[1.0, -1.0], [2.0, 3.0], [-4.0, 2.0]])
        processor._get_logits = lambda states, _head, _meta: states @ head.T
        normalized = torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
        aux = normalized.repeat(1, 3) + 10
        before_norm = normalized * 7
        for mode in (ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY):
            with (
                self.subTest(mode=mode),
                mock.patch(
                    "sglang.srt.layers.logits_processor.layernorm_sp.maybe_exit_gather",
                    side_effect=lambda **kw: (
                        kw["hidden_states"],
                        kw["hidden_states_before_norm"],
                    ),
                ),
            ):
                metadata = LogitsMetadata(
                    forward_mode=mode,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    extend_seq_lens=torch.tensor([1, 2]),
                )
                result = processor(
                    torch.tensor([1, 2, 3]),
                    normalized,
                    None,
                    metadata,
                    aux_hidden_states=aux,
                    hidden_states_before_norm=before_norm,
                )
                self.assertIs(result.target_hidden_states, normalized)
                # The existing generic hidden capture can prefer pre-norm
                # states; KL must remain independent of that choice.
                torch.testing.assert_close(result.hidden_states, before_norm)
                selected = (
                    normalized[[0, 2]] if mode == ForwardMode.EXTEND else normalized
                )
                torch.testing.assert_close(result.next_token_logits, selected @ head.T)
                processor.capture_target_hidden_states = False
                disabled = processor(
                    torch.tensor([1, 2, 3]),
                    normalized,
                    None,
                    metadata,
                    aux_hidden_states=aux,
                )
                self.assertIsNone(disabled.target_hidden_states)
                torch.testing.assert_close(disabled.hidden_states, aux)
                torch.testing.assert_close(
                    disabled.next_token_logits, result.next_token_logits
                )
                processor.capture_target_hidden_states = True


@unittest.skipUnless(
    torch.cuda.is_available() and not torch.version.hip, "requires CUDA"
)
class TestPrefillCapture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(self.directory.cleanup)
        self.path = self.directory.name + "/capture"

    def make_capture(self, **kwargs):
        options = dict(
            path=self.path,
            slots=2,
            window=4,
            hidden_size=6,
            dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
            metadata={"target_layer_ids": [1, 3]},
        )
        options.update(kwargs)
        capture = PrefillCapture(**options)
        self.addCleanup(capture.close)
        return capture

    def teacher_snapshot(self, capture, slot, count):
        offset = capture._slot_offset(slot) + (
            capture._teacher_offset if slot < capture.slots else capture._decode_teacher
        )
        size = count * capture.teacher_hidden_size * 2
        return torch.frombuffer(
            bytearray(capture._mapping[offset : offset + size]), dtype=capture.dtype
        ).view(count, capture.teacher_hidden_size)

    def test_kl_chunked_prefill_and_cached_suffix_survive_source_reuse(self):
        capture = self.make_capture(teacher_hidden_size=2)
        first, cached = _Request("chunk", 10), _Request("cached", 7)
        aux = torch.zeros((8, 6), device="cuda", dtype=torch.bfloat16)
        teacher = torch.arange(16, device="cuda", dtype=torch.bfloat16).view(8, 2)
        capture.offer([first], [0], [8], aux, teacher_hidden_states=teacher)
        teacher.fill_(-1)
        teacher = torch.arange(16, 24, device="cuda", dtype=torch.bfloat16).view(4, 2)
        capture.offer(
            [first, cached], [8, 5], [2, 2], aux[:4], teacher_hidden_states=teacher
        )
        teacher.fill_(-1)
        _, info, _, _ = self.snapshot(capture, slot=0)
        self.assertEqual(
            info["target_epoch"], capture.descriptor["initial_target_epoch"]
        )
        torch.testing.assert_close(
            self.teacher_snapshot(capture, 0, 4),
            torch.arange(12, 20, dtype=torch.bfloat16).view(4, 2),
        )
        _, info, _, _ = self.snapshot(capture, slot=1)
        self.assertFalse(info["tail_complete"])
        torch.testing.assert_close(
            self.teacher_snapshot(capture, 1, 2),
            torch.arange(20, 24, dtype=torch.bfloat16).view(2, 2),
        )
        self.assertEqual(capture.descriptor["teacher_prediction_offset"], 1)

    def test_kl_verify_uses_committed_teacher_rows_and_stop_trim(self):
        capture = self.make_capture(teacher_hidden_size=2)
        reqs = [_Request("reject", 5), _Request("stop", 5)]
        for req in reqs:
            req.output_ids = [10]
        teacher = torch.arange(16, device="cuda", dtype=torch.bfloat16).view(8, 2)
        receipt = capture.offer_verify(
            reqs,
            torch.zeros((2, 4, 6), device="cuda", dtype=torch.bfloat16),
            torch.tensor([[10, 11, 12, 99], [10, 21, 22, 23]], device="cuda"),
            torch.tensor([5, 6, 7, 8] * 2, device="cuda"),
            torch.tensor([3, 4], device="cuda", dtype=torch.int32),
            teacher_hidden_states=teacher,
        )
        teacher.fill_(-1)
        reqs[0].output_ids.extend([11, 12, 77])
        reqs[1].output_ids.extend([21, 22, 23, 77])
        reqs[1].done = True
        receipt.consume(
            SimpleNamespace(reqs=reqs),
            [
                SimpleNamespace(output_index=1, token_ids=(11, 12, 77)),
                SimpleNamespace(output_index=1, token_ids=(21,)),
            ],
        )
        for index, count in enumerate((3, 2)):
            slot = capture.slots + index
            _, info, _, _ = self.snapshot(capture, slot=slot)
            self.assertEqual(info["token_count"], count)
            torch.testing.assert_close(
                self.teacher_snapshot(capture, slot, count),
                torch.arange(
                    index * 8, index * 8 + count * 2, dtype=torch.bfloat16
                ).view(count, 2),
            )

    def test_kl_missing_or_invalid_teacher_drops_whole_record(self):
        capture = self.make_capture(teacher_hidden_size=2)
        req = _Request("missing", 4)
        aux = torch.zeros((4, 6), device="cuda", dtype=torch.bfloat16)
        for teacher in (
            None,
            torch.zeros((4, 3), device="cuda", dtype=torch.bfloat16),
            torch.zeros((4, 2)),
        ):
            capture.offer([req], [0], [4], aux, teacher_hidden_states=teacher)
        self.assertEqual(capture.stats["dropped_layout"], 3)
        self.assertEqual(capture.stats["admitted"], 0)

    def snapshot(self, capture, slot=0, state=READY, timeout=5):
        deadline = time.monotonic() + timeout
        fd = os.open(self.path, os.O_RDWR)
        try:
            while time.monotonic() < deadline:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    offset = capture._slot_offset(slot)
                    actual_state, length, generation = SLOT_HEADER.unpack_from(
                        capture._mapping, offset
                    )
                    if actual_state == state:
                        if state == COPYING:
                            return generation
                        info = json.loads(
                            capture._mapping[
                                offset + SLOT_HEADER.size : offset
                                + SLOT_HEADER.size
                                + length
                            ]
                        )
                        count = info["token_count"]
                        tokens = struct.unpack_from(
                            f"<{count}q",
                            capture._mapping,
                            offset + capture.descriptor["token_offset"],
                        )
                        start = offset + capture._slot_aux_offset(slot)
                        data = bytearray(
                            capture._mapping[
                                start : start + count * capture.hidden_size * 2
                            ]
                        )
                        aux = (
                            torch.frombuffer(data, dtype=capture.dtype)
                            .reshape(count, capture.hidden_size)
                            .clone()
                        )
                        return generation, info, tokens, aux
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                time.sleep(0.005)
        finally:
            os.close(fd)
        self.fail(f"Slot {slot} did not reach state {state}: {capture.stats}")

    def set_state(self, capture, state, slot=0):
        fd = os.open(self.path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            offset = capture._slot_offset(slot)
            _, length, generation = SLOT_HEADER.unpack_from(capture._mapping, offset)
            SLOT_HEADER.pack_into(capture._mapping, offset, state, length, generation)
        finally:
            os.close(fd)

    def test_tail_is_pinned_and_survives_immediate_source_overwrite(self):
        capture = self.make_capture()
        self.assertTrue(capture._host.is_pinned())
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        req = _Request("first", 10)
        source = torch.arange(60, device="cuda", dtype=torch.bfloat16).reshape(10, 6)
        allocated = torch.cuda.memory_allocated()
        with mock.patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("host sync")
        ):
            capture.offer([req], [0], [10], source)
        source.fill_(-1)
        _, info, tokens, aux = self.snapshot(capture)
        self.assertTrue(info["tail_complete"])
        self.assertEqual((info["start"], info["end"]), (6, 10))
        self.assertEqual(tokens, (6, 7, 8, 9))
        torch.testing.assert_close(
            aux, torch.arange(36, 60, dtype=torch.bfloat16).reshape(4, 6)
        )
        self.assertLessEqual(torch.cuda.memory_allocated(), allocated)

    def test_only_tp_zero_owns_the_arena_and_exports_full_feature_rows(self):
        self._check_tp_owner("kl")

    def test_ce_mode_omits_teacher_storage_and_graph_outputs(self):
        self._check_tp_owner("ce")

    def _check_tp_owner(self, mode):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        spec = SimpleNamespace(
            speculative_capture_mode=mode,
            speculative_capture_path=self.path,
            speculative_capture_slots=2,
            speculative_capture_verify_slots=64,
            speculative_capture_window=None,
            speculative_capture_sample_rate=1.0,
            speculative_draft_model_path="draft",
            speculative_draft_model_revision=None,
        )
        primary = SimpleNamespace(
            server_args=SimpleNamespace(model_path="target", revision=None),
            model_runner=SimpleNamespace(
                tp_rank=0,
                tp_size=2,
                dtype=torch.bfloat16,
                spec_aux_config=SimpleNamespace(dflash_target_layer_ids=[1, 3]),
                model=SimpleNamespace(
                    logits_processor=SimpleNamespace(
                        config=SimpleNamespace(hidden_size=3),
                        vocab_size=10,
                        logit_scale=None,
                        final_logit_softcapping=None,
                        use_fp32_lm_head=False,
                        rl_on_policy_target=None,
                    ),
                    lm_head=SimpleNamespace(
                        weight=torch.empty(10, 3, dtype=torch.bfloat16)
                    ),
                ),
            ),
            draft_model=SimpleNamespace(
                config=SimpleNamespace(layer_types=["sliding_attention"]),
                get_attention_sliding_window_size=lambda: 3,
                fc=SimpleNamespace(weight=SimpleNamespace(shape=(3, 6))),
            ),
            device="cuda:0",
            draft_window_size=None,
            page_size=1,
            block_size=16,
            _mask_token_id=99,
        )
        # A non-owner must not inspect model tensors, register memory, or try to
        # create/unlink the same path, whether it initializes first or last.
        peer = SimpleNamespace(model_runner=SimpleNamespace(tp_rank=1))
        with mock.patch(
            "sglang.srt.speculative.dflash_worker_v2.get_spec", return_value=spec
        ):
            DFlashWorkerV2._init_prefill_capture(peer)
            self.assertIsNone(peer._prefill_capture)
            self.assertFalse(os.path.exists(self.path))
            DFlashWorkerV2._init_prefill_capture(primary)
            capture = primary._prefill_capture
            self.addCleanup(capture.close)
            inode = os.stat(self.path).st_ino
            DFlashWorkerV2._init_prefill_capture(peer)
            DFlashWorkerV2.release_host_resources(peer)
        self.assertIsNone(peer._prefill_capture)
        self.assertEqual(os.stat(self.path).st_ino, inode)
        self.assertEqual(capture.descriptor["metadata"]["tp_size"], 2)
        self.assertEqual(capture.descriptor["metadata"]["tp_rank"], 0)
        self.assertEqual(capture.hidden_size, 6)
        self.assertEqual(capture.teacher_hidden_size, 3 if mode == "kl" else 0)
        self.assertEqual(
            primary.model_runner.model.logits_processor.capture_target_hidden_states,
            mode == "kl",
        )
        self.assertEqual(
            struct.unpack_from("<I", capture._mapping, 8)[0], 3 if mode == "kl" else 2
        )
        req = _Request("tp", 10)
        source = torch.arange(60, device="cuda", dtype=torch.bfloat16).reshape(10, 6)
        capture.offer(
            [req], [0], [10], source, teacher_hidden_states=source[:, :3].contiguous()
        )
        _, info, tokens, aux = self.snapshot(capture)
        self.assertTrue(info["tail_complete"])
        self.assertEqual(tokens, (6, 7, 8, 9))
        torch.testing.assert_close(
            aux, torch.arange(36, 60, dtype=torch.bfloat16).reshape(4, 6)
        )

    def test_batched_chunked_prefill_and_cached_prefix(self):
        capture = self.make_capture()
        first, second = _Request("chunked", 10), _Request("cached", 7)
        capture.offer(
            [first],
            [0],
            [8],
            torch.arange(48, device="cuda", dtype=torch.bfloat16).reshape(8, 6),
        )
        self.snapshot(capture, state=COPYING)
        # Packed rows: two final rows for the first request, then two for the
        # cache hit. The second request's expected [3,7) tail is missing [3,5).
        source = (
            torch.arange(24, device="cuda", dtype=torch.bfloat16).reshape(4, 6) + 48
        )
        capture.offer([first, second], [8, 5], [2, 2], source)
        _, info, tokens, aux = self.snapshot(capture, slot=0)
        self.assertTrue(info["tail_complete"])
        self.assertEqual(tokens, (6, 7, 8, 9))
        torch.testing.assert_close(
            aux, torch.arange(36, 60, dtype=torch.bfloat16).reshape(4, 6)
        )
        _, info, tokens, aux = self.snapshot(capture, slot=1)
        self.assertFalse(info["tail_complete"])
        self.assertEqual(
            (info["expected_start"], info["start"], info["end"]), (3, 5, 7)
        )
        self.assertEqual(tokens, (5, 6))
        torch.testing.assert_close(
            aux, torch.arange(60, 72, dtype=torch.bfloat16).reshape(2, 6)
        )

    def test_capacity_reader_ownership_and_generations(self):
        capture = self.make_capture(slots=1)
        a, b = _Request("a", 4), _Request("b", 4)
        source = torch.ones((4, 6), device="cuda", dtype=torch.bfloat16)
        capture.offer([a], [0], [4], source)
        generation, _, _, before = self.snapshot(capture)
        size = os.stat(self.path).st_size
        self.set_state(capture, READING)
        for _ in range(10):
            capture.offer([b], [0], [4], source * 2)
        self.assertEqual(capture.stats["dropped_full"], 10)
        self.assertEqual(os.stat(self.path).st_size, size)
        _, info, _, aux = self.snapshot(capture, state=READING)
        self.assertEqual(info["rid"], "a")
        torch.testing.assert_close(aux, before)
        self.set_state(capture, FREE)
        # The publisher reclaims shared FREE slots into the bounded local pool.
        with capture._lock:
            capture._refill_available()
        capture.offer([b], [0], [4], source * 2)
        next_generation, info, _, aux = self.snapshot(capture)
        self.assertEqual(next_generation, generation + 1)
        self.assertEqual(info["rid"], "b")
        torch.testing.assert_close(aux, before * 2)

    def test_completion_gates_publication(self):
        capture = self.make_capture()
        req = _Request("gated", 4)
        source = torch.ones((4, 6), device="cuda", dtype=torch.bfloat16)
        real_event = torch.cuda.Event()
        gate = SimpleNamespace(
            record=real_event.record,
            query=lambda: False,
            synchronize=real_event.synchronize,
        )
        with mock.patch(
            "sglang.srt.speculative.prefill_capture.torch.cuda.Event", return_value=gate
        ):
            capture.offer([req], [0], [4], source)
        real_event.synchronize()
        with capture._lock:
            capture._publish_completed()
        self.snapshot(capture, state=COPYING)
        gate.query = real_event.query
        self.snapshot(capture)

    def test_no_wait_on_consumer_metadata_lock(self):
        capture = self.make_capture()
        req = _Request("busy", 4)
        source = torch.ones((4, 6), device="cuda", dtype=torch.bfloat16)
        # A separate open description models the consumer's process lock.
        fd = os.open(self.path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            capture.offer([req], [0], [4], source)
        finally:
            os.close(fd)
        # Admission uses a preclaimed slot, so external metadata contention does
        # not drop this copy or stall inference. Publication waits independently.
        self.assertEqual(capture.stats["admitted"], 1)
        _, info, _, _ = self.snapshot(capture)
        self.assertEqual(info["rid"], "busy")

    def test_interrupted_or_missing_chunk_is_explicit(self):
        capture = self.make_capture()
        req = _Request("gap", 10)
        source = torch.ones((2, 6), device="cuda", dtype=torch.bfloat16)
        capture.offer([req], [6], [1], source[:1])
        capture.offer([req], [8], [2], source)
        _, info, tokens, _ = self.snapshot(capture)
        self.assertEqual(info["seal_reason"], "gap")
        self.assertFalse(info["tail_complete"])
        self.assertEqual(tokens, (6,))
        cancelled = _Request("cancelled", 10)
        capture.offer([cancelled], [6], [2], source)
        cancelled.done = True
        _, info, tokens, _ = self.snapshot(capture, slot=1)
        self.assertEqual(info["seal_reason"], "interrupted")
        self.assertFalse(info["tail_complete"])
        self.assertEqual(tokens, (6, 7))

    def test_sampling_and_no_tail_intersection_do_not_reserve(self):
        capture = self.make_capture(sample_rate=0)
        req = _Request("skip", 10)
        source = torch.ones((10, 6), device="cuda", dtype=torch.bfloat16)
        capture.offer([req], [0], [10], source)
        self.assertFalse(capture._pending)
        capture.sample_rate = 1
        capture.offer([req], [0], [4], source[:4])
        self.assertFalse(capture._pending)

    def test_cpu_only_external_process_can_read_and_ack(self):
        capture = self.make_capture()
        req = _Request("external", 10)
        source = torch.arange(60, device="cuda", dtype=torch.bfloat16).reshape(10, 6)
        capture.offer([req], [0], [10], source)
        self.snapshot(capture)
        # Deliberately imports no SGLang, Torch, or CUDA. Parse the wire format
        # directly to catch accidental reliance on process-local Python objects.
        script = r"""
import fcntl, json, mmap, os, struct, sys
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 0)
fcntl.flock(fd, fcntl.LOCK_EX)
arena = mmap.mmap(fd, 0)
magic, version, length, size = struct.unpack_from('<8sIIQ', arena)
assert magic == b'SGDFCAP\0' and version == 2 and size == len(arena)
desc = json.loads(arena[24:24+length])
offset = desc['slot_offset']
state, length, generation = struct.unpack_from('<IIQ', arena, offset)
assert state == 2
info = json.loads(arena[offset+16:offset+16+length])
struct.pack_into('<IIQ', arena, offset, 3, length, generation)
fcntl.flock(fd, fcntl.LOCK_UN)
tokens = struct.unpack_from('<4q', arena, offset+desc['token_offset'])
assert tokens == (6, 7, 8, 9) and info['tail_complete']
raw = struct.unpack_from('<24H', arena, offset+desc['aux_offset'])
expected = tuple(struct.unpack('<I', struct.pack('<f', float(v)))[0] >> 16 for v in range(36,60))
assert raw == expected
fcntl.flock(fd, fcntl.LOCK_EX)
assert struct.unpack_from('<IIQ', arena, offset) == (3, length, generation)
struct.pack_into('<IIQ', arena, offset, 0, length, generation)
print(json.dumps({'rid': info['rid'], 'generation': generation, 'torch_imported': 'torch' in sys.modules}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script, self.path],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {"rid": "external", "generation": 1, "torch_imported": False},
        )
        # The publisher may already have preclaimed the released slot for the
        # next generation. Read under its local lock to avoid a partial header.
        with capture._lock:
            state, length, generation = SLOT_HEADER.unpack_from(
                capture._mapping, capture._slot_offset(0)
            )
        self.assertIn((state, generation), ((FREE, 1), (COPYING, 2)))
        if state == COPYING:
            self.assertEqual(length, 0)

    def test_cuda_graph_output_can_be_replayed_after_capture(self):
        capture = self.make_capture()
        req = _Request("graph", 4)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value = torch.tensor(7, device="cuda", dtype=torch.bfloat16)
            output = torch.empty((4, 6), device="cuda", dtype=torch.bfloat16)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output.copy_(value.expand_as(output))
            graph.replay()
            capture.offer([req], [0], [4], output)
            value.fill_(9)
            graph.replay()
        _, _, _, aux = self.snapshot(capture)
        torch.testing.assert_close(aux, torch.full((4, 6), 7, dtype=torch.bfloat16))

    def test_exclusive_creation_and_cleanup(self):
        capture = self.make_capture()
        with self.assertRaises(FileExistsError):
            self.make_capture()
        req = _Request("close", 4)
        capture.offer(
            [req], [0], [4], torch.ones((4, 6), device="cuda", dtype=torch.bfloat16)
        )
        capture.close()
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(capture._thread.is_alive())
        self.assertIsNone(capture._host)
        self.assertIsNone(capture._mapping)
        capture.close()

    def test_worker_hook_captures_before_clearing_auxiliary_states(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        capture = self.make_capture()
        req = _Request("worker", 4)
        hidden = torch.ones((4, 6), device="cuda", dtype=torch.bfloat16)
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=hidden, target_hidden_states=None
            ),
            next_token_ids=torch.tensor([5], device="cuda"),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            is_extend_in_batch=False,
            reqs=[req],
            prefix_lens=[0],
            extend_lens=[4],
            seq_lens=torch.tensor([4], device="cuda"),
            out_cache_loc=torch.arange(4, device="cuda"),
        )
        worker = SimpleNamespace(
            _validate_phase1_sampling_support=mock.Mock(),
            target_worker=SimpleNamespace(
                forward_batch_generation=mock.Mock(return_value=result)
            ),
            _tp_sync=SimpleNamespace(sync=mock.Mock()),
            model_runner=SimpleNamespace(prefill_attention_backend_str="triton"),
            _append_target_hidden_to_draft_kv_by_loc=mock.Mock(),
            _prefill_capture=capture,
            _make_next_draft_input_prefill=mock.Mock(return_value="next-draft-input"),
        )
        with mock.patch(
            "sglang.srt.speculative.dflash_worker_v2.compute_position",
            return_value=(torch.arange(4, device="cuda"), None),
        ):
            actual = DFlashWorkerV2.forward_batch_generation(worker, batch)
        self.assertIs(actual, result)
        self.assertIsNone(result.logits_output.hidden_states)
        self.assertEqual(result.next_draft_input, "next-draft-input")
        worker._append_target_hidden_to_draft_kv_by_loc.assert_called_once()
        _, info, tokens, aux = self.snapshot(capture)
        self.assertEqual(info["rid"], "worker")
        self.assertEqual(tokens, (0, 1, 2, 3))
        torch.testing.assert_close(aux, torch.ones((4, 6), dtype=torch.bfloat16))

    def test_verify_waits_for_commit_and_excludes_rejected_inputs(self):
        capture = self.make_capture()
        req = _Request("decode", 5)
        req.output_ids = [10]
        source = torch.arange(24, device="cuda", dtype=torch.bfloat16).view(1, 4, 6)
        tokens = torch.tensor([[10, 11, 12, 999]], device="cuda")
        positions = torch.arange(5, 9, device="cuda")
        accepts = torch.tensor([3], dtype=torch.int32, device="cuda")
        allocated = torch.cuda.memory_allocated()
        with mock.patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("host sync")
        ):
            receipt = capture.offer_verify([req], source, tokens, positions, accepts)
        source.fill_(-1)
        tokens.fill_(-1)
        positions.fill_(-1)
        accepts.fill_(0)
        torch.cuda.current_stream().synchronize()
        with capture._lock:
            capture._publish_completed()
        self.snapshot(capture, slot=capture.slots, state=COPYING)
        req.output_ids.extend([11, 12, 77])
        receipt.consume(
            SimpleNamespace(reqs=[req]),
            [SimpleNamespace(output_index=1, token_ids=(11, 12, 77))],
        )
        _, info, ids, aux = self.snapshot(capture, slot=capture.slots)
        self.assertEqual(info["kind"], "verify_committed")
        self.assertEqual((info["start"], info["end"]), (5, 8))
        self.assertEqual(ids, (10, 11, 12))
        self.assertEqual(info["output_tokens"], [11, 12, 77])
        self.assertEqual(info["accepted_draft_count"], 2)
        torch.testing.assert_close(
            aux, torch.arange(18, dtype=torch.bfloat16).view(3, 6)
        )
        self.assertLessEqual(torch.cuda.memory_allocated(), allocated)

    def test_batched_copies_handle_admission_gaps_and_slot_wrap(self):
        capture = self.make_capture(slots=1, verify_slots=5, teacher_hidden_size=2)
        reqs = [_Request(f"sparse-{i}", 5) for i in range(4)]
        reqs[1].lora_id = "not-eligible"
        aux_expected = torch.arange(96, dtype=torch.bfloat16).view(4, 4, 6)
        teacher_expected = torch.arange(32, dtype=torch.bfloat16).view(16, 2)
        for iteration in range(2):
            for i, req in enumerate(reqs):
                req.output_ids = [10 + 10 * i]
            aux = aux_expected.cuda()
            teacher = teacher_expected.cuda()
            tokens = torch.tensor(
                [[10 + 10 * i + j for j in range(4)] for i in range(4)], device="cuda"
            )
            output = capture.offer_verify(
                reqs,
                aux,
                tokens,
                torch.arange(5, 9, device="cuda").repeat(4),
                torch.full((4,), 3, device="cuda", dtype=torch.int32),
                teacher_hidden_states=teacher,
            )
            # Each admitted source has a distinct row range; slots wrap on the
            # second iteration. Reuse the GPU buffers immediately afterwards.
            aux.fill_(-1)
            teacher.fill_(-1)
            self.assertEqual([i for i, _ in output.tickets], [0, 2, 3])
            self.assertEqual(
                [c.slot for _, c in output.tickets],
                [1, 2, 3] if iteration == 0 else [4, 5, 1],
            )
            commits = []
            for i, req in enumerate(reqs):
                retained = (11 + 10 * i, 12 + 10 * i, 999)
                req.output_ids.extend(retained)
                commits.append(SimpleNamespace(output_index=1, token_ids=retained))
            output.consume(SimpleNamespace(reqs=reqs), commits)
            for i, ticket in output.tickets:
                _, info, ids, rows = self.snapshot(capture, slot=ticket.slot)
                self.assertEqual(ids, (10 + 10 * i, 11 + 10 * i, 12 + 10 * i))
                self.assertEqual((info["start"], info["end"]), (5, 8))
                torch.testing.assert_close(rows, aux_expected[i, :3])
                torch.testing.assert_close(
                    self.teacher_snapshot(capture, ticket.slot, 3),
                    teacher_expected[i * 4 : i * 4 + 3],
                )
                self.set_state(capture, FREE, slot=ticket.slot)
            with capture._lock:
                capture._refill_available()

    def test_overlap_keeps_metadata_ordered_and_fences_target_buffer_reuse(self):
        capture = self.make_capture(overlap=True, teacher_hidden_size=2)
        req = _Request("overlap", 5)
        req.output_ids = [10]
        aux = torch.arange(24, device="cuda", dtype=torch.bfloat16).view(1, 4, 6)
        teacher = torch.arange(8, device="cuda", dtype=torch.bfloat16).view(4, 2)
        tokens = torch.tensor([[10, 11, 12, 999]], device="cuda")
        positions = torch.arange(5, 9, device="cuda")
        lengths = torch.tensor([3], device="cuda", dtype=torch.int32)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        with torch.cuda.stream(capture._copy_stream):
            torch.cuda._sleep(100_000_000)
        with mock.patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("host sync")
        ):
            receipt = capture.offer_verify(
                [req], aux, tokens, positions, lengths, teacher_hidden_states=teacher
            )
            # These scratch tensors may be reused by draft/verify preparation
            # before the next target forward. Their D2H stays on this stream.
            tokens.fill_(-1)
            positions.fill_(-1)
            lengths.fill_(0)
            marker = torch.cuda.Event()
            marker.record()
        marker.synchronize()
        # CUDA work-queue scheduling may serialize the artificial sleep and
        # marker even on separate streams. Check ordering/data here; actual
        # overlap and serving cost are measured separately with Nsight/traffic.
        self.assertIsNotNone(capture._last_copy_event)
        capture.wait_before_target_forward()
        aux.fill_(-1)
        teacher.fill_(-1)
        req.output_ids.extend([11, 12, 77])
        receipt.consume(
            SimpleNamespace(reqs=[req]),
            [SimpleNamespace(output_index=1, token_ids=(11, 12, 77))],
        )
        _, info, ids, observed = self.snapshot(capture, slot=capture.slots)
        self.assertEqual((info["start"], info["end"], ids), (5, 8, (10, 11, 12)))
        torch.testing.assert_close(
            observed, torch.arange(18, dtype=torch.bfloat16).view(3, 6)
        )
        torch.testing.assert_close(
            self.teacher_snapshot(capture, capture.slots, 3),
            torch.arange(6, dtype=torch.bfloat16).view(3, 2),
        )
        self.assertLessEqual(torch.cuda.memory_allocated(), allocated)

    def test_overlap_prefill_survives_next_target_graph_replay(self):
        capture = self.make_capture(overlap=True)
        source = torch.ones((4, 6), device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            source.add_(1)
        graph.replay()
        expected = source.cpu()
        with torch.cuda.stream(capture._copy_stream):
            torch.cuda._sleep(100_000_000)
        capture.offer([_Request("graph-overlap", 4)], [0], [4], source)
        capture.wait_before_target_forward()
        graph.replay()
        _, _, _, observed = self.snapshot(capture)
        torch.testing.assert_close(observed, expected)

    def test_verify_stop_trimming_retains_only_visible_input_features(self):
        capture = self.make_capture()
        req = _Request("stop", 5)
        req.output_ids = [10]
        receipt = capture.offer_verify(
            [req],
            torch.ones((1, 4, 6), device="cuda", dtype=torch.bfloat16),
            torch.tensor([[10, 11, 12, 13]], device="cuda"),
            torch.arange(5, 9, device="cuda"),
            torch.tensor([4], device="cuda", dtype=torch.int32),
        )
        # Scheduler retains just one new token after EOS/grammar/token limit.
        req.output_ids.extend([11, 12, 13, 77])
        req.done = True
        receipt.consume(
            SimpleNamespace(reqs=[req]),
            [SimpleNamespace(output_index=1, token_ids=(11,))],
        )
        _, info, ids, _ = self.snapshot(capture, slot=capture.slots)
        self.assertEqual(ids, (10, 11))
        self.assertEqual(info["end"], 7)
        self.assertTrue(info["finished"])

    def test_verify_discard_and_position_mismatch_release_slots(self):
        capture = self.make_capture()
        req = _Request("overshoot", 5)
        req.output_ids = [10]
        args = (
            [req],
            torch.ones((1, 4, 6), device="cuda", dtype=torch.bfloat16),
            torch.tensor([[10, 11, 12, 13]], device="cuda"),
            torch.arange(6, 10, device="cuda"),
            torch.tensor([3], device="cuda", dtype=torch.int32),
        )
        first = capture.offer_verify(*args)
        second = capture.offer_verify(*args)
        first.consume(SimpleNamespace(reqs=[req]), [None])
        req.output_ids.extend([11, 12, 77])
        second.consume(
            SimpleNamespace(reqs=[req]),
            [SimpleNamespace(output_index=1, token_ids=(11, 12, 77))],
        )
        torch.cuda.current_stream().synchronize()
        with capture._lock:
            capture._publish_completed()
        self.assertFalse(capture._pending)
        self.assertEqual(capture.stats["discarded"], 2)
        self.assertEqual(capture.stats["dropped_commit_mismatch"], 1)
        self.assertTrue(
            all(
                SLOT_HEADER.unpack_from(capture._mapping, capture._slot_offset(i))[0]
                in (FREE, COPYING)
                for i in range(capture.slots, capture.slots + 2)
            )
        )

    def test_prefix_identity_namespaces_and_target_update(self):
        capture = self.make_capture()
        first, second = _Request("a", 200), _Request("b", 200)
        a = capture._prefix_for(first, 140)
        self.assertEqual(a, capture._prefix_for(second, 140))
        self.assertEqual(len(a["page_tokens"]), 12)
        second.origin_input_ids[0] = 999
        second._dflash_capture_cursor = None
        self.assertNotEqual(a, capture._prefix_for(second, 140))
        second.origin_input_ids[0] = 0
        second.cache_salt = "private"
        second._dflash_capture_cursor = None
        self.assertNotEqual(
            a["namespace"], capture._prefix_for(second, 140)["namespace"]
        )
        capture.invalidate_target()
        self.assertNotEqual(
            a["namespace"], capture._prefix_for(first, 140)["namespace"]
        )
        for field in ("input_embeds", "positional_embed_overrides", "lora_id"):
            req = _Request("unsupported", 5)
            setattr(req, field, "not-token-only")
            self.assertFalse(capture._eligible(req))

    def test_verify_pool_is_bounded_and_does_not_consume_prefill_slots(self):
        capture = self.make_capture(slots=1, verify_slots=3)
        self.assertEqual(capture.descriptor["verify_slots"], 3)
        req = _Request("pending-decode", 5)
        req.output_ids = [10]
        args = (
            [req],
            torch.ones((1, 4, 6), device="cuda", dtype=torch.bfloat16),
            torch.tensor([[10, 11, 12, 13]], device="cuda"),
            torch.arange(5, 9, device="cuda"),
            torch.tensor([3], device="cuda", dtype=torch.int32),
        )
        for _ in range(capture.verify_slots):
            self.assertIsNotNone(capture.offer_verify(*args))
        self.assertIsNone(capture.offer_verify(*args))
        self.assertEqual(len(capture._pending), capture.verify_slots)
        fresh = _Request("prefill-still-admitted", 4)
        capture.offer([fresh], [0], [4], args[1].view(4, 6))
        _, info, _, _ = self.snapshot(capture)
        self.assertEqual(info["rid"], fresh.rid)
        self.assertEqual(capture.stats["dropped_full"], 1)


if __name__ == "__main__":
    unittest.main()
