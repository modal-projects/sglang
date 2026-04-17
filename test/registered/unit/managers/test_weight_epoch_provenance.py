import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    GenerateReqInput,
    PauseGenerationReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _AwaitableNone:
    def __await__(self):
        if False:
            yield None
        return None


class _RecordingSender:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)
        return _AwaitableNone()


class _NoopAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeModelUpdateLock:
    def __init__(self):
        self.writer_lock = _NoopAsyncContext()

    async def is_locked(self) -> bool:
        return False


class TestWeightEpochProvenance(unittest.TestCase):
    def _new_tokenizer_manager(self) -> TokenizerManager:
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.current_weight_epoch = 7
        tm.next_weight_epoch = 8
        tm.weight_version_by_epoch = {2: "v2", 5: "v5", 7: "v7"}
        tm.weight_epoch_reservation_lock = asyncio.Lock()
        tm.weight_update_orchestration_lock = asyncio.Lock()
        tm.is_pause = False
        tm.is_pause_cond = asyncio.Condition()
        tm.auto_create_handle_loop = lambda: None
        tm.send_to_scheduler = _RecordingSender()
        tm.abort_request = MagicMock()
        tm.model_update_lock = _FakeModelUpdateLock()
        tm.server_args = SimpleNamespace(
            weight_version="v7",
            load_format="auto",
            dp_size=1,
            enable_dp_attention=False,
            model_path="model-v7",
        )
        tm.served_model_name = "model-v7"
        tm.model_path = "model-v7"
        return tm

    def test_generate_req_getitem_indexes_extra_key_list(self):
        req = GenerateReqInput(
            text=["hello", "world"],
            sampling_params={"max_new_tokens": 1},
            extra_key=["cache-a", "cache-b"],
        )
        req.normalize_batch_and_arguments()

        self.assertEqual(req[0].extra_key, "cache-a")
        self.assertEqual(req[1].extra_key, "cache-b")

    def test_generate_req_parallel_sampling_expands_extra_key_list(self):
        req = GenerateReqInput(
            text=["hello", "world"],
            sampling_params={"max_new_tokens": 1, "n": 2},
            extra_key=["cache-a", "cache-b"],
        )
        req.normalize_batch_and_arguments()

        self.assertEqual([req[i].extra_key for i in range(4)], ["cache-a", "cache-b", "cache-a", "cache-b"])

    def test_stamp_request_weight_context_namespaces_cache(self):
        tm = self._new_tokenizer_manager()
        req = GenerateReqInput(
            text="hello",
            sampling_params={"max_new_tokens": 1},
            extra_key="tenant-a",
        )
        req.normalize_batch_and_arguments()

        tm._stamp_request_weight_context(req)

        self.assertEqual(req.request_weight_epoch, 7)
        self.assertEqual(req.cache_epoch, 7)
        self.assertEqual(req.request_weight_version, "v7")
        self.assertEqual(req.extra_key, "tenant-a|wv=7")

    def test_weight_provenance_marks_mixed_resume(self):
        tm = self._new_tokenizer_manager()
        req = GenerateReqInput(
            text="hello",
            sampling_params={"max_new_tokens": 2},
        )
        req.normalize_batch_and_arguments()
        req.request_weight_epoch = 2
        req.cache_epoch = 2
        req.request_weight_version = "v2"

        state = ReqState([], False, asyncio.Event(), req, MagicMock())
        state.output_token_weight_epochs = [2, 5]

        meta_info = {}
        tm._add_weight_provenance_to_meta_info(meta_info, state)

        self.assertEqual(meta_info["weight_epoch_start"], 2)
        self.assertEqual(meta_info["weight_epoch_end"], 5)
        self.assertEqual(meta_info["cache_epoch"], 2)
        self.assertTrue(meta_info["mixed_weight_epochs"])
        self.assertTrue(meta_info["resume_from_stale_kv"])
        self.assertEqual(meta_info["weight_version_start"], "v2")
        self.assertEqual(meta_info["weight_version_end"], "v5")
        self.assertEqual(meta_info["weight_version"], "v5")
        self.assertEqual(meta_info["output_token_weight_epochs"], [2, 5])

    def test_schedule_batch_copy_preserves_launch_weight_epoch(self):
        batch = ScheduleBatch(reqs=[])
        batch.launch_weight_epoch = 11

        copied = batch.copy()

        self.assertEqual(copied.launch_weight_epoch, 11)

    def test_reserve_weight_epoch_is_unique_under_concurrency(self):
        tm = self._new_tokenizer_manager()
        objs = [SimpleNamespace(weight_epoch=None), SimpleNamespace(weight_epoch=None)]

        async def reserve_all():
            await asyncio.gather(*(tm._reserve_weight_epoch(obj) for obj in objs))

        asyncio.run(reserve_all())

        self.assertEqual({obj.weight_epoch for obj in objs}, {8, 9})
        self.assertEqual(tm.next_weight_epoch, 10)

    def test_atomic_pause_wrapper_pauses_and_resumes(self):
        tm = self._new_tokenizer_manager()
        tm.pause_generation = AsyncMock()
        tm.continue_generation = AsyncMock()
        obj = SimpleNamespace(atomic_pause_mode="in_place", flush_cache=False)

        async def run():
            return await tm._run_weight_update_with_optional_pause(
                obj, AsyncMock(return_value=("ok", 1))
            )

        result = asyncio.run(run())

        self.assertEqual(result, ("ok", 1))
        tm.pause_generation.assert_awaited_once_with(
            PauseGenerationReqInput(mode="in_place")
        )
        tm.continue_generation.assert_awaited_once_with(ContinueGenerationReqInput())

    def test_atomic_pause_wrapper_does_not_resume_external_pause(self):
        tm = self._new_tokenizer_manager()
        tm.is_pause = True
        tm.pause_generation = AsyncMock()
        tm.continue_generation = AsyncMock()
        obj = SimpleNamespace(atomic_pause_mode="in_place", flush_cache=False)

        async def run():
            return await tm._run_weight_update_with_optional_pause(
                obj, AsyncMock(return_value="ok")
            )

        result = asyncio.run(run())

        self.assertEqual(result, "ok")
        tm.pause_generation.assert_not_awaited()
        tm.continue_generation.assert_not_awaited()

    def test_atomic_pause_wrapper_rejects_inplace_flush_cache(self):
        tm = self._new_tokenizer_manager()
        obj = SimpleNamespace(atomic_pause_mode="in_place", flush_cache=True)

        async def run():
            await tm._run_weight_update_with_optional_pause(
                obj, AsyncMock(return_value=None)
            )

        with self.assertRaisesRegex(
            ValueError, "flush_cache must be false when atomic_pause_mode='in_place'"
        ):
            asyncio.run(run())

    def test_weight_update_wrapper_serializes_plain_and_atomic_updates(self):
        tm = self._new_tokenizer_manager()
        tm.pause_generation = AsyncMock()
        tm.continue_generation = AsyncMock()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()
        events = []

        async def first_update():
            events.append("first-start")
            first_started.set()
            await release_first.wait()
            events.append("first-end")
            return "first"

        async def second_update():
            events.append("second-start")
            second_started.set()
            return "second"

        async def run():
            first_task = asyncio.create_task(
                tm._run_weight_update_with_optional_pause(
                    SimpleNamespace(atomic_pause_mode="retract", flush_cache=True),
                    first_update,
                )
            )
            await first_started.wait()

            second_task = asyncio.create_task(
                tm._run_weight_update_with_optional_pause(
                    SimpleNamespace(atomic_pause_mode=None),
                    second_update,
                )
            )
            await asyncio.sleep(0)
            self.assertFalse(second_started.is_set())

            release_first.set()
            return await asyncio.gather(first_task, second_task)

        results = asyncio.run(run())

        self.assertEqual(results, ["first", "second"])
        self.assertEqual(events, ["first-start", "first-end", "second-start"])
        tm.pause_generation.assert_awaited_once_with(
            PauseGenerationReqInput(mode="retract")
        )
        tm.continue_generation.assert_awaited_once_with(ContinueGenerationReqInput())

    def test_update_weights_from_disk_sends_pause_before_atomic_update(self):
        tm = self._new_tokenizer_manager()
        seen_updates = []
        update_sent = asyncio.Event()
        release_update = asyncio.Event()

        async def fake_wait_for_model_update(obj):
            tm.send_to_scheduler.send_pyobj(obj)
            seen_updates.append(obj)
            update_sent.set()
            await release_update.wait()
            return True, "ok", 0

        tm._wait_for_model_update_from_disk = fake_wait_for_model_update
        obj = UpdateWeightFromDiskReqInput(
            model_path="atomic-model",
            atomic_pause_mode="in_place",
            flush_cache=False,
            weight_version="atomic-v1",
        )

        async def run():
            task = asyncio.create_task(tm.update_weights_from_disk(obj, None))
            await update_sent.wait()
            sent_before_reply = [type(item).__name__ for item in tm.send_to_scheduler.sent]
            release_update.set()
            result = await task
            sent_after_reply = [type(item).__name__ for item in tm.send_to_scheduler.sent]
            return sent_before_reply, sent_after_reply, result

        sent_before_reply, sent_after_reply, result = asyncio.run(run())

        self.assertEqual(
            sent_before_reply,
            ["PauseGenerationReqInput", "UpdateWeightFromDiskReqInput"],
        )
        self.assertEqual(
            sent_after_reply,
            [
                "PauseGenerationReqInput",
                "UpdateWeightFromDiskReqInput",
                "ContinueGenerationReqInput",
            ],
        )
        self.assertEqual(seen_updates, [obj])
        self.assertEqual(result, (True, "ok Weight version updated to atomic-v1.", 0))


if __name__ == "__main__":
    unittest.main()
