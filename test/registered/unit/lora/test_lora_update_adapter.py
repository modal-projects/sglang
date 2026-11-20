"""Unit tests for /update_lora_adapter: read the adapter on every rank while serving,
then switch on every rank at one step.

Covers:
  * the tokenizer: prepare before apply, the retract pause scope, a pause someone else
    started, a failed flush, a pause asked for during an update, and the weight version
    switching with the adapter
  * the ranks: they apply what they prepared only when every rank holds it, and
    otherwise load as /load_lora_adapter does
  * the scheduler: one prepare at a time, no prepare where control requests reach the
    ranks at different steps, and the weight version recorded at the switch
  * the manager's prepare record
  * the prepare process: a clean interpreter that lists an adapter's files and reads
    them into slot order, casting at the copy as the install does
  * the new requests over msgpack IPC
"""

import asyncio
import mmap
import os
import tempfile
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import msgspec
import torch
from safetensors.torch import save_file

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_manager import LoRAManager, _PreparedLoRA  # noqa: E402
from sglang.srt.lora.lora_prepare_process import (  # noqa: E402
    LoRAPrepareClient,
    list_file_tensors,
    node_address,
)
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    ApplyLoRAAdapterReqInput,
    ContinueGenerationReqInput,
    FlushCacheReqOutput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    PauseGenerationReqInput,
    PrepareLoRAAdapterReqInput,
    PrepareLoRAAdapterReqOutput,
    UpdateLoRAAdapterReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.managers.tp_worker import BaseTpWorker  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _event(obj) -> str:
    if isinstance(obj, PauseGenerationReqInput):
        return f"pause:{obj.mode}"
    return {
        ContinueGenerationReqInput: "continue",
        PrepareLoRAAdapterReqInput: "prepare",
        ApplyLoRAAdapterReqInput: "apply",
    }[type(obj)]


def _tokenizer(
    registry,
    *,
    prepared=True,
    flush_ok=True,
    applied=True,
    during_prepare=None,
    during_apply=None,
):
    """A TokenizerManager whose scheduler side is recorded in `events`, and the applies
    it sends in `tm.applies`."""
    events, applies = [], []

    async def dispatch(obj):
        events.append(_event(obj))

    async def prepare(obj):
        events.append(_event(obj))
        if during_prepare is not None:
            await during_prepare()
        return [PrepareLoRAAdapterReqOutput(prepared=prepared, message="declined")]

    async def update(obj):
        events.append(_event(obj))
        applies.append(obj)
        if during_apply is not None:
            await during_apply()
        return [LoRAUpdateOutput(success=applied, loaded_adapters={})]

    async def flush_cache():
        events.append("flush")
        return FlushCacheReqOutput(success=flush_ok)

    tm = object.__new__(TokenizerManager)
    tm.__dict__.update(
        auto_create_handle_loop=lambda: None,
        lora_update_lock=asyncio.Lock(),
        is_pause_cond=asyncio.Condition(),
        _is_pause=False,
        _engine_state_changed_callback=None,
        pending_lora_unloads={},
        _pending_lora_publications={},
        lora_registry=registry,
        lora_ref_cache={},
        prepare_lora_adapter_communicator=prepare,
        update_lora_adapter_communicator=update,
        _async_dispatch_to_scheduler=dispatch,
        flush_cache=flush_cache,
        _weight_update_session_open=False,
        record_config_updates=Mock(),
        applies=applies,
    )
    return tm, events


def _patched_tokenizer_context():
    return patch.multiple(
        "sglang.srt.managers.tokenizer_control_mixin",
        get_lora=lambda: SimpleNamespace(enable_lora=True, max_loaded_loras=None),
        get_parallel=lambda: SimpleNamespace(dp_size=1, pp_size=1),
        get_serving=lambda: SimpleNamespace(tokenizer_worker_num=1),
    )


def _registry_with_a() -> LoRARegistry:
    registry = LoRARegistry()
    asyncio.run(registry.register(LoRARef(lora_name="a", lora_path="/v1")))
    return registry


def _update(pause_mode="in_place", weight_version=None) -> UpdateLoRAAdapterReqInput:
    return UpdateLoRAAdapterReqInput(
        lora_name="a",
        lora_path="/v2",
        pause_mode=pause_mode,
        weight_version=weight_version,
    )


class TestUpdateLoRAAdapterTokenizer(CustomTestCase):
    def test_in_place_prepares_then_applies_without_pausing(self):
        registry = _registry_with_a()
        tm, events = _tokenizer(registry)
        with _patched_tokenizer_context():
            result = asyncio.run(tm.update_lora_adapter(_update()))
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(events, ["prepare", "apply"])
        self.assertEqual(registry.get_all_adapters()["a"].lora_path, "/v2")

    def test_retract_pauses_flushes_applies_and_resumes(self):
        tm, events = _tokenizer(_registry_with_a())
        with _patched_tokenizer_context():
            result = asyncio.run(tm.update_lora_adapter(_update("retract")))
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(
            events, ["prepare", "pause:retract", "flush", "apply", "continue"]
        )
        self.assertFalse(tm.is_pause)

    def test_retract_on_a_paused_engine_leaves_the_pause_alone(self):
        tm, events = _tokenizer(_registry_with_a())
        tm.is_pause = True
        with _patched_tokenizer_context():
            result = asyncio.run(tm.update_lora_adapter(_update("retract")))
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(events, ["prepare", "apply"])
        self.assertTrue(tm.is_pause)

    def test_retract_with_a_failed_flush_resumes_and_applies_nothing(self):
        registry = _registry_with_a()
        tm, events = _tokenizer(registry, flush_ok=False)
        with _patched_tokenizer_context():
            result = asyncio.run(tm.update_lora_adapter(_update("retract")))
        self.assertFalse(result.success)
        self.assertEqual(events, ["prepare", "pause:retract", "flush", "continue"])
        self.assertFalse(tm.is_pause)
        self.assertEqual(registry.get_all_adapters()["a"].lora_path, "/v1")

    def test_retract_of_a_new_name_does_not_pause(self):
        tm, events = _tokenizer(LoRARegistry())
        with _patched_tokenizer_context():
            result = asyncio.run(tm.update_lora_adapter(_update("retract")))
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(events, ["prepare", "apply"])

    def test_not_prepared_fails_and_changes_nothing(self):
        for pause_mode in ("in_place", "retract"):
            with self.subTest(pause_mode=pause_mode):
                registry = _registry_with_a()
                tm, events = _tokenizer(registry, prepared=False)
                with _patched_tokenizer_context():
                    result = asyncio.run(tm.update_lora_adapter(_update(pause_mode)))
                self.assertFalse(result.success)
                self.assertIn("declined", result.error_message)
                self.assertEqual(events, ["prepare"])
                self.assertEqual(registry.get_all_adapters()["a"].lora_path, "/v1")

    def test_retract_prepares_while_requests_still_come_in(self):
        held = []

        async def during_prepare():
            held.append(tm.is_pause_cond.locked())

        tm, events = _tokenizer(_registry_with_a(), during_prepare=during_prepare)
        with _patched_tokenizer_context():
            result = asyncio.run(tm.update_lora_adapter(_update("retract")))
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(held, [False])
        self.assertEqual(
            events, ["prepare", "pause:retract", "flush", "apply", "continue"]
        )

    def test_an_adapter_changed_while_prepared_is_not_switched(self):
        registry = _registry_with_a()

        async def during_prepare():
            await registry.unregister("a")

        tm, events = _tokenizer(registry, during_prepare=during_prepare)
        with _patched_tokenizer_context():
            result = asyncio.run(tm.update_lora_adapter(_update("retract")))
        self.assertFalse(result.success)
        self.assertIn("changed while it was being prepared", result.error_message)
        self.assertEqual(events, ["prepare"])

    def test_pause_asked_for_during_an_update_stays_in_effect(self):
        registry = _registry_with_a()

        async def scenario():
            applying, release = asyncio.Event(), asyncio.Event()

            async def during_apply():
                applying.set()
                await release.wait()

            tm, events = _tokenizer(registry, during_apply=during_apply)
            update = asyncio.create_task(tm.update_lora_adapter(_update("retract")))
            await applying.wait()
            pause = asyncio.create_task(
                tm.pause_generation(PauseGenerationReqInput(mode="in_place"))
            )
            await asyncio.sleep(0)
            self.assertFalse(pause.done())
            release.set()
            result = await update
            await pause
            return result, events, tm.is_pause

        with _patched_tokenizer_context():
            result, events, is_pause = asyncio.run(scenario())
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(
            events,
            [
                "prepare",
                "pause:retract",
                "flush",
                "apply",
                "continue",
                "pause:in_place",
            ],
        )
        self.assertTrue(is_pause)

    def test_the_weight_version_switches_with_the_adapter(self):
        for pause_mode in ("in_place", "retract"):
            with self.subTest(pause_mode=pause_mode):
                tm, _ = _tokenizer(_registry_with_a())
                with _patched_tokenizer_context():
                    result = asyncio.run(
                        tm.update_lora_adapter(_update(pause_mode, weight_version="7"))
                    )
                self.assertTrue(result.success, result.error_message)
                self.assertEqual([a.weight_version for a in tm.applies], ["7"])
                tm.record_config_updates.assert_called_once_with(
                    "tokenizer.weight_version", weight_version="7"
                )

    def test_a_failed_update_keeps_the_weight_version(self):
        for kwargs in ({"prepared": False}, {"applied": False}):
            with self.subTest(**kwargs):
                tm, _ = _tokenizer(_registry_with_a(), **kwargs)
                with _patched_tokenizer_context():
                    result = asyncio.run(
                        tm.update_lora_adapter(_update(weight_version="7"))
                    )
                self.assertFalse(result.success)
                tm.record_config_updates.assert_not_called()


def _worker(manager):
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            lora_manager=manager,
            load_lora_adapter=Mock(return_value=LoRAUpdateOutput(success=True)),
        )
    )
    worker.load_lora_adapter = lambda req: BaseTpWorker.load_lora_adapter(worker, req)
    return worker


def _apply(upsert=False) -> ApplyLoRAAdapterReqInput:
    return ApplyLoRAAdapterReqInput(
        load=LoadLoRAAdapterReqInput(
            lora_name="a", lora_path="/v2", lora_id="id-a", upsert=upsert
        ),
        prepare_id="p1",
    )


class TestApplyOnRanks(CustomTestCase):
    def _run(self, own_prepared, gathered=None):
        manager = Mock()
        manager.prepared_outcome.return_value = PrepareLoRAAdapterReqOutput(
            prepared=own_prepared
        )
        manager.apply_prepared_lora_adapter.return_value = LoRAUpdateOutput(
            success=True
        )
        manager.create_lora_update_result.side_effect = (
            lambda success, error_message=None: LoRAUpdateOutput(
                success=success, error_message=error_message
            )
        )
        worker = _worker(manager)
        group = MagicMock(world_size=1 if gathered is None else len(gathered))
        group.all_gather_object.return_value = gathered
        with patch(
            "sglang.srt.managers.tp_worker.get_parallel",
            return_value=SimpleNamespace(tp_group=group),
        ):
            result = BaseTpWorker.apply_lora_adapter(worker, _apply())
        return result, manager, worker

    def test_every_rank_prepared_applies_what_was_prepared(self):
        result, manager, worker = self._run(True, gathered=[(True, ""), (True, "")])
        self.assertTrue(result.success)
        manager.apply_prepared_lora_adapter.assert_called_once()
        worker.model_runner.load_lora_adapter.assert_not_called()

    def test_a_rank_not_prepared_changes_nothing_on_every_rank(self):
        result, manager, worker = self._run(
            True, gathered=[(True, ""), (False, "read failed")]
        )
        self.assertFalse(result.success)
        self.assertIn("read failed", result.error_message)
        manager.apply_prepared_lora_adapter.assert_not_called()
        manager.discard_prepared_lora_adapter.assert_called_once_with("p1")
        worker.model_runner.load_lora_adapter.assert_not_called()

    def test_single_rank_not_prepared_changes_nothing(self):
        result, manager, worker = self._run(False)
        self.assertFalse(result.success)
        manager.apply_prepared_lora_adapter.assert_not_called()
        worker.model_runner.load_lora_adapter.assert_not_called()

    def test_prepare_reply_waits_for_the_rank_to_finish(self):
        manager = Mock()
        worker = _worker(manager)
        request = PrepareLoRAAdapterReqInput(load=_apply().load, prepare_id="p1")
        with patch(
            "sglang.srt.managers.tp_worker.get_parallel",
            return_value=SimpleNamespace(tp_group=MagicMock(world_size=1)),
        ):
            manager.prepare_reply.return_value = None
            self.assertIsNone(BaseTpWorker.lora_prepare_reply(worker, request))
            manager.prepare_reply.return_value = PrepareLoRAAdapterReqOutput(
                prepared=True, stats={"prepare_s": 0.5}
            )
            reply = BaseTpWorker.lora_prepare_reply(worker, request)
        self.assertTrue(reply.prepared)
        self.assertEqual(reply.stats, {"prepare_s": 0.5})


def _scheduler(*, pp_size=1, per_dp_group=False):
    scheduler = object.__new__(Scheduler)
    scheduler.pending_lora_prepare = None
    scheduler.lora_prepare_address = "/tmp/node.sock"
    scheduler.tp_worker = Mock()
    scheduler.tp_worker.load_lora_adapter.return_value = LoRAUpdateOutput(success=True)
    scheduler.kv_event_lora_names = Mock()
    scheduler.weight_updater = Mock()
    scheduler.send_control_output = Mock()
    scheduler.request_receiver = SimpleNamespace(
        broadcasts_control_per_dp_group=lambda: per_dp_group
    )
    context = patch(
        "sglang.srt.managers.scheduler.get_parallel",
        return_value=SimpleNamespace(pp_size=pp_size),
    )
    return scheduler, context


def _prepare(prepare_id="p1") -> PrepareLoRAAdapterReqInput:
    return PrepareLoRAAdapterReqInput(load=_apply().load, prepare_id=prepare_id)


class TestSchedulerPrepare(CustomTestCase):
    def test_one_prepare_at_a_time_and_reply_when_done(self):
        scheduler, context = _scheduler()
        first, second = _prepare("p1"), _prepare("p2")
        with context:
            self.assertIsNone(scheduler.prepare_lora_adapter(first))
            busy = scheduler.prepare_lora_adapter(second)
            scheduler.tp_worker.lora_prepare_reply.return_value = None
            scheduler.check_pending_lora_prepare()
            scheduler.send_control_output.assert_not_called()
            done = PrepareLoRAAdapterReqOutput(prepared=True)
            scheduler.tp_worker.lora_prepare_reply.return_value = done
            scheduler.check_pending_lora_prepare()
        self.assertFalse(busy.prepared)
        scheduler.tp_worker.prepare_lora_adapter.assert_called_once_with(
            first, "/tmp/node.sock"
        )
        scheduler.send_control_output.assert_called_once_with(done, first)
        self.assertIsNone(scheduler.pending_lora_prepare)

    def test_ranks_receiving_control_apart_do_not_prepare(self):
        for kwargs in ({"pp_size": 2}, {"per_dp_group": True}):
            with self.subTest(**kwargs):
                scheduler, context = _scheduler(**kwargs)
                with context:
                    reply = scheduler.prepare_lora_adapter(_prepare())
                self.assertFalse(reply.prepared)
                self.assertIn("/load_lora_adapter", reply.message)
                self.assertIsNone(scheduler.pending_lora_prepare)
                scheduler.tp_worker.prepare_lora_adapter.assert_not_called()

    def test_the_weight_version_is_recorded_at_a_successful_switch(self):
        for success in (True, False):
            with self.subTest(success=success):
                scheduler, _ = _scheduler()
                scheduler.tp_worker.apply_lora_adapter.return_value = LoRAUpdateOutput(
                    success=success
                )
                apply = _apply()
                apply.weight_version = "7"
                scheduler.apply_lora_adapter(apply)
                record = scheduler.weight_updater.record_weight_version_after_update
                if success:
                    record.assert_called_once_with("7")
                else:
                    record.assert_not_called()


class TestPreparedRecord(CustomTestCase):
    def _manager(self, *, cpu_backup=True, device="cuda"):
        manager = object.__new__(LoRAManager)
        manager._prepared = None
        manager.lora_no_cpu_backup = not cpu_backup
        manager.device = torch.device(device)
        return manager

    def test_not_prepared_without_cpu_copies_or_cuda(self):
        ref = LoRARef(lora_name="a", lora_path="/v2")
        for manager in (self._manager(cpu_backup=False), self._manager(device="cpu")):
            manager.prepare_lora_adapter(ref, "p1", "/tmp/node.sock")
            reply = manager.prepare_reply("p1")
            self.assertFalse(reply.prepared)
            self.assertTrue(reply.message)

    def test_another_prepare_is_not_held(self):
        manager = self._manager()
        manager._prepared = _PreparedLoRA(
            "p1", reply=PrepareLoRAAdapterReqOutput(prepared=True)
        )
        reply = manager.prepare_reply("p2")
        self.assertFalse(reply.prepared)
        self.assertIn("not held", reply.message)

    def test_a_running_prepare_is_never_dropped(self):
        manager = self._manager()
        running = _PreparedLoRA("p1")
        manager._prepared = running
        manager.discard_prepared_lora_adapter("p1")
        self.assertIs(manager._prepared, running)
        with self.assertRaises(AssertionError):
            manager.prepare_lora_adapter(
                LoRARef(lora_name="a", lora_path="/v2"), "p2", "/tmp/node.sock"
            )
        running.reply = PrepareLoRAAdapterReqOutput(prepared=True)
        manager.discard_prepared_lora_adapter("p2")
        self.assertIs(manager._prepared, running)
        manager.discard_prepared_lora_adapter("p1")
        self.assertIsNone(manager._prepared)


def _unpinned_staging(self, nbytes):
    if self._staging is None or self._staging.numel() < nbytes:
        os.ftruncate(self._staging_fd, nbytes)
        self._staging = torch.frombuffer(
            mmap.mmap(self._staging_fd, nbytes), dtype=torch.uint8
        )


def _replies(*clients: LoRAPrepareClient) -> list:
    replies = [None] * len(clients)
    deadline = time.monotonic() + 120
    while any(reply is None for reply in replies):
        if time.monotonic() > deadline:
            raise TimeoutError("no reply from the LoRA prepare process")
        for i, client in enumerate(clients):
            if replies[i] is None:
                replies[i] = client.poll()
        time.sleep(0.05)
    return replies


def _regions(reply, layout, dtype):
    result, shared = reply
    return [
        shared[offset : offset + numel * dtype.itemsize].view(dtype)
        for offset, (_, numel) in zip(result["region_offsets"], layout[0])
    ]


@patch.object(LoRAPrepareClient, "_fit_staging", _unpinned_staging)
class TestNodePrepareProcess(CustomTestCase):
    def test_gpus_share_alike_regions_and_read_their_own(self):
        bf16 = torch.bfloat16
        a = torch.randn(3, 4, dtype=torch.float32)
        b = torch.randn(2, 3).to(bf16)
        c = torch.randn(4).to(bf16)
        # Both GPUs hold `a` alike (as stored); GPU 0 also holds `b` transposed and
        # scaled, with two unwritten positions, GPU 1 holds `c`.
        layouts = [
            (
                [(0, 12), (12, 8)],
                [
                    (0, 0, (4, 1), 0, (4, 1), (3, 4), False),
                    (1, 0, (1, 3), 12, (2, 1), (3, 2), True),
                ],
            ),
            (
                [(0, 12), (12, 4)],
                [
                    (0, 0, (4, 1), 0, (4, 1), (3, 4), False),
                    (2, 0, (1,), 12, (1,), (4,), False),
                ],
            ),
        ]
        expected = [torch.zeros(20, dtype=bf16), torch.zeros(16, dtype=bf16)]
        for slot in expected:
            slot[:12] = a.reshape(-1).to(bf16)
        expected[0][12:18] = (b.t() * 0.3).reshape(-1)
        expected[1][12:16] = c
        address = node_address(uuid.uuid4().hex[:12])
        with tempfile.TemporaryDirectory() as directory:
            file = os.path.join(directory, "adapter_model.safetensors")
            # A zero-element tensor, as PEFT writes for some modules, is listed too.
            save_file(
                {"a": a, "b": b, "c": c, "z": torch.empty(0, 4, dtype=bf16)}, file
            )
            gpus = [LoRAPrepareClient(address), LoRAPrepareClient(address)]
            try:
                self.assertIsNotNone(gpus[0]._listener)
                self.assertIsNone(gpus[1]._listener)

                def update(prepare_id, trace):
                    for gpu in gpus:
                        gpu.list_tensors(prepare_id, [file])
                    listed = [result for result, _ in _replies(*gpus)]
                    tensors = None
                    if trace:
                        for gpu in gpus:
                            gpu.request_tensors(prepare_id)
                        tensors = [result["tensors"] for result, _ in _replies(*gpus)]
                    for gpu, layout, slot in zip(gpus, layouts, expected):
                        gpu.read(prepare_id, "layout", layout, slot.numel(), bf16, 0.3)
                    return listed, tensors, _replies(*gpus)

                listed, tensors, read = update("p1", True)
                self.assertEqual(listed[0]["digest"], listed[1]["digest"])
                self.assertEqual(
                    tensors[0],
                    [
                        ["a", file, [3, 4]],
                        ["b", file, [2, 3]],
                        ["c", file, [4]],
                        ["z", file, [0, 4]],
                    ],
                )
                for gpu, reply, layout, slot in zip(gpus, read, layouts, expected):
                    self.assertNotIn("error", reply[0])
                    views = _regions(reply, layout, bf16)
                    self.assertTrue(torch.equal(torch.cat(views), slot))
                    staged = gpu.staged(slot.numel(), bf16)
                    self.assertTrue(torch.equal(staged, slot))
                # `a` was built once, at one place in the shared copy.
                self.assertEqual(
                    read[0][0]["region_offsets"][0], read[1][0]["region_offsets"][0]
                )
                self.assertEqual(
                    read[0][0]["reused_bytes"] + read[1][0]["reused_bytes"], 24
                )

                # The next update builds a new version; the layouts went once.
                _, _, read = update("p2", False)
                for gpu, reply, layout, slot in zip(gpus, read, layouts, expected):
                    self.assertIn("layout", gpu._layouts_sent)
                    views = _regions(reply, layout, bf16)
                    self.assertTrue(torch.equal(torch.cat(views), slot))

                # The process exits: every connection resets, the owner starts a
                # new one, and both GPUs connect and send their layouts again.
                gpus[0]._process.kill()
                gpus[0]._process.wait()
                _, _, read = update("p3", False)
                for reply, layout, slot in zip(read, layouts, expected):
                    self.assertNotIn("error", reply[0])
                    views = _regions(reply, layout, bf16)
                    self.assertTrue(torch.equal(torch.cat(views), slot))
            finally:
                for gpu in gpus:
                    gpu._disconnect()
                gpus[0]._process.kill()
                gpus[0]._process.wait()
                os.unlink(address)


@patch.object(LoRAPrepareClient, "_fit_staging", _unpinned_staging)
class TestPrepareTimeout(CustomTestCase):
    def test_a_timed_out_request_leaves_its_staging_behind(self):
        address = node_address(uuid.uuid4().hex[:12])
        gpu = LoRAPrepareClient(address)
        staging = os.fstat(gpu._staging_fd).st_ino
        try:
            with patch("sglang.srt.lora.lora_prepare_process._PREPARE_TIMEOUT_S", -1):
                gpu.list_tensors("p1", ["/nonexistent"])
                result, _ = gpu.poll()
            self.assertIn("TimeoutError", result["error"])
            # The process may still write the old staging; the next read gets a new one.
            self.assertNotEqual(os.fstat(gpu._staging_fd).st_ino, staging)
            self.assertIsNone(gpu._connection)
        finally:
            gpu._disconnect()
            gpu._process.kill()
            gpu._process.wait()
            os.unlink(address)


class TestListFileTensors(CustomTestCase):
    def test_an_incomplete_file_is_named_as_such(self):
        with tempfile.TemporaryDirectory() as directory:
            file = os.path.join(directory, "adapter_model.safetensors")
            save_file({"a": torch.ones(64, dtype=torch.bfloat16)}, file)
            tensors, _ = list_file_tensors([file])
            self.assertEqual(tensors, [("a", file, (64,))])
            for size in (os.path.getsize(file) - 2, 20, 4):
                with self.subTest(size=size):
                    os.truncate(file, size)
                    with self.assertRaisesRegex(ValueError, "incomplete file"):
                        list_file_tensors([file])

    def test_a_non_float_tensor_is_not_carried(self):
        with tempfile.TemporaryDirectory() as directory:
            file = os.path.join(directory, "adapter_model.safetensors")
            save_file({"a": torch.ones(4, dtype=torch.int32)}, file)
            self.assertIsNone(list_file_tensors([file]))


class TestUpdateRequestsOverIpc(CustomTestCase):
    def test_round_trip(self):
        load = _apply(upsert=True).load
        for obj in (
            _update("retract", weight_version="7"),
            PrepareLoRAAdapterReqInput(load=load, prepare_id="p1"),
            ApplyLoRAAdapterReqInput(load=load, prepare_id="p1", weight_version="7"),
            PrepareLoRAAdapterReqOutput(prepared=True, stats={"prepare_s": 0.5}),
        ):
            with self.subTest(type=type(obj).__name__):
                self.assertEqual(msgpack_decode(msgpack_encode(obj)), obj)

    def test_abort_is_not_a_pause_mode(self):
        with self.assertRaises(msgspec.ValidationError):
            msgpack_decode(msgpack_encode(_update("abort")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
