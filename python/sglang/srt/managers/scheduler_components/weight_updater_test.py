import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    CommitWeightUpdateReqInput,
    CommitWeightUpdateReqOutput,
    EndWeightUpdateReqInput,
    PrepareWeightUpdateReqInput,
    PrepareWeightUpdateReqOutput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.test_utils import CustomTestCase


class TestPreparedWeightUpdates(CustomTestCase):
    def _manager(self, *, backend="cpu", worker=None, draft_worker=None):
        self.outputs = []
        self.versions = []
        self.flushes = []
        worker = worker or Mock()
        runner = worker.model_runner
        runner.weight_updater = worker
        return SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=runner,
                weight_update_runners=lambda: [("target", runner)],
            ),
            draft_worker=draft_worker,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: self.flushes.append(kwargs) or True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.versions.append(
                    new_version
                )
            ),
            weight_stage_cpu_group=None,
            host_cpu_group=None,
            weight_update_staging=backend,
            weight_update_local_checkpoint_dir="/local" if backend == "disk" else None,
            weight_update_base_checkpoint_dir="/base",
            weight_update_base_version=0,
            send_control_output=lambda request, output: self.outputs.append(
                (request, output)
            ),
        )

    def test_cpu_destination_is_initialized_before_serving(self):
        worker = Mock()
        worker.initialize_rank_weight_stager.return_value = {"initialized": True}
        manager = self._manager(worker=worker)

        manager.initialize_weight_staging()

        worker.initialize_rank_weight_stager.assert_called_once_with(
            checkpoint_dir="/base",
            version=0,
            host_group=None,
            max_compile_group_bytes=8 << 30,
            canonical_checkpoint_dir=None,
        )

    def test_failed_cpu_destination_initialization_is_torn_down(self):
        worker = Mock()
        worker.initialize_rank_weight_stager.side_effect = RuntimeError("broken")
        manager = self._manager(worker=worker)

        with self.assertRaisesRegex(RuntimeError, "failed to initialize"):
            manager.initialize_weight_staging()

        worker.close_rank_weight_stager.assert_called_once_with()

    def test_prepare_runs_in_background_and_commit_publishes_after_copy(self):
        events = []
        worker = Mock()
        worker.stage_rank_weight_update.side_effect = lambda **kwargs: (
            events.append(("prepare", kwargs)) or {"prepared": True}
        )
        worker.validate_rank_weight_commit.side_effect = lambda version: events.append(
            ("validate", version)
        )
        worker.commit_rank_weight_update.side_effect = lambda version: (
            events.append(("commit", version)) or {"copied": True}
        )
        manager = self._manager(worker=worker)
        request = PrepareWeightUpdateReqInput(
            checkpoint_source_dir="/published",
            target_version=3,
        )

        self.assertIsNone(manager.prepare_weight_update(request))
        manager._pending_weight_preparation[1].join(timeout=2)
        manager.check_pending_weight_preparation()

        self.assertEqual(len(self.outputs), 1)
        self.assertTrue(self.outputs[0][1].success)
        self.assertEqual(manager._prepared_weight_version, 3)

        result = manager.commit_weight_update(
            CommitWeightUpdateReqInput(target_version=3)
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [event[0] for event in events], ["prepare", "validate", "commit"]
        )
        self.assertEqual(self.flushes, [{"empty_cache": False}])
        self.assertEqual(self.versions, ["3"])
        self.assertEqual(manager._served_weight_version, 3)
        self.assertIsNone(manager._prepared_weight_version)

    def test_pending_prepare_yields_scheduler_thread(self):
        manager = self._manager()
        thread = Mock()
        thread.is_alive.return_value = True
        manager._pending_weight_preparation = (Mock(), thread)

        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.time.sleep"
        ) as sleep:
            manager.check_pending_weight_preparation()

        sleep.assert_called_once_with(0)
        thread.join.assert_not_called()

    def test_prepare_requires_rank_consensus_before_starting_thread(self):
        worker = Mock()
        manager = self._manager(worker=worker)
        local = {
            "backend": "cpu",
            "pending": False,
            "prepared": None,
            "served": 0,
        }
        peer = dict(local, served=1)

        with patch.object(
            SchedulerWeightUpdaterManager,
            "_all_gather",
            return_value=[local, peer],
        ):
            result = manager.prepare_weight_update(
                PrepareWeightUpdateReqInput(
                    checkpoint_source_dir="/published",
                    target_version=2,
                )
            )

        self.assertFalse(result.success)
        self.assertIn("differs across ranks", result.message)
        worker.stage_rank_weight_update.assert_not_called()

    def test_commit_can_skip_cache_flush(self):
        worker = Mock()
        worker.commit_rank_weight_update.return_value = {"copied": True}
        manager = self._manager(worker=worker)
        manager._prepared_weight_version = 1

        result = manager.commit_weight_update(
            CommitWeightUpdateReqInput(target_version=1, flush_cache=False)
        )

        self.assertTrue(result.success)
        self.assertEqual(self.flushes, [])
        worker.commit_rank_weight_update.assert_called_once_with(1)
        self.assertEqual(self.versions, ["1"])

    def test_newer_prepare_supersedes_inactive_target(self):
        worker = Mock()
        worker.stage_rank_weight_update.return_value = {"prepared": True}
        manager = self._manager(worker=worker)
        manager._prepared_weight_version = 2
        request = PrepareWeightUpdateReqInput(
            checkpoint_source_dir="/published",
            target_version=5,
        )

        self.assertIsNone(manager.prepare_weight_update(request))
        manager._pending_weight_preparation[1].join(timeout=2)
        manager.check_pending_weight_preparation()

        worker.stage_rank_weight_update.assert_called_once_with(
            checkpoint_source_dir="/published",
            target_version=5,
        )
        self.assertEqual(manager._prepared_weight_version, 5)
        self.assertTrue(self.outputs[0][1].success)

    def test_repeated_prepare_of_inactive_target_is_idempotent(self):
        worker = Mock()
        manager = self._manager(worker=worker)
        manager._prepared_weight_version = 5

        result = manager.prepare_weight_update(
            PrepareWeightUpdateReqInput(
                checkpoint_source_dir="/published",
                target_version=5,
            )
        )

        self.assertTrue(result.success)
        worker.stage_rank_weight_update.assert_not_called()

    @patch("sglang.srt.weight_sync.disk_checkpoint.materialize")
    def test_newer_disk_prepare_advances_inactive_checkpoint(self, materialize):
        materialize.return_value = {"prepared": True}
        worker = Mock()
        manager = self._manager(backend="disk", worker=worker)
        manager._prepared_weight_version = 2
        request = PrepareWeightUpdateReqInput(
            checkpoint_source_dir="/published",
            target_version=5,
        )

        self.assertIsNone(manager.prepare_weight_update(request))
        manager._pending_weight_preparation[1].join(timeout=2)
        manager.check_pending_weight_preparation()

        materialize.assert_called_once_with(
            local_checkpoint_dir="/local",
            base_checkpoint_dir="/base",
            checkpoint_source_dir="/published",
            target_version=5,
            base_version=0,
        )
        self.assertEqual(manager._prepared_weight_version, 5)
        self.assertTrue(self.outputs[0][1].success)

    def test_invalid_superseding_prepare_retains_inactive_target(self):
        worker = Mock()
        manager = self._manager(worker=worker)
        manager._prepared_weight_version = 5

        result = manager.prepare_weight_update(
            PrepareWeightUpdateReqInput(
                checkpoint_source_dir="",
                target_version=6,
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(manager._prepared_weight_version, 5)

    def test_prepare_rejects_target_before_inactive_target(self):
        worker = Mock()
        manager = self._manager(worker=worker)
        manager._prepared_weight_version = 5

        result = manager.prepare_weight_update(
            PrepareWeightUpdateReqInput(
                checkpoint_source_dir="/published",
                target_version=3,
            )
        )

        self.assertFalse(result.success)
        self.assertIn("must not precede prepared version 5", result.message)
        worker.stage_rank_weight_update.assert_not_called()

    def test_commit_requires_collective_preflight_before_mutation(self):
        worker = Mock()
        manager = self._manager(worker=worker)
        manager._prepared_weight_version = 2
        with patch.object(
            SchedulerWeightUpdaterManager,
            "_all_gather",
            side_effect=[
                [False, False],
                [None, "peer has no prepared target"],
            ],
        ):
            result = manager.commit_weight_update(
                CommitWeightUpdateReqInput(target_version=2)
            )

        self.assertFalse(result.success)
        self.assertIn("peer has no prepared target", result.message)
        worker.commit_rank_weight_update.assert_not_called()
        self.assertEqual(self.versions, [])

    def test_partial_live_commit_is_fatal(self):
        worker = Mock()
        worker.commit_rank_weight_update.return_value = {"copied": True}
        manager = self._manager(worker=worker)
        manager._prepared_weight_version = 1

        def gather(value, group):
            if isinstance(value, bool):
                return [value]
            if value is None:
                return [None]
            if isinstance(value, tuple):
                return [value, ("peer copy failed", value[1])]
            raise AssertionError(value)

        with (
            patch.object(
                SchedulerWeightUpdaterManager,
                "_all_gather",
                side_effect=gather,
            ),
            self.assertRaisesRegex(RuntimeError, "terminating the engine"),
        ):
            manager.commit_weight_update(CommitWeightUpdateReqInput(target_version=1))

        self.assertEqual(self.versions, [])
        self.assertEqual(manager._served_weight_version, 0)

    def test_disk_commit_updates_only_the_target_model(self):
        worker = Mock()
        worker.model_runner.model = torch.nn.Linear(1, 1, bias=False)
        worker.model_runner.load_config.load_format = "safetensors"
        worker.update_weights_from_disk.return_value = (True, "ok")
        draft_worker = Mock()
        manager = self._manager(
            backend="disk",
            worker=worker,
            draft_worker=draft_worker,
        )
        manager._prepared_weight_version = 4

        result = manager.commit_weight_update(
            CommitWeightUpdateReqInput(target_version=4)
        )

        self.assertTrue(result.success)
        worker.update_weights_from_disk.assert_called_once_with(
            "/local", "safetensors", recapture_cuda_graph=False
        )
        draft_worker.weight_update_runners.assert_not_called()
        self.assertEqual(self.versions, ["4"])

    def test_disk_commit_fails_closed_if_live_tensor_layout_changes(self):
        worker = Mock()
        model = torch.nn.Linear(1, 1, bias=False)
        worker.model_runner.model = model
        worker.model_runner.load_config.load_format = "safetensors"

        def replace_weight(*args, **kwargs):
            model.weight = torch.nn.Parameter(torch.ones_like(model.weight))
            return True, "ok"

        worker.update_weights_from_disk.side_effect = replace_weight
        manager = self._manager(backend="disk", worker=worker)
        manager._prepared_weight_version = 1

        with self.assertRaisesRegex(RuntimeError, "changed live weight tensor layouts"):
            manager.commit_weight_update(CommitWeightUpdateReqInput(target_version=1))

        self.assertEqual(self.versions, [])
        self.assertEqual(manager._served_weight_version, 0)

    def test_direct_weight_mutation_is_rejected_when_staging_is_enabled(self):
        worker = Mock()
        manager = self._manager(worker=worker)

        result = manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path="/other")
        )

        self.assertFalse(result.success)
        self.assertIn("prepare_weight_update", result.message)
        worker.update_weights_from_disk.assert_not_called()

    def test_live_update_sessions_are_rejected_during_staging(self):
        for backend in ("cpu", "disk"):
            with self.subTest(backend=backend):
                worker = Mock()
                manager = self._manager(backend=backend, worker=worker)

                begin = manager.begin_weight_update(BeginWeightUpdateReqInput())
                end = manager.end_weight_update(EndWeightUpdateReqInput())

                self.assertFalse(begin.success)
                self.assertFalse(end.success)
                self.assertIsNone(manager._session)
                worker.begin_weight_update.assert_not_called()
                worker.end_weight_update.assert_not_called()

    def test_disabled_staging_adds_no_collective_to_direct_updates(self):
        worker = Mock()
        worker.update_weights_from_disk.return_value = (True, "ok")
        manager = self._manager(backend=None, worker=worker)
        request = UpdateWeightFromDiskReqInput(
            model_path="/other",
            weight_version="next",
        )

        with patch.object(
            SchedulerWeightUpdaterManager,
            "_all_gather",
            side_effect=AssertionError("unexpected collective"),
        ):
            result = manager.update_weights_from_disk(request)

        self.assertTrue(result.success)
        worker.update_weights_from_disk.assert_called_once_with(
            request.model_path,
            request.load_format,
            recapture_cuda_graph=request.recapture_cuda_graph,
        )
        self.assertEqual(self.versions, ["next"])


class _AsyncContext:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    async def __aenter__(self):
        self.events.append(f"enter_{self.name}")

    async def __aexit__(self, exc_type, exc, traceback):
        self.events.append(f"exit_{self.name}")


class TestPreparedWeightUpdateFrontend(CustomTestCase):
    def test_prepare_does_not_take_the_model_update_lock(self):
        async def scenario():
            events = []

            async def communicate(request):
                events.append("prepare")
                return [
                    PrepareWeightUpdateReqOutput(
                        success=True,
                        message="ok",
                    )
                ]

            manager = SimpleNamespace(
                auto_create_handle_loop=lambda: None,
                prepare_weight_update_communicator=communicate,
            )
            result = await TokenizerControlMixin.prepare_weight_update(
                manager,
                PrepareWeightUpdateReqInput(
                    checkpoint_source_dir="/published",
                    target_version=1,
                ),
            )
            self.assertTrue(result.success)
            self.assertEqual(events, ["prepare"])

        asyncio.run(scenario())

    def test_commit_serializes_with_inference_and_then_publishes_version(self):
        async def scenario():
            events = []

            async def communicate(request):
                events.append("commit")
                return [
                    CommitWeightUpdateReqOutput(
                        success=True,
                        message="ok",
                    )
                ]

            manager = SimpleNamespace(
                auto_create_handle_loop=lambda: None,
                abort_request=lambda **kwargs: events.append("abort"),
                is_pause_cond=_AsyncContext(events, "pause_condition"),
                is_pause=False,
                model_update_lock=SimpleNamespace(
                    writer_lock=_AsyncContext(events, "writer")
                ),
                commit_weight_update_communicator=communicate,
                mm_processor=None,
                _update_weight_version_if_provided=lambda version: events.append(
                    f"version_{version}"
                ),
            )
            result = await TokenizerControlMixin.commit_weight_update(
                manager,
                CommitWeightUpdateReqInput(target_version=7),
            )
            self.assertTrue(result.success)
            self.assertEqual(
                events,
                [
                    "enter_pause_condition",
                    "exit_pause_condition",
                    "enter_writer",
                    "commit",
                    "exit_writer",
                    "version_7",
                ],
            )

        asyncio.run(scenario())


class TestStagedMemoryRejection(unittest.TestCase):
    def _check_rejected_operation(self, operation, request_type, initial_tags):
        for tags in (None, ["kv_cache", "weights", "cuda_graph"]):
            for pending in (False, True):
                with self.subTest(tags=tags, pending=pending):
                    memory_saver = Mock()
                    flush = Mock(return_value=True)
                    manager = SchedulerWeightUpdaterManager(
                        tp_worker=None,
                        draft_worker=None,
                        tp_cpu_group=None,
                        memory_saver_adapter=memory_saver,
                        flush_cache=flush,
                        is_fully_idle=lambda **kwargs: True,
                        weight_update_staging="cpu",
                        offload_tags=set(initial_tags),
                    )
                    message = (
                        "A background weight preparation is still running."
                        if pending
                        else None
                    )
                    with (
                        patch.object(
                            SchedulerWeightUpdaterManager,
                            "_pending_preparation_message",
                            return_value=message,
                        ),
                        self.assertRaisesRegex(RuntimeError, "preparation|staged"),
                    ):
                        getattr(manager, operation)(request_type(tags=tags))

                    self.assertEqual(manager.offload_tags, set(initial_tags))
                    memory_saver.pause.assert_not_called()
                    memory_saver.resume.assert_not_called()
                    flush.assert_not_called()

    def test_rejected_weight_release_preserves_memory_state(self):
        self._check_rejected_operation(
            "release_memory_occupation", ReleaseMemoryOccupationReqInput, set()
        )

    def test_rejected_weight_resume_preserves_memory_state(self):
        self._check_rejected_operation(
            "resume_memory_occupation",
            ResumeMemoryOccupationReqInput,
            {"kv_cache", "weights", "cuda_graph"},
        )


class TestWeightCacheMemoryRejection(unittest.TestCase):
    def _check_rejected_operation(self, operation, request_type, initial_tags):
        for tags in (None, ["kv_cache", "weights", "cuda_graph"]):
            for mode in ("client", "daemon"):
                with self.subTest(tags=tags, mode=mode):
                    memory_saver = Mock()
                    flush = Mock(return_value=True)
                    manager = SchedulerWeightUpdaterManager(
                        tp_worker=None,
                        draft_worker=None,
                        tp_cpu_group=None,
                        memory_saver_adapter=memory_saver,
                        flush_cache=flush,
                        is_fully_idle=lambda **kwargs: True,
                        weight_update_staging=None,
                        offload_tags=set(initial_tags),
                    )
                    with (
                        patch(
                            "sglang.srt.managers.scheduler_components.weight_updater.get_model",
                            return_value=SimpleNamespace(weight_cache_mode=mode),
                        ),
                        self.assertRaisesRegex(RuntimeError, "weight cache is active"),
                    ):
                        getattr(manager, operation)(request_type(tags=tags))

                    self.assertEqual(manager.offload_tags, set(initial_tags))
                    memory_saver.pause.assert_not_called()
                    memory_saver.resume.assert_not_called()
                    flush.assert_not_called()

    def test_rejected_weight_release_preserves_memory_state(self):
        self._check_rejected_operation(
            "release_memory_occupation", ReleaseMemoryOccupationReqInput, set()
        )

    def test_rejected_weight_resume_preserves_memory_state(self):
        self._check_rejected_operation(
            "resume_memory_occupation",
            ResumeMemoryOccupationReqInput,
            {"kv_cache", "weights", "cuda_graph"},
        )


if __name__ == "__main__":
    unittest.main()
