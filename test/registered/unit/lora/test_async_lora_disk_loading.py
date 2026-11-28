"""Unit tests for --enable-async-lora-disk-loading CLI parsing and background preparation."""

from __future__ import annotations

import argparse
import threading
import types
import unittest
from unittest import mock

import torch
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers import scheduler as scheduler_mod
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestAsyncLoRADiskLoading(CustomTestCase):
    def test_cli_flag_defaults_false_and_parses_opt_in(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flags, expected in (
            ([], False),
            (["--enable-async-lora-disk-loading"], True),
        ):
            with self.subTest(flags=flags):
                parsed = parser.parse_args(["--model-path", "dummy", *flags])
                server_args = ServerArgs.from_cli_args(parsed)
                self.assertIs(server_args.enable_async_lora_disk_loading, expected)

        keyword_args = ServerArgs(
            model_path="dummy", enable_async_lora_disk_loading=True
        )
        self.assertTrue(keyword_args.enable_async_lora_disk_loading)

    def test_async_prepare_runs_off_scheduler_thread_and_installs_on_poll(self):
        prepare_started = threading.Event()
        release_prepare = threading.Event()
        prepare_threads: list[int] = []
        installed: list[tuple[LoRARef, object]] = []
        sent_outputs: list[tuple[object, object]] = []

        def fake_prepare(lora_ref: LoRARef):
            prepare_threads.append(threading.get_ident())
            prepare_started.set()
            assert release_prepare.wait(timeout=5.0)
            return ("fake_config", "fake_adapter")

        manager = types.SimpleNamespace(
            prepare_lora_adapter=fake_prepare,
            validate_new_adapter=lambda cfg, ref: None,
            install_prepared_lora_adapter=lambda ref, prepared: (
                installed.append((ref, prepared))
                or types.SimpleNamespace(success=True, error_message="")
            ),
            create_lora_update_result=lambda ok, msg="": types.SimpleNamespace(
                success=ok, error_message=msg
            ),
        )

        sched = object.__new__(Scheduler)
        sched.server_args = types.SimpleNamespace(enable_async_lora_disk_loading=True)
        sched.tp_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(lora_manager=manager),
            load_lora_adapter=mock.MagicMock(),
        )
        sched.send_control_output = lambda out, req: sent_outputs.append((out, req))

        req = LoadLoRAAdapterReqInput(
            lora_name="adapter_a",
            lora_path="/tmp/fake-adapter",
            lora_id="lora-1",
        )
        parallel = types.SimpleNamespace(pp_size=1, attn_dp_enabled=False)

        try:
            with (
                mock.patch.dict("os.environ", {}, clear=False),
                mock.patch.object(
                    scheduler_mod, "get_parallel", return_value=parallel
                ),
            ):
                ret = sched.load_lora_adapter(req)
                self.assertIsNone(ret)
                sched.tp_worker.load_lora_adapter.assert_not_called()
                self.assertTrue(prepare_started.wait(timeout=5.0))
                self.assertNotEqual(prepare_threads[0], threading.get_ident())

                # Polling before completion must not block or install yet.
                sched._poll_lora_prepares()
                self.assertEqual(installed, [])
                self.assertEqual(sent_outputs, [])

                release_prepare.set()
                sched._pending_lora_prepares[0][1].result(timeout=5.0)
                sched._poll_lora_prepares()

            self.assertEqual(len(installed), 1)
            self.assertEqual(installed[0][0].lora_id, "lora-1")
            self.assertEqual(len(sent_outputs), 1)
            self.assertTrue(sent_outputs[0][0].success)
            self.assertIs(sent_outputs[0][1], req)
        finally:
            release_prepare.set()
            if hasattr(sched, "_lora_prepare_executor"):
                sched._lora_prepare_executor.shutdown(wait=True)

    def test_unload_cancels_inflight_async_prepare(self):
        prepare_started = threading.Event()
        release_prepare = threading.Event()
        sent_outputs: list[tuple[object, object]] = []
        installed: list[object] = []

        def fake_prepare(lora_ref: LoRARef):
            prepare_started.set()
            assert release_prepare.wait(timeout=5.0)
            return ("fake_config", "fake_adapter")

        manager = types.SimpleNamespace(
            prepare_lora_adapter=fake_prepare,
            validate_new_adapter=lambda cfg, ref: None,
            install_prepared_lora_adapter=lambda ref, prepared: installed.append(ref),
            create_lora_update_result=lambda ok, msg="": types.SimpleNamespace(
                success=ok, error_message=msg
            ),
        )

        sched = object.__new__(Scheduler)
        sched.server_args = types.SimpleNamespace(enable_async_lora_disk_loading=True)
        sched.tp_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(lora_manager=manager),
            unload_lora_adapter=lambda req: types.SimpleNamespace(success=True),
        )
        sched.send_control_output = lambda out, req: sent_outputs.append((out, req))

        load_req = LoadLoRAAdapterReqInput(
            lora_name="adapter_cancel",
            lora_path="/tmp/fake-adapter",
            lora_id="lora-cancel",
        )
        unload_req = UnloadLoRAAdapterReqInput(
            lora_name="adapter_cancel", lora_id="lora-cancel"
        )
        parallel = types.SimpleNamespace(pp_size=1, attn_dp_enabled=False)

        try:
            with mock.patch.object(
                scheduler_mod, "get_parallel", return_value=parallel
            ):
                self.assertIsNone(sched.load_lora_adapter(load_req))
                self.assertTrue(prepare_started.wait(timeout=5.0))
                sched.unload_lora_adapter(unload_req)
                release_prepare.set()
                sched._pending_lora_prepares[0][1].result(timeout=5.0)
                sched._poll_lora_prepares()

            self.assertEqual(installed, [])
            self.assertEqual(len(sent_outputs), 1)
            self.assertFalse(sent_outputs[0][0].success)
            self.assertIn("unloaded during preparation", sent_outputs[0][0].error_message)
        finally:
            release_prepare.set()
            if hasattr(sched, "_lora_prepare_executor"):
                sched._lora_prepare_executor.shutdown(wait=True)

    def test_sync_fallback_when_flag_disabled_or_unsupported_parallelism(self):
        cases = [
            (False, 1, False),
            (True, 2, False),
            (True, 1, True),
        ]
        for enabled, pp_size, attn_dp_enabled in cases:
            with self.subTest(
                enabled=enabled, pp_size=pp_size, attn_dp_enabled=attn_dp_enabled
            ):
                sched = object.__new__(Scheduler)
                sched.server_args = types.SimpleNamespace(
                    enable_async_lora_disk_loading=enabled
                )
                sync_result = types.SimpleNamespace(success=True)
                sched.tp_worker = types.SimpleNamespace(
                    load_lora_adapter=mock.MagicMock(return_value=sync_result)
                )
                req = LoadLoRAAdapterReqInput(
                    lora_name="adapter_sync",
                    lora_path="/tmp/fake-adapter",
                    lora_id="lora-sync",
                )
                parallel = types.SimpleNamespace(
                    pp_size=pp_size, attn_dp_enabled=attn_dp_enabled
                )
                with (
                    mock.patch.dict("os.environ", {}, clear=True),
                    mock.patch.object(
                        scheduler_mod, "get_parallel", return_value=parallel
                    ),
                ):
                    result = sched.load_lora_adapter(req)
                self.assertIs(result, sync_result)
                sched.tp_worker.load_lora_adapter.assert_called_once_with(req)

    def test_install_prepared_lora_adapter_deduplicates_shared_tensor_cpu_bytes(self):
        mgr = object.__new__(LoRAManager)
        mgr.validate_new_adapter = lambda cfg, ref: None
        mgr.speculative_algorithm = None
        mgr.configs = {}
        mgr.loras = {}
        mgr.lora_refs = {}
        mgr.num_pinned_loras = 0
        mgr.create_lora_update_result = lambda ok, msg="": types.SimpleNamespace(
            success=ok, error_message=msg
        )

        shared = torch.zeros((4, 8), dtype=torch.float32)  # 128 bytes
        unique = torch.zeros((2, 4), dtype=torch.float32)  # 32 bytes
        adapter = types.SimpleNamespace(
            embedding_layers={},
            layers=[
                types.SimpleNamespace(weights={"a": shared, "b": unique}),
                types.SimpleNamespace(weights={"a": shared}),
            ],
        )
        ref = LoRARef(
            lora_id="shared-id", lora_name="shared", lora_path="/tmp/p", pinned=False
        )

        with mock.patch("sglang.srt.lora.lora_manager.logger") as mock_logger:
            result = mgr.install_prepared_lora_adapter(ref, ("cfg", adapter))

        self.assertTrue(result.success)
        mock_logger.info.assert_called_once_with(
            "LoRA prepared id=%s cpu_bytes=%d", "shared-id", 160
        )


if __name__ == "__main__":
    unittest.main()
