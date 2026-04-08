from __future__ import annotations

import asyncio
import json
import os
import pathlib
from types import SimpleNamespace
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

APP_NAME = "sglang-weight-update-review-fixes"
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0",
)
GPU = os.getenv("SGLANG_MODAL_GPU", "A10G")

RUNTIME_CONFIG_SECRET = modal.Secret.from_dict(
    {
        "SGLANG_MODAL_IMAGE_TAG": SGLANG_IMAGE_TAG,
        "SGLANG_MODAL_GPU": GPU,
    }
)

SOURCE_DIRS = [
    (
        REPO_ROOT / "python/sglang",
        "/sgl-workspace/sglang/python/sglang",
    ),
]

app = modal.App(name=APP_NAME)
image = modal.Image.from_registry(SGLANG_IMAGE_TAG)
if modal.is_local():
    for local_path, remote_path in SOURCE_DIRS:
        image = image.add_local_dir(local_path, remote_path, copy=False)


class _AwaitableNone:
    def __await__(self):
        if False:
            yield None
        return None


class _RecordingSender:
    def __init__(self):
        self.sent: list[Any] = []

    def send_pyobj(self, obj: Any):
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


class _LoopRunner:
    def run_until_complete(self, coro):
        return asyncio.run(coro)


def _py_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/sgl-workspace/sglang/python:/sgl-workspace/sglang"
    return env


@app.function(image=image, gpu=GPU, timeout=30 * 60, secrets=[RUNTIME_CONFIG_SECRET])
def run_review_fix_validation() -> dict[str, Any]:
    os.environ["PYTHONPATH"] = "/sgl-workspace/sglang/python:/sgl-workspace/sglang"

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import (
        ContinueGenerationReqInput,
        PauseGenerationReqInput,
        UpdateWeightFromDiskReqInput,
        UpdateWeightFromDiskReqOutput,
    )
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    def _new_fake_tokenizer_manager() -> TokenizerManager:
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.auto_create_handle_loop = lambda: None
        tm.send_to_scheduler = _RecordingSender()
        tm.model_update_lock = _FakeModelUpdateLock()
        tm.weight_epoch_reservation_lock = asyncio.Lock()
        tm.weight_update_orchestration_lock = asyncio.Lock()
        tm.is_pause = False
        tm.is_pause_cond = asyncio.Condition()
        tm.current_weight_epoch = 0
        tm.next_weight_epoch = 1
        tm.weight_version_by_epoch = {0: "v0"}
        tm.server_args = SimpleNamespace(
            load_format="auto",
            dp_size=1,
            enable_dp_attention=False,
            weight_version="v0",
            model_path="initial-model",
        )
        tm.served_model_name = "initial-model"
        tm.model_path = "initial-model"
        tm.abort_request = lambda *args, **kwargs: None
        return tm

    async def _wait_for_update_send(
        sender: _RecordingSender, expected_count: int, timeout_s: float = 1.0
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            sent_count = sum(
                isinstance(obj, UpdateWeightFromDiskReqInput) for obj in sender.sent
            )
            if sent_count >= expected_count:
                return
            await asyncio.sleep(0)
        raise TimeoutError(
            f"Timed out waiting for {expected_count} update requests; "
            f"have {sent_count}, sent={[type(obj).__name__ for obj in sender.sent]}"
        )

    async def _run_serialization_probe() -> dict[str, Any]:
        tm = _new_fake_tokenizer_manager()
        atomic_obj = UpdateWeightFromDiskReqInput(
            model_path="atomic-model",
            atomic_pause_mode="retract",
            flush_cache=True,
            weight_version="atomic-v1",
        )
        plain_obj = UpdateWeightFromDiskReqInput(
            model_path="plain-model",
            flush_cache=True,
            weight_version="plain-v2",
        )

        atomic_task = asyncio.create_task(tm.update_weights_from_disk(atomic_obj, None))
        await _wait_for_update_send(tm.send_to_scheduler, 1)

        plain_task = asyncio.create_task(tm.update_weights_from_disk(plain_obj, None))
        await asyncio.sleep(0)

        sent_types_before_first_reply = [
            type(obj).__name__ for obj in tm.send_to_scheduler.sent
        ]
        if sent_types_before_first_reply != [
            "PauseGenerationReqInput",
            "UpdateWeightFromDiskReqInput",
        ]:
            raise AssertionError(
                f"Unexpected send sequence before first reply: {sent_types_before_first_reply}"
            )
        if plain_task.done():
            raise AssertionError("Plain update should still be blocked before first reply.")

        tm._handle_update_weights_from_disk_req_output(
            UpdateWeightFromDiskReqOutput(
                success=True,
                message="atomic ok",
                num_paused_requests=0,
            )
        )
        atomic_result = await asyncio.wait_for(atomic_task, timeout=1.0)
        await _wait_for_update_send(tm.send_to_scheduler, 2)
        await asyncio.sleep(0)

        if not plain_task.done() and isinstance(
            tm.send_to_scheduler.sent[-1], ContinueGenerationReqInput
        ):
            await asyncio.sleep(0)

        sent_types_before_second_reply = [
            type(obj).__name__ for obj in tm.send_to_scheduler.sent
        ]
        if sent_types_before_second_reply != [
            "PauseGenerationReqInput",
            "UpdateWeightFromDiskReqInput",
            "ContinueGenerationReqInput",
            "UpdateWeightFromDiskReqInput",
        ]:
            raise AssertionError(
                f"Unexpected send sequence before second reply: {sent_types_before_second_reply}"
            )
        if plain_task.done():
            raise AssertionError("Plain update should be waiting on its own reply.")

        try:
            tm._handle_update_weights_from_disk_req_output(
                UpdateWeightFromDiskReqOutput(
                    success=True,
                    message="plain ok",
                    num_paused_requests=0,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced in result
            raise AssertionError(f"Second reply raised unexpectedly: {exc!r}") from exc

        plain_result = await asyncio.wait_for(plain_task, timeout=1.0)
        sent_types_final = [type(obj).__name__ for obj in tm.send_to_scheduler.sent]

        return {
            "sent_types": sent_types_final,
            "atomic_result": atomic_result,
            "plain_result": plain_result,
            "current_weight_epoch": tm.current_weight_epoch,
            "next_weight_epoch": tm.next_weight_epoch,
            "weight_version_by_epoch": tm.weight_version_by_epoch,
            "server_weight_version": tm.server_args.weight_version,
            "server_model_path": tm.server_args.model_path,
        }

    def _run_engine_api_probe() -> dict[str, Any]:
        default_tm = _new_fake_tokenizer_manager()

        async def _immediate_wait_default(obj):
            default_tm.seen_obj = obj
            return True, "ok", 0

        default_tm._wait_for_model_update_from_disk = _immediate_wait_default
        default_engine = SimpleNamespace(
            loop=_LoopRunner(),
            tokenizer_manager=default_tm,
        )

        default_result = Engine.update_weights_from_disk(
            default_engine,
            model_path="dummy-model",
            atomic_pause_mode="in_place",
        )

        good_tm = _new_fake_tokenizer_manager()

        async def _immediate_wait_good(obj):
            good_tm.seen_obj = obj
            return True, "ok", 0

        good_tm._wait_for_model_update_from_disk = _immediate_wait_good
        good_engine = SimpleNamespace(
            loop=_LoopRunner(),
            tokenizer_manager=good_tm,
        )

        explicit_result = Engine.update_weights_from_disk(
            good_engine,
            model_path="dummy-model",
            atomic_pause_mode="in_place",
            flush_cache=False,
        )

        return {
            "default_result": default_result,
            "explicit_result": explicit_result,
            "explicit_seen_atomic_pause_mode": good_tm.seen_obj.atomic_pause_mode,
            "explicit_seen_flush_cache": good_tm.seen_obj.flush_cache,
            "explicit_sent_types": [
                type(obj).__name__ for obj in good_tm.send_to_scheduler.sent
            ],
        }

    serialization_result = asyncio.run(_run_serialization_probe())
    engine_api_result = _run_engine_api_probe()

    return {
        "image_tag": os.getenv("SGLANG_MODAL_IMAGE_TAG", SGLANG_IMAGE_TAG),
        "gpu": os.getenv("SGLANG_MODAL_GPU", GPU),
        "serialization_probe": serialization_result,
        "engine_api_probe": engine_api_result,
    }


@app.local_entrypoint()
def main() -> None:
    with modal.enable_output():
        result = run_review_fix_validation.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
