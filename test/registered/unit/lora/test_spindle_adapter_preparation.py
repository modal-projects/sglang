"""Exercise the installed SGLang loading patch without allocating a model."""

import os
import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_parallel


def test_expert_local_loading_and_collective_async_install():
    config = LoRAConfig(
        config_dict={
            "peft_type": "LORA",
            "r": 2,
            "lora_alpha": 2,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
        }
    )
    hf = SimpleNamespace(num_hidden_layers=1)
    weights = {}
    for expert in range(8):
        for module in ("gate_proj", "up_proj", "down_proj"):
            for matrix, shape in (("A", (2, 4)), ("B", (3, 2))):
                weights[
                    f"base_model.model.layers.0.mlp.experts.{expert}.{module}.lora_{matrix}.weight"
                ] = torch.full(shape, float(expert))
    for matrix, shape in (("A", (2, 4)), ("B", (3, 2))):
        weights[
            f"base_model.model.layers.0.mlp.shared_experts.down_proj.lora_{matrix}.weight"
        ] = torch.ones(shape)
    full = LoRAAdapter("full", config, hf, LoadConfig(), None)
    full.initialize_weights_from_tensors(weights)
    with tempfile.TemporaryDirectory() as root:
        save_file(weights, str(Path(root) / "adapter_model.safetensors"))
        config.path = root
        for start in range(0, 8, 2):
            local = LoRAAdapter(
                "local", config, hf, LoadConfig(), None, expert_range=(start, start + 2)
            )
            local.initialize_weights()
            assert (
                str(Path(root) / "adapter_model.safetensors")
                not in Path("/proc/self/maps").read_text()
            ), "CPU adapter cache retained a file mapping"
            expected = {
                k: v for k, v in full.layers[0].weights.items() if local._owns_expert(k)
            }
            assert local.layers[0].weights.keys() == expected.keys()
            for key, tensor in expected.items():
                torch.testing.assert_close(
                    local.layers[0].weights[key], tensor, rtol=0, atol=0
                )
            assert sum(t.numel() for t in local.layers[0].weights.values()) < sum(
                t.numel() for t in full.layers[0].weights.values()
            )
    print(
        "PASS: cached adapters release file mappings and allow volume refresh",
        flush=True,
    )
    print(
        "PASS: all EP partitions preserve owned and shared weights exactly", flush=True
    )

    for fail, cancel in ((False, False), (True, False), (False, True)):
        entered, release = threading.Event(), threading.Event()
        installed, sent = [], []

        def prepare(ref):
            entered.set()
            assert release.wait(10)
            if fail:
                raise ValueError("test read failure")
            return config, object()

        manager = SimpleNamespace(
            prepare_lora_adapter=prepare,
            validate_new_adapter=lambda cfg, ref: None,
            install_prepared_lora_adapter=lambda ref, prepared: (
                installed.append(ref) or SimpleNamespace(success=True)
            ),
            create_lora_update_result=lambda success, error: SimpleNamespace(
                success=success, error=error
            ),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(lora_manager=manager),
            unload_lora_adapter=lambda req: None,
        )
        scheduler.tp_cpu_group = None
        scheduler.rust_server = None
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda output, req: sent.append(output)
            )
        )
        request = LoadLoRAAdapterReqInput(
            lora_name="test", lora_path="unused", lora_id="id"
        )
        with (
            patch.dict(os.environ, {"SPINDLE_GLM_ASYNC_LORA_LOADING": "1"}),
            get_parallel().override(pp_size=1, attn_dp_enabled=False),
        ):
            try:
                assert scheduler.load_lora_adapter(request) is None
                assert entered.wait(5)
                for _ in range(5):
                    scheduler._poll_lora_prepares()
                assert not installed and not sent
                if cancel:
                    scheduler.unload_lora_adapter(
                        UnloadLoRAAdapterReqInput(lora_name="test", lora_id="id")
                    )
                release.set()
                future = scheduler._pending_lora_prepares[0][1]
                try:
                    future.result(timeout=10)
                except ValueError:
                    assert fail
                scheduler._poll_lora_prepares()
                assert sent[0].success == (not fail and not cancel)
                assert len(installed) == int(not fail and not cancel)
                assert not scheduler._pending_lora_prepares
            finally:
                release.set()
                scheduler._lora_prepare_executor.shutdown(wait=True)
    print(
        "PASS: background preparation, failure and unload preserve registration ordering",
        flush=True,
    )

    future = Future()
    future.set_result((config, object()))
    scheduler._pending_lora_prepares.append((request, future, 0))
    sent.clear()
    installed.clear()
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch(
            "torch.distributed.all_reduce",
            side_effect=lambda value, **kwargs: value.zero_(),
        ),
    ):
        scheduler._poll_lora_prepares()
    assert not installed and not sent
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.all_reduce"),
        patch("torch.distributed.get_world_size", return_value=2),
        patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda values, value, **kwargs: values.__setitem__(
                slice(None), [None, "peer failed"]
            ),
        ),
    ):
        scheduler._poll_lora_prepares()
    assert not installed and not sent[0].success
    print(
        "PASS: every TP rank must finish successfully before installation", flush=True
    )
