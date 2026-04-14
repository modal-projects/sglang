from __future__ import annotations

import json
import multiprocessing
import os
import unittest
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from safetensors.torch import load_file

from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.test_utils import CustomTestCase

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_LOADER = "sglang.srt.model_loader.lora_merge_loader.merge_lora_tensors_inplace"

BASE_MODEL = os.getenv("QWEN35_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
BASE_URL = os.getenv("QWEN35_LIVE_UPDATE_BASE_URL", "http://127.0.0.1:30000")
ADAPTER_DIR = Path(os.getenv("QWEN35_LORA_DIR", str(REPO_ROOT)))
ADAPTER_CONFIG_PATH = Path(
    os.getenv("QWEN35_LORA_CONFIG", str(ADAPTER_DIR / "adapter_config.json"))
)
ADAPTER_WEIGHTS_PATH = Path(
    os.getenv(
        "QWEN35_LORA_WEIGHTS",
        str(ADAPTER_DIR / "sampler_weights_init.safetensors"),
    )
)

WEIGHT_VERSION = os.getenv(
    "QWEN35_LIVE_UPDATE_WEIGHT_VERSION", "qwen35-merged-lora-live-update"
)
ATOMIC_PAUSE_MODE = os.getenv("QWEN35_LIVE_UPDATE_ATOMIC_PAUSE_MODE", "in_place")
FLUSH_CACHE = os.getenv("QWEN35_LIVE_UPDATE_FLUSH_CACHE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MEM_FRACTION_STATIC = os.getenv("QWEN35_LIVE_UPDATE_MEM_FRACTION_STATIC", "0.85")
CHUNKED_PREFILL_SIZE = os.getenv("QWEN35_LIVE_UPDATE_CHUNKED_PREFILL_SIZE", "8192")
MAX_PREFILL_TOKENS = os.getenv("QWEN35_LIVE_UPDATE_MAX_PREFILL_TOKENS", "8192")
PAGE_SIZE = os.getenv("QWEN35_LIVE_UPDATE_PAGE_SIZE", "64")
MAX_RUNNING_REQUESTS = os.getenv("QWEN35_LIVE_UPDATE_MAX_RUNNING_REQUESTS", "32")
CUDA_GRAPH_MAX_BS = os.getenv("QWEN35_LIVE_UPDATE_CUDA_GRAPH_MAX_BS", "32")
KV_CACHE_DTYPE = os.getenv("QWEN35_LIVE_UPDATE_KV_CACHE_DTYPE", "bf16")
MAMBA_SCHEDULER_STRATEGY = os.getenv(
    "QWEN35_LIVE_UPDATE_MAMBA_SCHEDULER_STRATEGY", "extra_buffer"
)
MAMBA_SSM_DTYPE = os.getenv("QWEN35_LIVE_UPDATE_MAMBA_SSM_DTYPE", "bfloat16")
PREFILL_ATTENTION_BACKEND = os.getenv(
    "QWEN35_PREFILL_ATTENTION_BACKEND", "trtllm_mha"
)
DECODE_ATTENTION_BACKEND = os.getenv(
    "QWEN35_DECODE_ATTENTION_BACKEND", "trtllm_mha"
)
MOE_RUNNER_BACKEND = os.getenv("QWEN35_MOE_RUNNER_BACKEND", "flashinfer_trtllm")


def _chat_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": BASE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _load_adapter_assets() -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    with open(ADAPTER_CONFIG_PATH, "r") as f:
        adapter_config = json.load(f)
    adapter_tensors = list(load_file(str(ADAPTER_WEIGHTS_PATH)).items())
    return adapter_config, adapter_tensors


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    response = requests.request(method, url, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(
            f"HTTP request failed: method={method} url={url} "
            f"status={response.status_code} body={response.text}"
        )
    return response.json()


def _server_args() -> list[str]:
    parsed = urlparse(BASE_URL)
    return {
        "model_path": BASE_MODEL,
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 30000,
        "served_model_name": BASE_MODEL,
        "enable_multimodal": True,
        "reasoning_parser": "qwen3",
        "mem_fraction_static": float(MEM_FRACTION_STATIC),
        "chunked_prefill_size": int(CHUNKED_PREFILL_SIZE),
        "max_prefill_tokens": int(MAX_PREFILL_TOKENS),
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "prefill_attention_backend": PREFILL_ATTENTION_BACKEND,
        "decode_attention_backend": DECODE_ATTENTION_BACKEND,
        "page_size": int(PAGE_SIZE),
        "moe_runner_backend": MOE_RUNNER_BACKEND,
        "mamba_scheduler_strategy": MAMBA_SCHEDULER_STRATEGY,
        "mamba_ssm_dtype": MAMBA_SSM_DTYPE,
        "cuda_graph_bs": list(range(1, int(CUDA_GRAPH_MAX_BS) + 1)),
        "cuda_graph_max_bs": int(CUDA_GRAPH_MAX_BS),
        "max_running_requests": int(MAX_RUNNING_REQUESTS),
        "custom_weight_loader": [MERGE_LOADER],
        "weight_version": "baseline",
        "device": "cuda",
    }


class TestQwen35MergedLoRALiveUpdate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not ADAPTER_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing adapter config: {ADAPTER_CONFIG_PATH}")
        if not ADAPTER_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Missing adapter weights: {ADAPTER_WEIGHTS_PATH}")
        cls.http_engine = HttpServerEngineAdapter(**_server_args())
        cls.base_url = cls.http_engine.server_args.url()
        cls.model = BASE_MODEL

    @classmethod
    def tearDownClass(cls):
        cls.http_engine.shutdown()

    def _server_log_summary(self) -> str:
        return (
            f"\nserver_alive={self.http_engine.process.is_alive()}"
            f"\nbase_url={self.base_url}"
        )

    def _get_model_info(self) -> dict[str, Any]:
        return _request_json(
            "get", f"{self.base_url}/get_model_info", timeout=60
        )

    def _chat(self, prompt: str) -> dict[str, Any]:
        try:
            return _request_json(
                "post",
                f"{self.base_url}/v1/chat/completions",
                payload=_chat_payload(prompt),
                timeout=600,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Chat request failed for prompt={prompt!r}: {exc}"
                f"{self._server_log_summary()}"
            ) from exc

    def _merge_adapter(self) -> dict[str, Any]:
        adapter_config, adapter_tensors = _load_adapter_assets()
        payload = {
            "serialized_named_tensors": [
                MultiprocessingSerializer.serialize(adapter_tensors, output_str=True)
            ],
            "manifest": {"adapter_config": adapter_config},
            "load_format": MERGE_LOADER,
            "flush_cache": FLUSH_CACHE,
            "atomic_pause_mode": ATOMIC_PAUSE_MODE,
            "weight_version": WEIGHT_VERSION,
        }
        try:
            return _request_json(
                "post",
                f"{self.base_url}/update_weights_from_tensor",
                payload=payload,
                timeout=1800,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Merged LoRA update request failed: {exc}"
                f"{self._server_log_summary()}"
            ) from exc

    def test_qwen35_prod_backends_survive_merged_lora_live_update(self):
        initial_info = self._get_model_info()
        self.assertEqual(initial_info["weight_epoch"], 0)
        self.assertEqual(initial_info["weight_version"], "baseline")

        pre_merge = self._chat("Reply with exactly BASELINE.")
        self.assertEqual(pre_merge["metadata"]["weight_epoch_start"], 0)
        self.assertEqual(pre_merge["metadata"]["weight_epoch_end"], 0)

        merge_result = self._merge_adapter()
        self.assertTrue(merge_result["success"], merge_result)

        updated_info = self._get_model_info()
        self.assertEqual(updated_info["weight_epoch"], 1)
        self.assertEqual(updated_info["weight_version"], WEIGHT_VERSION)

        post_merge = self._chat("Reply with exactly OK.")
        metadata = post_merge["metadata"]
        self.assertEqual(metadata["weight_version"], WEIGHT_VERSION)
        self.assertEqual(metadata["weight_version_start"], WEIGHT_VERSION)
        self.assertEqual(metadata["weight_version_end"], WEIGHT_VERSION)
        self.assertEqual(metadata["weight_epoch_start"], 1)
        self.assertEqual(metadata["weight_epoch_end"], 1)
        self.assertFalse(metadata["mixed_weight_epochs"])
        self.assertFalse(metadata["resume_from_stale_kv"])

        completion = post_merge["choices"][0]["message"]["content"]
        self.assertTrue(completion)

        print(
            json.dumps(
                {
                    "merge_result": merge_result,
                    "initial_info": initial_info,
                    "updated_info": updated_info,
                    "post_merge_metadata": metadata,
                    "post_merge_completion": completion,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
