"""Modal deployment: Kimi K3 (TP8 B300, DFlash spec-decode) serving the
anthropic-messages-claude-code branch of modal-projects/sglang.

Differences from the base ``ep-kimi-k3-1`` serving script:

1.  The base image (``modalresearch/sglang:kimi-k3-cu13-...``) carries a full
    sglang checkout at ``/sgl-workspace/sglang``; a ``run_commands`` layer
    fetches ``anthropic-messages-claude-code`` from the modal fork and checks
    out that commit's ``python/`` tree over the image's copy (house pattern
    from autoinference: ``git checkout <sha> -- python/``). The Anthropic
    Messages gaps closed by the branch (stop_sequence propagation, new
    stop_reasons, disable_parallel_tool_use, output_config.format, SSE ping
    keepalive, usage enrichment, Route contract layer: request-id / 529 /
    body caps / models negotiation / x-api-key) are therefore live on
    ``/v1/messages*`` while the base image's kernels, sgl-kernel build and
    Kimi-K3 patches stay untouched.

2.  The HF + JIT kernel-cache volumes are actually MOUNTED (they were
    declared but unmounted before): every JIT-cache env var points under
    ``JIT_CACHE_PATH``, and the HF cache dir matches ``HF_CACHE_MOUNT_PATH``,
    so first-boot compile artifacts persist across cold starts.

3.  ``SGLANG_CUTE_AOT_ABI_SALT`` additionally carries the patched commit sha
    so fork-patched kernels never read base-image AOT caches by accident.

4.  An ``/v1/messages`` (Anthropic dialect) warmup runs after the OpenAI
    chat warmup so the first real Claude Code request doesn't pay the
    request-path cold cost.

Deploy:
    .venv/bin/python -m modal deploy docs_new/deploy/kimi_k3_anthropic_modal.py
(or any env with ``modal>=1`` and modal credentials)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES
PORT = 8000

MODEL_NAME = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
MODEL_PATH = "/flash-endpoint-model/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
SPECULATIVE_DRAFT_MODEL_REPO_ID = "modal-labs/Kimi-K3-DFlash"
SPECULATIVE_DRAFT_MODEL_REVISION = "c192d15a43407bf758b5ae0880d5c72052fef1de"
LOAD_FORMAT = "fastsafetensors"
DRAFT_LOAD_FORMAT = "safetensors"

DRAFT_KV_CACHE_DTYPE = "bf16"
MEM_FRACTION_STATIC = "0.915"
PREFILL_CUDA_GRAPH_MAX_BS = "4096"
PREFILL_CUDA_GRAPH_BS = "128 256 512 768 1024 1536 2048 3072 4096"
DECODE_CUDA_GRAPH_MAX_BS = "48"

SGLANG_BASE_IMAGE = "modalresearch/sglang:kimi-k3-cu13-20260806-b9e90a6d6"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "anthropic-messages-claude-code"
SGLANG_FORK_COMMIT = "f41df9b37601c854cfeb435986a36c52d6919fe4"
SGLANG_FORK_MOUNT = "/sgl-workspace/sglang"
FLASHINFER_TARGET_VERSION = "0.6.16rc5"
FLASHINFER_EXTRA_INDEX = "https://flashinfer.ai/whl"
FLASHINFER_EXTRA_INDEX_CU130 = "https://flashinfer.ai/whl/cu130"
FASTSAFETENSORS_VERSION = "0.3.3"
AUTOINFERENCE_UTILS_VERSION = "0.2.3"

GPU = "B300:8"
TP_SIZE = 8
CPU = 16
MEMORY_MIB = 1024 * 1024

TARGET_CONCURRENCY = 6
REQUIRE_AUTHENTICATION = True
ROUTING_REGION = "us-west"

HF_CACHE_MOUNT_PATH = "/cache/huggingface"
JIT_CACHE_MOUNT_PATH = "/root/kimi-k3-jit-cache-volume"
JIT_CACHE_PATH = f"{JIT_CACHE_MOUNT_PATH}/kimi-k3-cu13-sm103-rc5-native-kepoch2-4153db87"

HF_CACHE_VOLUME_NAME = "huggingface-cache"
JIT_CACHE_VOLUME_NAME = "kimi-k3-b300-jit-cache"

hf_cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    create_if_missing=True,
)
jit_cache = modal.Volume.from_name(
    JIT_CACHE_VOLUME_NAME,
    create_if_missing=True,
)

BASE_RUNTIME_ENV = {
    "SYNC_TOKEN_IDS_ACROSS_TP": "1",
    "SGLANG_TRTLLM_GEN_MOE_EAGER_WORKSPACE_BYTES": "4294967296",
    "SGLANG_TRTLLM_GEN_MOE_MAX_TILE_N": "256",
    "SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS": "8192",
    "KIMI_K3_DRAFT_KV_CACHE_DTYPE": DRAFT_KV_CACHE_DTYPE,
    "KIMI_K3_MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
    "KIMI_K3_PREFILL_CUDA_GRAPH_MAX_BS": PREFILL_CUDA_GRAPH_MAX_BS,
    "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(TP_SIZE)),
    "HF_HOME": HF_CACHE_MOUNT_PATH,
    "HF_HUB_CACHE": HF_CACHE_MOUNT_PATH,
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "SGLANG_FASTSAFETENSORS_NOGDS": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_K3_AR_FUSION": "1",
    "SGLANG_TRTLLM_GEN_MOE_SOURCE": "flashinfer",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
    "CUTE_DSL_ARCH": "sm_103a",
    "FLASH_ATTENTION_ARCH": "sm_103",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{JIT_CACHE_PATH}/flash-attention-cute-dsl",
    "CUTE_DSL_CACHE_DIR": f"{JIT_CACHE_PATH}/cute-dsl",
    "SGLANG_CUTE_AOT_CACHE_DIR": f"{JIT_CACHE_PATH}/cute-aot",
    # Kernel-ABI salt includes the forked commit: last-writer-wins on
    # python/ must never be served stale base-image AOT artifacts.
    "SGLANG_CUTE_AOT_ABI_SALT": f"{SGLANG_BASE_IMAGE}:{SGLANG_FORK_COMMIT}",
    "TVM_FFI_CACHE_DIR": f"{JIT_CACHE_PATH}/tvm-ffi",
    "SGLANG_CACHE_DIR": f"{JIT_CACHE_PATH}/sglang",
    "SGLANG_DG_CACHE_DIR": f"{JIT_CACHE_PATH}/deep-gemm",
    "FLASHINFER_WORKSPACE_BASE": f"{JIT_CACHE_PATH}/flashinfer-workspace",
    "TRITON_CACHE_DIR": f"{JIT_CACHE_PATH}/triton",
    "TRITON_CACHE_AUTOTUNING": "1",
    "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "1",
    "FLA_CACHE_RESULTS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:False",
    "TORCHINDUCTOR_CACHE_DIR": f"{JIT_CACHE_PATH}/inductor",
    "CUDA_CACHE_PATH": f"{JIT_CACHE_PATH}/cuda",
    "SGLANG_SSE_KEEPALIVE_INTERVAL": "1",
    "SGLANG_K3_TARGET_DENSE_FP8": "wide",
    "SGLANG_K3_TARGET_DENSE_FP8_REPRESENTATION": "tensor_static",
    "SGLANG_K3_TARGET_DENSE_FP8_MEMORY_DIAGNOSTICS": "1",
    "SGLANG_K3_TARGET_DENSE_FP8_RANGE_DIAGNOSTICS": "0",
    "SGLANG_K3_ATTN_RES_FP8_FUSION": "1",
    "SGLANG_VLM_MEDIA_URL_FETCH_ENABLED": "false",
    "SGLANG_OPENAI_MEDIA_URL_FETCH_ENABLED": "false",
}

serving_image = (
    modal.Image.from_registry(SGLANG_BASE_IMAGE)
    .entrypoint([])
    # Branch patch (house pattern): fetch the fork branch and overlay its
    # python/ tree onto the image's checkout. The marker grep fails the
    # image build loudly if the overlay did not stick.
    .run_commands(
        f"cd {SGLANG_FORK_MOUNT} && "
        f"(git remote add modal-fork {SGLANG_FORK_REPO} || true) && "
        f"git fetch modal-fork {SGLANG_FORK_BRANCH} && "
        f"git checkout {SGLANG_FORK_COMMIT} -- python/ && "
        "grep -q AnthropicCacheableBlock "
        "python/sglang/srt/entrypoints/anthropic/protocol.py && "
        f"printf '%s\\n' {SGLANG_FORK_COMMIT} > /etc/sglang-branch-sha"
    )
    .uv_pip_install(
        f"autoinference-utils=={AUTOINFERENCE_UTILS_VERSION}",
        f"fastsafetensors=={FASTSAFETENSORS_VERSION}",
    )
    .uv_pip_install(
        f"flashinfer-python=={FLASHINFER_TARGET_VERSION}",
        f"flashinfer-cubin=={FLASHINFER_TARGET_VERSION}",
        f"flashinfer-jit-cache=={FLASHINFER_TARGET_VERSION}",
        extra_index_url=FLASHINFER_EXTRA_INDEX_CU130,
        extra_options=(
            f"--extra-index-url {FLASHINFER_EXTRA_INDEX} "
            "--index-strategy unsafe-best-match"
        ),
    )
    .env(BASE_RUNTIME_ENV)
)

EXTRA_SERVER_ARGS = {
    "--trust-remote-code": "",
    "--load-format": LOAD_FORMAT,
    "--dist-timeout": "3600",
    "--context-length": "1048576",
    "--moe-runner-backend": "flashinfer_mxfp4",
    "--kv-cache-dtype": "fp8_e4m3",
    "--chunked-prefill-size": "16384",
    "--page-size": "64",
    "--mem-fraction-static": MEM_FRACTION_STATIC,
    "--schedule-policy": "lpm",
    "--attention-backend": "trtllm_mla",
    "--prefill-attention-backend": "trtllm_mla",
    "--decode-attention-backend": "cutedsl_mla",
    "--mamba-ssm-dtype": "bfloat16",
    "--linear-attn-prefill-backend": "triton",
    "--linear-attn-decode-backend": "flashinfer",
    "--linear-attn-verify-backend": "nv_cutedsl",
    "--enable-linear-replayssm-spec": "",
    "--linear-replayssm-cache-len": "32",
    "--api-early-reject-max-concurrency": "32",
    "--max-mamba-cache-size": "130",
    "--mamba-max-states-per-path": "4",
    "--mamba-radix-cache-strategy": "extra_buffer_lazy",
    "--cuda-graph-max-bs-decode": DECODE_CUDA_GRAPH_MAX_BS,
    "--cuda-graph-bs-decode": "1 2 4 8 12 16 24 32 48",
    "--enable-cache-report": "",
    "--speculative-algorithm": "DFLASH",
    "--speculative-attention-mode": "decode",
    "--speculative-draft-model-revision": SPECULATIVE_DRAFT_MODEL_REVISION,
    "--speculative-draft-load-format": DRAFT_LOAD_FORMAT,
    "--speculative-num-steps": "1",
    "--speculative-num-draft-tokens": "8",
    "--speculative-dflash-block-size": "8",
    "--speculative-draft-window-size": "4096",
    "--speculative-eagle-topk": "1",
    "--speculative-draft-attention-backend": "trtllm_mha",
    "--speculative-draft-kv-cache-dtype": DRAFT_KV_CACHE_DTYPE,
    "--speculative-draft-model-quantization": "fp8",
    "--speculative-draft-fp8-activation-scheme": "static",
    "--reasoning-parser": "kimi_k3",
    "--tool-call-parser": "kimi_k3",
    "--stream-response-default-include-usage": "",
    "--cuda-graph-backend-prefill": "breakable",
    "--cuda-graph-max-bs-prefill": PREFILL_CUDA_GRAPH_MAX_BS,
    "--cuda-graph-bs-prefill": PREFILL_CUDA_GRAPH_BS,
    "--enable-metrics": "",
}

SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
} | EXTRA_SERVER_ARGS

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with a short greeting."}],
    "max_tokens": 64,
    "temperature": 0.7,
}

ANTHROPIC_WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "max_tokens": 64,
    "stop_sequences": ["</warmup>"],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Reply with a short greeting."}]}
    ],
}

app = modal.App(name="ep-kimi-k3-anthropic-claude-code")


@app.server(
    include_source=True,
    image=serving_image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    volumes={
        "/flash-endpoint-model": modal.Volume.from_name("endpoint-ep-OAimsKXm6F5j7B5HHffvN3"),
        HF_CACHE_MOUNT_PATH: hf_cache,
        JIT_CACHE_MOUNT_PATH: jit_cache,
    },
    min_containers=0,
    target_concurrency=TARGET_CONCURRENCY,
    scaledown_window=10 * MINUTES,
    startup_timeout=3 * HOURS,
    port=PORT,
    unauthenticated=not REQUIRE_AUTHENTICATION,
    exit_grace_period=25,
    routing_region=ROUTING_REGION,
    experimental_options={"override_eof_timeout": 1800},
)
class Server:

    @modal.enter()
    def startup(self) -> None:
        from autoinference_utils.endpoint import (
            SGLangEndpoint,
            warmup_chat_completions,
        )

        def materialize_model_path(path: str, destination_name: str) -> str:
            # transformers' local-dir remote-code loader resolves relative
            # imports inside blobs/ on symlinked snapshots; repo-ids are fine.
            source = Path(path)
            if not source.is_dir():
                return path
            destination = Path("/tmp") / destination_name
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            for entry in source.iterdir():
                target = destination / entry.name
                if entry.name.endswith(".safetensors"):
                    target.symlink_to(entry)
                elif entry.is_dir():
                    shutil.copytree(entry, target, symlinks=False)
                else:
                    shutil.copy2(entry, target, follow_symlinks=True)
            print(f"Materialized {path} -> {destination}")
            return str(destination)

        started = time.monotonic()

        branch_sha = Path("/etc/sglang-branch-sha").read_text().strip()
        print(
            "Kimi K3 runtime configuration: "
            f"model_revision={MODEL_REVISION!r}, "
            f"fork_commit={branch_sha!r}, "
            f"draft_repo={SPECULATIVE_DRAFT_MODEL_REPO_ID!r}, "
            f"draft_revision={SPECULATIVE_DRAFT_MODEL_REVISION!r}, "
            f"server_args={SERVER_ARGS!r}"
        )

        self.endpoint = SGLangEndpoint(
            model_path=materialize_model_path(MODEL_PATH, "kimi-k3-model"),
            worker_port=PORT,
            tp=TP_SIZE,
            speculative_model_path="/flash-endpoint-model/huggingface/hub/models--modal-labs--Kimi-K3-DFlash/snapshots/c192d15a43407bf758b5ae0880d5c72052fef1de",
            extra_server_args=SERVER_ARGS,
            health_timeout=3 * HOURS,
            health_poll_interval=10.0,
            health_request_timeout=30.0,
        )
        self.endpoint.start()
        endpoint_elapsed = time.monotonic() - started

        warmup_chat_completions(
            port=PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=1,
            request_timeout=30 * MINUTES,
        )
        warmup_elapsed = time.monotonic() - started

        anthropic_warmup_elapsed = self._warmup_anthropic(port=PORT)

        print(
            "Kimi K3 TP8 DFlash is ready (Anthropic branch patched). "
            f"endpoint={endpoint_elapsed:.2f}s cumulative, "
            f"warmup={warmup_elapsed:.2f}s cumulative, "
            f"anthropic_warmup={anthropic_warmup_elapsed:.2f}s."
        )

    @staticmethod
    def _warmup_anthropic(port: int, timeout_s: float = 30 * MINUTES) -> float:
        """Drive /v1/messages once (non-stream + stream) so the branch's
        Anthropic conversion/response path is warm before the first client."""
        import json
        import urllib.request

        def _post(path: str, payload: dict) -> dict:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(payload).encode(),
                headers={
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())

        started = time.monotonic()
        body = dict(ANTHROPIC_WARMUP_PAYLOAD, stream=False)
        out = _post("/v1/messages", body)
        assert out.get("type") == "message", f"unexpected warmup envelope: {out!r}"
        stream_body = dict(ANTHROPIC_WARMUP_PAYLOAD, stream=True)
        # Streaming: drain the SSE feed; assert the canonical terminal frames.
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/messages",
            data=json.dumps(stream_body).encode(),
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        saw_message_start = saw_message_stop = False
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace")
                if '"message_start"' in line:
                    saw_message_start = True
                if '"message_stop"' in line:
                    saw_message_stop = True
        assert saw_message_start and saw_message_stop, (
            f"Anthropic SSE warmup missing canonical frames: "
            f"start={saw_message_start} stop={saw_message_stop}"
        )
        return time.monotonic() - started

    @modal.exit()
    def stop(self) -> None:
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
