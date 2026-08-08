"""Qwen3.8-Max NVFP4-RTN on one 8xB300 node, TP8: trtllm MoE + static activation scales.

Validated 2026-08-07: GPQA-Diamond 175/198 (88.4%); the FP8 reference scores
178/198 on the same harness, so this quant is at effective FP8 parity.

CHECKPOINT CAVEAT: Qwen3.8-Max-NVFP4-RTN-v2 is round-to-nearest with
uncalibrated static activation scales (input_scale=1.0). We want to replace
it with a calibrated-scales checkpoint -- either our own calibration run or
the RadixArk one Harmya is evaluating -- and revisit the activation-scale
mode then. Per-token activation quant is NOT a workaround on the trtllm
backend: the flashinfer trtllm-gen per-token path miscomputes at this
model's shape (512 experts / topk 10 / hidden 8192), collapsing output to
"!!!" -- hence SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=0 below.

To deploy: set APP_NAME below (it ships unset so a checked-in copy can never
redeploy someone else's app), then run (runc is required for sane
weight-load times):

  MODAL_FUNCTION_RUNTIME=runc MODAL_PROFILE=modal-labs MODAL_ENVIRONMENT=qwen-bringup \
    uv run modal deploy modal_deployments/qwen38_max_nvfp4_rtn_8xb300/serve.py
"""

from __future__ import annotations

import modal

MINUTES = 60
PORT = 8000

TP_SIZE = 8
GPU = "B300:8"

# NVFP4 routed experts (92 main layers), BF16 everything else incl. the
# mtp.* head; needs 8xB300 (2.3 TB).
MODEL_MOUNT = "/model"
MODEL_PATH = f"{MODEL_MOUNT}/Qwen3.8-Max-NVFP4-RTN-v2"
SERVED_MODEL_NAME = "qwen3.8-max-nvfp4-rtn"
# 3h: volume throughput degrades under concurrent readers; a 1.5TB load has
# been observed to decelerate past a 90-min budget mid-load.
STARTUP_TIMEOUT = 180 * MINUTES

# Set your own app name, e.g. "qwen38-max-nvfp4-rtn-8xb300-<you>". Ships
# unset so deploying this file as-is can never redeploy a live app.
APP_NAME = None
if not APP_NAME:
    raise SystemExit("Edit APP_NAME in this file before deploying.")

# sglang tree checked out over the nightly image; qwen38-bringup carries the
# language_model_only fix this text-only VL export needs.
SGLANG_REF = "qwen38-bringup"

serving_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260806-ae5f8c94")
    .uv_pip_install(
        "autoinference-utils==0.2.2",
        "fastsafetensors==0.3.3",
    )
    .run_commands(
        "cd /sgl-workspace/sglang && "
        "git fetch https://github.com/modal-projects/sglang.git "
        f"{SGLANG_REF} && "
        "git checkout FETCH_HEAD",
    )
    .env({
        "HF_HUB_OFFLINE": "1",
        # static scales: the trtllm-gen per-token path is broken at this
        # model's shape (see module docstring)
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "0",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

SERVER_ARGS = {
    "--served-model-name": SERVED_MODEL_NAME,
    "--quantization": "modelopt_fp4",
    "--dtype": "bfloat16",
    "--attention-backend": "trtllm_mha",
    "--page-size": "64",
    "--linear-attn-prefill-backend": "flashinfer",
    "--linear-attn-decode-backend": "flashinfer",
    "--mamba-ssm-dtype": "bfloat16",
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    "--chunked-prefill-size": "16384",
    "--context-length": "131072",
    "--cuda-graph-max-bs": "16",
    "--dist-timeout": "3600",
    "--max-running-requests": "16",
    "--mem-fraction-static": "0.85",
    "--moe-runner-backend": "flashinfer_trtllm",
    "--trust-remote-code": "",
    "--load-format": "fastsafetensors",
    # GDS off: cuFile opens fail on FUSE volumes ("Error opening file");
    # compact JSON -- the endpoint wrapper splits arg values on whitespace
    "--model-loader-extra-config": '{"enable_gds":false}',
}

app = modal.App(name=APP_NAME)


@app.server(
    image=serving_image,
    gpu=GPU,
    cpu=32,
    memory=262144,
    min_containers=1,
    max_containers=1,
    scaledown_window=10 * MINUTES,
    port=PORT,
    routing_region="us-west",
    unauthenticated=True,
    exit_grace_period=25,
    startup_timeout=STARTUP_TIMEOUT,
    target_concurrency=16,
    volumes={
        MODEL_MOUNT: modal.Volume.from_name("qwen38-max-nvfp4-rtn"),
    },
)
class Server:
    @modal.enter()
    def startup(self):
        from autoinference_utils.endpoint import (
            SGLangEndpoint,
            warmup_chat_completions,
        )

        print(f"Starting SGLang with server args: {SERVER_ARGS}")
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_PATH,
            worker_port=PORT,
            tp=TP_SIZE,
            extra_server_args=SERVER_ARGS,
            health_timeout=STARTUP_TIMEOUT,
            health_poll_interval=10.0,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=PORT,
            payload={
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 8,
                "temperature": 0,
            },
            successful_requests=2,
            request_timeout=180.0,
        )
        print(f"{SERVED_MODEL_NAME} ({GPU}) is ready.")

    @modal.exit()
    def stop(self):
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
