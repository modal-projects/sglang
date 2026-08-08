"""Modal B200 benchmarks for the recent Stitch SGLang correctness stack.

The focused suite isolates the four production paths changed by the stack:

* DFlash sampling-mask materialization.
* DFlash Mamba radix track-index refresh.
* FlashInfer GDN prefill checkpoints.
* Routed-expert capture export.

Run the CPU-only image/source preflight before allocating a GPU::

    uv run modal run -e stitch-dev \
      benchmark/recent_sglang_changes_modal.py::preflight

Then run the focused B200 suite::

    uv run modal run -e stitch-dev \
      benchmark/recent_sglang_changes_modal.py::focused

Every remote entrypoint prints one machine-readable ``VERDICT`` line.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_SGLANG_PACKAGE = "/sgl-workspace/sglang/python/sglang"
TARGET_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
DRAFT_MODEL = "modal-labs/Qwen3.6-35B-A3B-DFlash"

app = modal.App("sglang-recent-correctness-perf")

hf_cache = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
sglang_cache = modal.Volume.from_name("sglang-cache", create_if_missing=True, version=2)

image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.16")
    .run_commands(f"rm -rf {REMOTE_SGLANG_PACKAGE}")
    .add_local_dir(
        str(REPO_ROOT / "python" / "sglang"),
        remote_path=REMOTE_SGLANG_PACKAGE,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env(
        {
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        }
    )
    # Modal volumes cannot mount over non-empty image paths.
    .run_commands("rm -rf /root/.cache/huggingface /root/.cache/sglang")
)


def _print_verdict(name: str, payload: dict) -> dict:
    print(f"VERDICT {name} PASS {json.dumps(payload, sort_keys=True)}", flush=True)
    return payload


def _server_arguments(port: int) -> list[str]:
    """The exact end-to-end server shape, shared with the cheap parser probe."""
    return [
        "--model-path",
        TARGET_MODEL,
        "--trust-remote-code",
        "--quantization",
        "modelopt_fp4",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-dflash-block-size",
        "8",
        "--speculative-draft-attention-backend",
        "fa4",
        "--attention-backend",
        "trtllm_mha",
        "--linear-attn-prefill-backend",
        "flashinfer",
        "--linear-attn-decode-backend",
        "flashinfer",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--moe-runner-backend",
        "flashinfer_trtllm_routed",
        "--disable-shared-experts-fusion",
        "--enable-return-routed-experts",
        "--mem-fraction-static",
        "0.65",
        "--max-total-tokens",
        "32768",
        "--max-mamba-cache-size",
        "160",
        "--chunked-prefill-size",
        "8192",
        "--max-running-requests",
        "32",
        "--max-queued-requests",
        "32",
        "--cuda-graph-max-bs-decode",
        "32",
        "--cuda-graph-backend-prefill",
        "tc_piecewise",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]


@app.function(image=image, cpu=2, memory=4096, timeout=10 * 60)
def preflight() -> dict:
    """Fail cheaply if the image does not contain the intended source overlay."""
    import argparse
    import inspect

    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative import dflash_utils

    source = inspect.getsource(dflash_utils.build_dflash_sampling_mask_output)
    implementations = {
        "synchronous": (
            "support_ids.cpu().tolist()",
            "selected_logprobs",
        ),
        "deferred": (
            "support_mask = target_probs > 0",
            "DFlashSamplingMaskOutput",
            "selected_logprobs",
        ),
    }
    implementation = next(
        (
            name
            for name, fragments in implementations.items()
            if all(fragment in source for fragment in fragments)
        ),
        None,
    )
    if implementation is None:
        raise RuntimeError("unrecognized DFlash sampling-mask implementation")

    parser = argparse.ArgumentParser(prog="sglang recent perf preflight")
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(_server_arguments(port=30_000))
    expected_args = {
        "speculative_algorithm": "DFLASH",
        "speculative_dflash_block_size": 8,
        "max_total_tokens": 32_768,
        "max_mamba_cache_size": 160,
        "enable_return_routed_experts": True,
    }
    actual_args = {name: getattr(parsed, name) for name in expected_args}
    if actual_args != expected_args:
        raise RuntimeError(
            f"end-to-end server argument mismatch: {actual_args} != {expected_args}"
        )

    payload = {
        "module": inspect.getfile(dflash_utils),
        "implementation": implementation,
        "server_args": actual_args,
    }
    return _print_verdict("sglang_recent_perf_preflight", payload)


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=30 * 60)
def focused() -> dict:
    """Run focused wall-clock benchmarks on one B200."""
    import statistics
    import time
    from types import SimpleNamespace

    import torch

    from sglang.srt.managers.schedule_batch import (
        set_mamba_track_indices_from_reqs,
    )
    from sglang.srt.managers.utils import _async_d2h
    from sglang.srt.speculative.dflash_utils import (
        build_dflash_sampling_mask_output,
    )
    from sglang.srt.state_capturer.base import (
        BaseDeviceCache,
        BaseHostCache,
        TopkCaptureOutput,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("focused benchmarks require CUDA")

    torch.manual_seed(0)
    device = torch.device("cuda")

    def measure_ms(fn, *, warmup: int, iterations: int) -> dict[str, float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000)

        samples.sort()
        p95_index = min(len(samples) - 1, int(len(samples) * 0.95))
        return {
            "mean_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": samples[p95_index],
            "min_ms": samples[0],
        }

    results: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(),
        # torch.__version__ is a TorchVersion object, which would make Modal's
        # local client import torch while deserializing the otherwise plain dict.
        "torch": str(torch.__version__),
    }

    # Qwen3.6 has a 248,320-token vocabulary. A batch of 32 and four emitted
    # tokens per eight-token DFlash block is representative of the rollout.
    batch_size = 32
    block_size = 8
    vocab_size = 248_320
    accepted_tokens = 4
    row_count = batch_size * block_size
    output_lens = torch.full(
        (batch_size,), accepted_tokens, dtype=torch.int32, device=device
    )
    return_sampling_masks = [True] * batch_size

    for support_size in (20, 4_096):
        flat_probs = torch.zeros(
            (row_count, vocab_size), dtype=torch.bfloat16, device=device
        )
        support_offsets = torch.arange(support_size, device=device, dtype=torch.int64)
        support_ids = (
            torch.arange(row_count, device=device, dtype=torch.int64)[:, None] * 4099
            + support_offsets[None, :]
        ) % vocab_size
        flat_probs.scatter_(
            1,
            support_ids,
            torch.full(
                (row_count, support_size),
                1.0 / support_size,
                dtype=flat_probs.dtype,
                device=device,
            ),
        )
        target_probs = flat_probs.view(batch_size, block_size, vocab_size)
        output_token_ids = support_ids[:, 0].view(batch_size, block_size)
        sampling_output = None

        def materialize_sampling_masks() -> None:
            nonlocal sampling_output
            pending = build_dflash_sampling_mask_output(
                target_probs=target_probs,
                output_token_ids=output_token_ids,
                output_lens=output_lens,
                return_sampling_masks=return_sampling_masks,
            )
            if hasattr(pending, "map_device_tensors"):
                pending.map_device_tensors(_async_d2h)
                torch.cuda.synchronize()
                sampling_output = pending.finalize()
            else:
                sampling_output = pending

        results[f"dflash_sampling_mask_support_{support_size}"] = {
            **measure_ms(materialize_sampling_masks, warmup=2, iterations=10),
            "batch_size": batch_size,
            "block_size": block_size,
            "accepted_tokens_per_request": accepted_tokens,
            "support_size": support_size,
            "returned_support_ids": batch_size * accepted_tokens * support_size,
        }
        if sampling_output is None or len(sampling_output[0]) != batch_size:
            raise RuntimeError("DFlash sampling-mask benchmark produced invalid output")

        del sampling_output, flat_probs, support_ids
        torch.cuda.empty_cache()

    # Measure the exact per-verify track-index rebuild added for DFlash Mamba.
    ping_pong_mapping = torch.arange(4096 * 2, dtype=torch.int64, device=device).view(
        4096, 2
    )
    batch = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=ping_pong_mapping
        ),
        req_pool_indices=torch.arange(batch_size, device=device, dtype=torch.int64),
        reqs=[SimpleNamespace(mamba_next_track_idx=i % 2) for i in range(batch_size)],
        mamba_track_indices=None,
    )
    results["mamba_track_index_refresh"] = {
        **measure_ms(
            lambda: set_mamba_track_indices_from_reqs(batch),
            warmup=50,
            iterations=500,
        ),
        "batch_size": batch_size,
    }

    seq_lens_pre = torch.arange(255, 255 + batch_size, dtype=torch.int64, device=device)
    commit_lens = torch.full(
        (batch_size,), accepted_tokens, dtype=torch.int32, device=device
    )

    def compute_mamba_tracking_steps() -> None:
        interval = 256
        seq_lens_post = seq_lens_pre + commit_lens.to(seq_lens_pre.dtype)
        to_track = seq_lens_pre // interval != seq_lens_post // interval
        tracking_point = seq_lens_post // interval * interval
        track_ith = torch.clamp(tracking_point - seq_lens_pre - 1, min=0)
        can_track = to_track & (track_ith < commit_lens.to(track_ith.dtype))
        torch.where(
            can_track,
            track_ith.to(torch.int64),
            torch.full_like(track_ith, -1, dtype=torch.int64),
        )

    results["mamba_tracking_step_math"] = {
        **measure_ms(compute_mamba_tracking_steps, warmup=50, iterations=500),
        "batch_size": batch_size,
    }

    # Measure the R3 cost newly exposed by propagating the capture output from
    # the target worker into GenerationBatchResult.
    verify_tokens = batch_size * block_size
    num_layers = 40
    experts_per_token = 8
    device_cache = BaseDeviceCache(
        max_batch_size=verify_tokens,
        num_layers=num_layers,
        topk_size=experts_per_token,
        device="cuda",
        name="benchmark_routed_experts",
    )
    host_cache = BaseHostCache(
        num_tokens=16_384,
        num_layers=num_layers,
        topk_size=experts_per_token,
        name="benchmark_routed_experts",
    )
    layer_topk = torch.randint(
        0,
        256,
        (verify_tokens, experts_per_token),
        dtype=torch.int32,
        device=device,
    )
    out_cache_loc = torch.arange(verify_tokens, dtype=torch.int64, device=device)

    def capture_all_layers() -> None:
        for layer_id in range(num_layers):
            device_cache.capture(layer_id, layer_topk)

    results["r3_device_capture"] = {
        **measure_ms(capture_all_layers, warmup=10, iterations=100),
        "verify_tokens": verify_tokens,
        "num_layers": num_layers,
        "experts_per_token": experts_per_token,
    }

    def export_r3_capture() -> None:
        output = TopkCaptureOutput(
            out_cache_loc=out_cache_loc,
            topk=device_cache.buffer,
            host_cache=host_cache,
        )
        output.map_device_tensors(_async_d2h)
        torch.cuda.synchronize()
        output.finalize()

    results["r3_result_export"] = {
        **measure_ms(export_r3_capture, warmup=10, iterations=100),
        "d2h_bytes": (
            out_cache_loc.numel() * out_cache_loc.element_size()
            + device_cache.buffer.numel() * device_cache.buffer.element_size()
        ),
    }

    # Run the actual FlashInfer SM100 GDN prefill kernel with Qwen3.6's head
    # dimensions. This includes checkpoint writes, not merely Python setup.
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )

    gdn_kernel = FlashInferGDNKernel()
    prefill_batch_size = 8
    total_tokens = 8_192
    tokens_per_request = total_tokens // prefill_batch_size
    num_k_heads = 16
    num_v_heads = 32
    head_dim = 128
    q = torch.randn(
        (1, total_tokens, num_k_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        (1, total_tokens, num_v_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    g = torch.randn((1, total_tokens, num_v_heads), dtype=torch.bfloat16, device=device)
    beta = torch.randn_like(g)
    ssm_states = torch.zeros(
        (prefill_batch_size + 1, num_v_heads, head_dim, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    cache_indices = torch.arange(prefill_batch_size, dtype=torch.int64, device=device)
    query_start_loc = torch.arange(
        0,
        total_tokens + 1,
        tokens_per_request,
        dtype=torch.int32,
        device=device,
    )

    def run_gdn_prefill(return_checkpoints: bool) -> None:
        gdn_kernel.extend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            return_intermediate_states=return_checkpoints,
            checkpoint_every_n_tokens=64,
        )

    without_checkpoints = measure_ms(
        lambda: run_gdn_prefill(False), warmup=2, iterations=10
    )
    with_checkpoints = measure_ms(
        lambda: run_gdn_prefill(True), warmup=2, iterations=10
    )
    results["gdn_prefill_without_checkpoints"] = {
        **without_checkpoints,
        "total_tokens": total_tokens,
    }
    results["gdn_prefill_with_checkpoints"] = {
        **with_checkpoints,
        "total_tokens": total_tokens,
        "checkpoint_interval": 64,
        "checkpoint_buffer_bytes": (
            total_tokens
            // 64
            * num_v_heads
            * head_dim
            * head_dim
            * torch.bfloat16.itemsize
        ),
        "median_slowdown_ratio": (
            with_checkpoints["median_ms"] / without_checkpoints["median_ms"]
        ),
    }

    return _print_verdict("sglang_recent_perf_focused", results)


@app.function(
    image=image,
    gpu="B200",
    cpu=16,
    memory=(128 * 1024, 512 * 1024),
    ephemeral_disk=512 * 1024,
    region="us-west",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/sglang": sglang_cache,
    },
    timeout=60 * 60,
    startup_timeout=60 * 60,
)
def end_to_end() -> dict:
    """Compare DFlash top-p requests with and without returned masks."""
    import asyncio
    import os
    import signal
    import statistics
    import subprocess
    import time
    import urllib.request

    import httpx

    port = 30_000
    base_url = f"http://127.0.0.1:{port}"
    command = ["python", "-m", "sglang.launch_server", *_server_arguments(port)]
    server_env = {
        **os.environ,
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
        "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
    }
    print(f"STARTING_SERVER {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, env=server_env)

    def wait_for_server(timeout_seconds: float = 45 * 60) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"SGLang exited with code {process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"{base_url}/health_generate", timeout=5
                ) as response:
                    if response.status == 200:
                        print("SERVER_READY", flush=True)
                        return
            except Exception as exc:  # noqa: BLE001 - preserve last health error
                last_error = exc
            time.sleep(2)
        raise TimeoutError(f"SGLang did not become ready: {last_error}")

    async def run_round(
        *, return_sampling_mask: bool, concurrency: int, round_index: int
    ) -> dict[str, float | int | bool]:
        prompt_body = (
            "Analyze the algorithmic tradeoffs in this distributed systems design, "
            "identify failure modes, and propose concrete mitigations. "
        ) * 24
        payloads = [
            {
                "text": f"Workload {request_index}. {prompt_body}",
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": -1,
                    "max_new_tokens": 512,
                    "ignore_eos": True,
                    # Keep samples and DFlash acceptance behavior paired between
                    # modes. round_index only labels the result.
                    "sampling_seed": 10_000 + request_index,
                },
                "return_sampling_mask": return_sampling_mask,
                "return_logprob": False,
            }
            for request_index in range(concurrency)
        ]

        async with httpx.AsyncClient(timeout=20 * 60) as client:
            started = time.perf_counter()
            responses = await asyncio.gather(
                *(
                    client.post(f"{base_url}/generate", json=payload)
                    for payload in payloads
                )
            )
            elapsed = time.perf_counter() - started

        output_tokens = 0
        returned_masks = 0
        returned_support_ids = 0
        for response in responses:
            response.raise_for_status()
            body = response.json()
            output_ids = body["output_ids"]
            output_tokens += len(output_ids)
            if return_sampling_mask:
                masks = body["meta_info"].get("output_token_sampling_mask") or []
                if len(masks) != len(output_ids):
                    raise RuntimeError(
                        "sampling-mask count did not match generated tokens: "
                        f"{len(masks)} != {len(output_ids)}"
                    )
                returned_masks += len(masks)
                returned_support_ids += sum(len(mask) for mask in masks)

        return {
            "return_sampling_mask": return_sampling_mask,
            "concurrency": concurrency,
            "elapsed_seconds": elapsed,
            "output_tokens": output_tokens,
            "returned_masks": returned_masks,
            "returned_support_ids": returned_support_ids,
            "output_tokens_per_second": output_tokens / elapsed,
        }

    try:
        wait_for_server()
        # Compile/warm all decode graphs and populate the same prompt prefixes for
        # both modes before recording either side.
        asyncio.run(
            run_round(return_sampling_mask=False, concurrency=32, round_index=0)
        )
        asyncio.run(run_round(return_sampling_mask=True, concurrency=32, round_index=0))

        rounds = []
        # Alternate order so cache/clock drift does not consistently favor one mode.
        modes = (False, True, True, False, True, False)
        for round_index, return_masks in enumerate(modes, start=1):
            result = asyncio.run(
                run_round(
                    return_sampling_mask=return_masks,
                    concurrency=32,
                    round_index=round_index,
                )
            )
            rounds.append(result)
            print(f"ROUND {json.dumps(result, sort_keys=True)}", flush=True)

        throughput_by_mode = {
            return_masks: [
                float(round_result["output_tokens_per_second"])
                for round_result in rounds
                if round_result["return_sampling_mask"] is return_masks
            ]
            for return_masks in (False, True)
        }
        without_masks = statistics.median(throughput_by_mode[False])
        with_masks = statistics.median(throughput_by_mode[True])
        payload = {
            "model": TARGET_MODEL,
            "draft_model": DRAFT_MODEL,
            "rounds": rounds,
            "median_output_tokens_per_second_without_masks": without_masks,
            "median_output_tokens_per_second_with_masks": with_masks,
            "throughput_ratio_with_over_without": with_masks / without_masks,
        }
        return _print_verdict("sglang_recent_perf_end_to_end", payload)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
