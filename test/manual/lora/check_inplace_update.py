"""Check a live adapter replacement while another adapter keeps streaming."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.utils import kill_process_tree

MODEL = "Qwen/Qwen3-0.6B"


def make_adapter(directory: Path, seed: int, scale: float) -> None:
    config = AutoConfig.from_pretrained(MODEL)
    directory.mkdir()
    rank = 8
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": MODEL,
                "r": rank,
                "lora_alpha": rank,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
            }
        )
    )
    generator = torch.Generator().manual_seed(seed)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    weights = {}
    for layer in range(config.num_hidden_layers):
        for target, heads in (
            ("q_proj", config.num_attention_heads),
            ("v_proj", config.num_key_value_heads),
        ):
            for matrix, shape in (
                ("A", (rank, config.hidden_size)),
                ("B", (heads * head_dim, rank)),
            ):
                key = f"base_model.model.model.layers.{layer}.self_attn.{target}.lora_{matrix}.weight"
                weights[key] = (torch.randn(shape, generator=generator) * scale).to(
                    torch.bfloat16
                )
    save_file(weights, str(directory / "adapter_model.safetensors"))


def post(url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url + endpoint, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def token_ids(payload: dict[str, Any]) -> list[int]:
    if "output_ids" in payload:
        return payload["output_ids"]
    return [entry[1] for entry in payload["meta_info"]["output_token_logprobs"]]


def stream(
    url: str,
    adapter: str,
    prompt: list[int],
    tokens: int,
    started: threading.Event,
    samples: list[tuple[float, int]],
) -> dict[str, Any]:
    payload = {
        "input_ids": prompt,
        "lora_path": adapter,
        "stream": True,
        "return_logprob": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": tokens,
            "ignore_eos": True,
        },
    }
    last = None
    with requests.post(
        url + "/generate", json=payload, stream=True, timeout=180
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=None):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            last = json.loads(line[6:])
            count = last["meta_info"]["completion_tokens"]
            if not samples or count > samples[-1][1]:
                samples.append((time.monotonic(), count))
            if count >= 32:
                started.set()
    assert last is not None
    return last


def run(url: str, root: Path, tokens: int) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt = tokenizer.encode(
        "Write the integers from one to one thousand, separated by spaces. Begin: 1 2 3"
    )
    for name, path in (("A", root / "a1"), ("B", root / "b")):
        post(url, "/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        post(
            url,
            "/generate",
            {
                "input_ids": prompt,
                "lora_path": name,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            },
        )
    results = []
    for update in (False, True):
        print(f"Streaming case: update={update}", flush=True)
        events = [threading.Event(), threading.Event()]
        samples: list[list[tuple[float, int]]] = [[], []]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(stream, url, name, prompt, tokens, event, output)
                for name, event, output in zip(("A", "B"), events, samples, strict=True)
            ]
            assert all(event.wait(120) for event in events)
            started = time.monotonic()
            if update:
                invalid = requests.post(
                    url + "/update_lora_adapter",
                    json={
                        "lora_name": "A",
                        "lora_path": str(root / "missing"),
                        "version": "bad",
                    },
                    timeout=180,
                )
                assert invalid.status_code == 400, invalid.text
                updated = post(
                    url,
                    "/update_lora_adapter",
                    {
                        "lora_name": "A",
                        "lora_path": str(root / "a2"),
                        "version": "v2",
                        "expected_version": "A",
                    },
                )
                assert updated["success"], updated
            else:
                time.sleep(0.25)
            ended = time.monotonic()
            a, b = [future.result(timeout=180) for future in futures]
        assert len(token_ids(a)) == len(token_ids(b)) == tokens
        b_gaps = [right[0] - left[0] for left, right in zip(samples[1], samples[1][1:])]
        left, right = max(
            zip(samples[1], samples[1][1:]), key=lambda pair: pair[1][0] - pair[0][0]
        )
        result = {
            "update": update,
            "b_max_gap_from_token": left[1],
            "b_max_gap_to_token": right[1],
            "b_max_gap_start_relative_to_update_s": left[0] - started,
            "b_max_gap_end_relative_to_update_s": right[0] - started,
            "window_s": ended - started,
            "b_tokens_during_window": sum(
                started <= timestamp <= ended for timestamp, _ in samples[1]
            ),
            "b_max_inter_chunk_s": max(b_gaps),
            "b_median_inter_chunk_s": sorted(b_gaps)[len(b_gaps) // 2],
            "a_retractions": a["meta_info"].get("num_retractions"),
        }
        if update:
            assert a["meta_info"]["num_retractions"] >= 1
            assert b["meta_info"]["num_retractions"] == 0
            spans = a["meta_info"]["weight_versions"]
            assert [span["version"] for span in spans] == ["A", "v2"], spans
            assert spans[0]["start"] == 0 and spans[0]["end"] == spans[1]["start"]
            assert spans[1]["end"] == tokens
            assert result["b_tokens_during_window"] > 0, result
            assert all(
                span["version"] != "v2"
                for span in b["meta_info"].get("weight_versions", [])
            )
            post(
                url,
                "/load_lora_adapter",
                {"lora_name": "reference", "lora_path": str(root / "a2")},
            )
            boundary = spans[1]["start"]
            count = min(16, tokens - boundary)
            reference = post(
                url,
                "/generate",
                {
                    "input_ids": prompt + token_ids(a)[:boundary],
                    "lora_path": "reference",
                    "return_logprob": True,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": count,
                        "ignore_eos": True,
                    },
                },
            )
            actual = token_ids(a)[boundary : boundary + count]
            assert actual == token_ids(reference), (actual, token_ids(reference))
            actual_probs = [
                entry[0] for entry in a["meta_info"]["output_token_logprobs"]
            ][boundary : boundary + count]
            reference_probs = [
                entry[0] for entry in reference["meta_info"]["output_token_logprobs"]
            ]
            torch.testing.assert_close(
                torch.tensor(actual_probs),
                torch.tensor(reference_probs),
                atol=0.05,
                rtol=0.01,
            )
            retry = post(
                url,
                "/update_lora_adapter",
                {"lora_name": "A", "lora_path": str(root / "a2"), "version": "v2"},
            )
            assert retry["success"]
            result["version_spans"] = spans
            result["fresh_prefix_tokens_verified"] = count
        results.append(result)
    migration = check_version_migration(url, prompt, root, tokens)
    return {
        "model": MODEL,
        "tokens_per_request": tokens,
        "cases": results,
        "migration": migration,
    }


def check_version_migration(
    url: str, prompt: list[int], root: Path, tokens: int
) -> dict[str, Any]:
    print("Streaming case: requests starting on different policy versions", flush=True)
    events = [threading.Event() for _ in range(3)]
    samples: list[list[tuple[float, int]]] = [[], [], []]
    with ThreadPoolExecutor(max_workers=3) as executor:
        old = executor.submit(stream, url, "A", prompt, tokens, events[0], samples[0])
        peer = executor.submit(stream, url, "B", prompt, tokens, events[1], samples[1])
        assert events[0].wait(120) and events[1].wait(120)
        post(
            url,
            "/update_lora_adapter",
            {
                "lora_name": "A",
                "lora_path": str(root / "a1"),
                "version": "v3",
                "expected_version": "v2",
            },
        )
        new = executor.submit(stream, url, "A", prompt, tokens, events[2], samples[2])
        assert events[2].wait(120)
        started = time.monotonic()
        updated = post(
            url,
            "/update_lora_adapter",
            {
                "lora_name": "A",
                "lora_path": str(root / "a2"),
                "version": "v4",
                "expected_version": "v3",
            },
        )
        ended = time.monotonic()
        assert set(updated["loaded_adapters"]) == {"A", "B", "reference"}
        a_old, a_new, b = [future.result(timeout=180) for future in (old, new, peer)]
    for result, versions in ((a_old, ["v2", "v3", "v4"]), (a_new, ["v3", "v4"])):
        spans = result["meta_info"]["weight_versions"]
        assert [span["version"] for span in spans] == versions, spans
        assert len(token_ids(result)) == tokens
        boundary = spans[-1]["start"]
        count = min(8, tokens - boundary)
        reference = post(
            url,
            "/generate",
            {
                "input_ids": prompt + token_ids(result)[:boundary],
                "lora_path": "reference",
                "return_logprob": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": count,
                    "ignore_eos": True,
                },
            },
        )
        assert token_ids(result)[boundary : boundary + count] == token_ids(reference)
    assert b["meta_info"]["num_retractions"] == 0
    assert all(
        span["version"] not in ("v3", "v4")
        for span in b["meta_info"].get("weight_versions", [])
    )
    return {
        "old_request_spans": a_old["meta_info"]["weight_versions"],
        "new_request_spans": a_new["meta_info"]["weight_versions"],
        "update_s": ended - started,
        "b_tokens_during_update": sum(
            started <= timestamp <= ended for timestamp, _ in samples[1]
        ),
        "b_retractions": b["meta_info"]["num_retractions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lora-inplace-") as temporary:
        root = Path(temporary)
        for name, seed, scale in (("a1", 7, 0.05), ("a2", 8, 0.1), ("b", 9, 0.05)):
            make_adapter(root / name, seed, scale)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp-size",
            str(args.tp_size),
            "--enable-lora",
            "--max-lora-rank",
            "8",
            "--max-loras-per-batch",
            "4",
            "--lora-target-modules",
            "q_proj",
            "v_proj",
            "--mem-fraction-static",
            "0.6",
            "--max-running-requests",
            "8",
            "--context-length",
            "4096",
            "--cuda-graph-max-bs-decode",
            "8",
        ]
        if args.disable_cuda_graph:
            command.append("--disable-cuda-graph")
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while True:
                    assert process.poll() is None, (root / "server.log").read_text()[
                        -20000:
                    ]
                    try:
                        if requests.get(url + "/health", timeout=2).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup timed out")
                    time.sleep(1)
                print(json.dumps(run(url, root, args.tokens), indent=2), flush=True)
            except BaseException:
                print(
                    (root / "server.log").read_text()[-30000:],
                    file=sys.stderr,
                    flush=True,
                )
                raise
            finally:
                kill_process_tree(process.pid)
                process.wait(timeout=30)


if __name__ == "__main__":
    main()
