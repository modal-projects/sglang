from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import inspect
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import sglang as sgl
from sglang.srt.server_args import ServerArgs as RuntimeServerArgs
from sglang.srt.weight_sync.lora_payload_utils import (
    convert_peft_lora_tensors_to_weight_sync_payload,
    negate_lora_payload,
    serialize_weight_sync_payload,
)
from sglang.srt.weight_sync.update_bytes import (
    build_update_weights_request_from_named_tensors,
    load_named_tensors_from_bytes,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_ADAPTER_REPO = "yushengsu/lora-diff-Qwen3-VL-30B-A3B-Instruct"
WEIGHT_SYNC_CONSISTENCY_MODE_CHOICES = ("strict", "unsafe_reuse_kv")
SERVICE_IMPACT_UPDATE_TRANSPORT_CHOICES = (
    "prepared_tensor",
    "prepared_bytes_handle",
    "inline_bytes",
    "inline_bytes_to_thread",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the live LoRA weight-sync path with optional torch profiling.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path")
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--adapter-repo-type", default="dataset")
    parser.add_argument("--adapter-safetensors", default="adapter_model.safetensors")
    parser.add_argument("--adapter-config", default="adapter_config.json")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--benchmark-mode",
        choices=("sync", "service-impact"),
        default="sync",
    )
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--allow-skipped", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--profile-output-dir")
    parser.add_argument("--profile-cycle", type=int, default=1)
    parser.add_argument(
        "--profile-activities",
        nargs="+",
        default=["CPU", "GPU"],
        help="Profiler activities passed to Engine.start_profile.",
    )
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--prefill-attention-backend", default="fa4")
    parser.add_argument("--decode-attention-backend", default="fa4")
    parser.add_argument("--moe-runner-backend", default="triton")
    parser.add_argument(
        "--weight-sync-consistency-mode",
        choices=WEIGHT_SYNC_CONSISTENCY_MODE_CHOICES,
    )
    parser.add_argument(
        "--experts-shared-outer-loras",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--service-impact-requests", type=int, default=64)
    parser.add_argument("--service-impact-concurrency", type=int, default=8)
    parser.add_argument(
        "--service-impact-prompt",
        default="Count from 1 to 64, separated by spaces.",
    )
    parser.add_argument("--service-impact-max-new-tokens", type=int, default=96)
    parser.add_argument("--service-impact-update-after-s", type=float, default=0.5)
    parser.add_argument("--service-impact-profile-update", action="store_true")
    parser.add_argument("--service-impact-warmup-requests", type=int, default=0)
    parser.add_argument("--service-impact-warmup-concurrency", type=int, default=0)
    parser.add_argument("--service-impact-warmup-max-new-tokens", type=int, default=0)
    parser.add_argument(
        "--service-impact-update-transport",
        choices=SERVICE_IMPACT_UPDATE_TRANSPORT_CHOICES,
        default="prepared_tensor",
    )
    parser.add_argument(
        "--prepared-bytes-transport-format",
        choices=("serialized_named_tensors", "flattened_bucket"),
        default="flattened_bucket",
    )
    parser.add_argument(
        "--prepared-bytes-transport-bucket-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _resolve_adapter_path(args: argparse.Namespace) -> str:
    return resolve_adapter_path(
        adapter_path=args.adapter_path,
        adapter_repo=args.adapter_repo,
        adapter_repo_type=args.adapter_repo_type,
    )


def _load_adapter_payload(
    adapter_path: str,
    *,
    adapter_safetensors: str,
    adapter_config: str,
    allow_skipped: bool,
) -> tuple[List[tuple[str, torch.Tensor]], Dict[str, Any], bytes, bytes, List[str]]:
    tensor_path = os.path.join(adapter_path, adapter_safetensors)
    config_path = os.path.join(adapter_path, adapter_config)

    adapter_tensors = load_file(tensor_path)
    with open(config_path, "r", encoding="utf-8") as f:
        adapter_config_dict = json.load(f)

    prepared = convert_peft_lora_tensors_to_weight_sync_payload(
        adapter_tensors,
        adapter_config=adapter_config_dict,
    )
    if prepared.skipped_tensor_names and not allow_skipped:
        raise ValueError(
            "Skipped LoRA tensors while building the live weight-sync payload: "
            + ", ".join(prepared.skipped_tensor_names[:10])
        )

    apply_bytes = serialize_weight_sync_payload(prepared.named_tensors)
    revert_bytes = serialize_weight_sync_payload(negate_lora_payload(prepared.named_tensors))
    return (
        prepared.named_tensors,
        prepared.loader_metadata,
        apply_bytes,
        revert_bytes,
        prepared.skipped_tensor_names,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _get_prompt_logprobs(engine: sgl.Engine, input_ids: Iterable[int]) -> List[float]:
    out = engine.generate(
        input_ids=list(input_ids),
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
    )
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


def _trace_files(output_dir: Optional[str]) -> set[Path]:
    if not output_dir:
        return set()
    path = Path(output_dir)
    if not path.exists():
        return set()
    return set(path.glob("*.trace.json*"))


def _parse_trace_phase_totals(trace_path: Path) -> Dict[str, float]:
    if trace_path.suffix == ".gz":
        with gzip.open(trace_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with open(trace_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

    events = payload.get("traceEvents", payload if isinstance(payload, list) else [])
    totals_ms: Dict[str, float] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        name = event.get("name")
        dur_us = event.get("dur")
        if (
            not isinstance(name, str)
            or dur_us is None
            or not (
                name.startswith("weight_sync.") or name.startswith("lora_sync.")
            )
        ):
            continue
        totals_ms[name] = totals_ms.get(name, 0.0) + (float(dur_us) / 1000.0)
    return totals_ms


def _aggregate_trace_phase_totals(trace_paths: List[Path]) -> Dict[str, float]:
    max_totals: Dict[str, float] = {}
    for trace_path in trace_paths:
        for phase, total_ms in _parse_trace_phase_totals(trace_path).items():
            max_totals[phase] = max(max_totals.get(phase, 0.0), total_ms)
    return dict(sorted(max_totals.items()))


def _weight_sync_phase_breakdown(phase_totals: Dict[str, float]) -> Dict[str, float]:
    if not phase_totals:
        return {}
    breakdown = {
        "wait_for_writer_lock_ms": phase_totals.get(
            "weight_sync.wait_for_writer_lock", 0.0
        ),
        "unsafe_skip_writer_lock_ms": phase_totals.get(
            "weight_sync.unsafe_skip_writer_lock", 0.0
        ),
        "scheduler_step_boundary_wait_ms": phase_totals.get(
            "weight_sync.scheduler_step_boundary_wait", 0.0
        ),
        "deserialize_ms": phase_totals.get("weight_sync.deserialize", 0.0),
        "model_runner_update_ms": phase_totals.get(
            "weight_sync.model_runner_update", 0.0
        ),
        "worker_update_ms": phase_totals.get("weight_sync.worker_update", 0.0),
        "flush_cache_ms": phase_totals.get("weight_sync.flush_cache", 0.0),
        "barrier_ms": phase_totals.get("weight_sync.barrier", 0.0),
        "resume_admissions_ms": phase_totals.get(
            "weight_sync.resume_admissions", 0.0
        ),
    }
    return {key: value for key, value in breakdown.items() if value > 0.0}


def _weight_version_transition_counts(
    request_records: List[Dict[str, Any]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in request_records:
        before = record.get("weight_version_before")
        after = record.get("weight_version_after")
        if before is None and after is None:
            continue
        key = f"{before}->{after}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _weight_version_transition_token_counts(
    token_events: List[Dict[str, Any]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in token_events:
        before = event.get("weight_version_before")
        after = event.get("weight_version_after")
        if before is None and after is None:
            continue
        key = f"{before}->{after}"
        counts[key] = counts.get(key, 0) + int(event.get("new_tokens", 0) or 0)
    return dict(sorted(counts.items()))


def _first_event_after_s(
    event_times_s: List[float],
    threshold_s: float,
) -> Optional[float]:
    return min((t for t in event_times_s if t >= threshold_s), default=None)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
    }


def resolve_adapter_path(
    *,
    adapter_path: Optional[str],
    adapter_repo: str,
    adapter_repo_type: str,
) -> str:
    if adapter_path:
        return adapter_path
    return snapshot_download(
        adapter_repo,
        repo_type=adapter_repo_type,
    )


def build_engine(
    *,
    model_path: str,
    tp_size: int,
    attention_backend: str,
    prefill_attention_backend: str,
    decode_attention_backend: str,
    moe_runner_backend: str,
    disable_cuda_graph: bool,
    experts_shared_outer_loras: bool,
    mem_fraction_static: float,
    weight_sync_consistency_mode: Optional[str] = None,
) -> sgl.Engine:
    if weight_sync_consistency_mode is None:
        os.environ.pop("SGLANG_WEIGHT_SYNC_CONSISTENCY_MODE", None)
    else:
        os.environ["SGLANG_WEIGHT_SYNC_CONSISTENCY_MODE"] = (
            weight_sync_consistency_mode
        )

    engine_kwargs = {
        "model_path": model_path,
        "tp_size": tp_size,
        "attention_backend": attention_backend,
        "prefill_attention_backend": prefill_attention_backend,
        "decode_attention_backend": decode_attention_backend,
        "moe_runner_backend": moe_runner_backend,
        "disable_cuda_graph": disable_cuda_graph,
        "experts_shared_outer_loras": experts_shared_outer_loras,
        "mem_fraction_static": mem_fraction_static,
    }
    server_args_params = inspect.signature(RuntimeServerArgs.__init__).parameters
    filtered_kwargs = {
        key: value for key, value in engine_kwargs.items() if key in server_args_params
    }
    return sgl.Engine(**filtered_kwargs)


def prepare_weight_update(
    engine: sgl.Engine,
    *,
    weights_bytes: bytes,
    loader_metadata: Dict[str, Any],
    flush_cache: bool,
    base_weight_version: Optional[str] = None,
    weight_version: Optional[str] = None,
    transport_format: Optional[str] = None,
    transport_bucket_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    named_tensors_start = time.perf_counter()
    named_tensors = load_named_tensors_from_bytes(weights_bytes)
    decode_done = time.perf_counter()

    request_loader_metadata = dict(loader_metadata)
    request_loader_metadata["synchronize_after_update"] = True
    update_req = build_update_weights_request_from_named_tensors(
        named_tensors,
        tp_size=engine.server_args.tp_size,
        flush_cache=flush_cache,
        transport_format=transport_format,
        transport_bucket_bytes=transport_bucket_bytes,
        base_weight_version=base_weight_version,
        weight_version=weight_version,
        loader_metadata=request_loader_metadata,
    )
    request_done = time.perf_counter()
    return {
        "decode_ms": (decode_done - named_tensors_start) * 1000.0,
        "request_build_ms": (request_done - decode_done) * 1000.0,
        "prepare_ms": (request_done - named_tensors_start) * 1000.0,
        "update_req": update_req,
        "transport_format": getattr(update_req, "transport_format", None),
        "transport_metadata": getattr(update_req, "transport_metadata", None),
    }


def prepare_weight_update_handle_from_bytes(
    engine: sgl.Engine,
    *,
    weights_bytes: bytes,
    loader_metadata: Dict[str, Any],
    flush_cache: bool,
    base_weight_version: Optional[str] = None,
    weight_version: Optional[str] = None,
    transport_format: str = "flattened_bucket",
    transport_bucket_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    request_loader_metadata = dict(loader_metadata)
    request_loader_metadata["synchronize_after_update"] = True
    effective_transport_format = (
        None if transport_format == "serialized_named_tensors" else transport_format
    )
    return engine.loop.run_until_complete(
        engine.tokenizer_manager.prepare_weights_from_bytes(
            weights_bytes,
            flush_cache=flush_cache,
            base_weight_version=base_weight_version,
            weight_version=weight_version,
            loader_metadata=request_loader_metadata,
            transport_format=effective_transport_format,
            transport_bucket_bytes=transport_bucket_bytes,
            request=None,
        )
    )


async def apply_prepared_weight_update_async(
    engine: sgl.Engine,
    *,
    update_req: Any,
    profile_output_dir: Optional[str] = None,
    profile_prefix: Optional[str] = None,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
) -> Dict[str, Any]:
    before_traces = _trace_files(profile_output_dir)
    profile_enabled = profile_output_dir is not None
    if profile_enabled:
        await engine.tokenizer_manager.start_profile(
            output_dir=profile_output_dir,
            activities=profile_activities,
            with_stack=profile_with_stack,
            record_shapes=profile_record_shapes,
            profile_prefix=profile_prefix,
        )

    apply_start = time.perf_counter()
    try:
        success, message = await engine.tokenizer_manager.update_weights_from_tensor(
            update_req, None
        )
    finally:
        if profile_enabled:
            await engine.tokenizer_manager.stop_profile()
    apply_done = time.perf_counter()
    engine_timing = copy.deepcopy(
        getattr(engine.tokenizer_manager, "last_weight_update_timing", None)
    )

    trace_paths: List[Path] = []
    if profile_enabled:
        trace_paths = sorted(_trace_files(profile_output_dir) - before_traces)

    return {
        "success": success,
        "message": message,
        "apply_ms": (apply_done - apply_start) * 1000.0,
        "engine_timing": engine_timing,
        "trace_paths": [str(path) for path in trace_paths],
        "trace_phase_totals_ms": _aggregate_trace_phase_totals(trace_paths),
        "phase_breakdown_ms": _weight_sync_phase_breakdown(
            _aggregate_trace_phase_totals(trace_paths)
        ),
    }


async def apply_prepared_weight_update_handle_async(
    engine: sgl.Engine,
    *,
    update_handle: str,
    profile_output_dir: Optional[str] = None,
    profile_prefix: Optional[str] = None,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
) -> Dict[str, Any]:
    before_traces = _trace_files(profile_output_dir)
    profile_enabled = profile_output_dir is not None
    if profile_enabled:
        await engine.tokenizer_manager.start_profile(
            output_dir=profile_output_dir,
            activities=profile_activities,
            with_stack=profile_with_stack,
            record_shapes=profile_record_shapes,
            profile_prefix=profile_prefix,
        )

    apply_start = time.perf_counter()
    try:
        success, message = await engine.tokenizer_manager.commit_prepared_weight_update(
            update_handle, None
        )
    finally:
        if profile_enabled:
            await engine.tokenizer_manager.stop_profile()
    apply_done = time.perf_counter()
    engine_timing = copy.deepcopy(
        getattr(engine.tokenizer_manager, "last_weight_update_timing", None)
    )

    trace_paths: List[Path] = []
    if profile_enabled:
        trace_paths = sorted(_trace_files(profile_output_dir) - before_traces)

    return {
        "success": success,
        "message": message,
        "apply_ms": (apply_done - apply_start) * 1000.0,
        "engine_timing": engine_timing,
        "trace_paths": [str(path) for path in trace_paths],
        "trace_phase_totals_ms": _aggregate_trace_phase_totals(trace_paths),
        "phase_breakdown_ms": _weight_sync_phase_breakdown(
            _aggregate_trace_phase_totals(trace_paths)
        ),
    }


async def apply_weight_update_with_transport_async(
    engine: sgl.Engine,
    *,
    update_transport: str,
    prepared_update_req: Any,
    weights_bytes: bytes,
    loader_metadata: Dict[str, Any],
    flush_cache: bool,
    base_weight_version: Optional[str],
    weight_version: Optional[str],
    profile_output_dir: Optional[str] = None,
    profile_prefix: Optional[str] = None,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
) -> Dict[str, Any]:
    task_start = time.perf_counter()

    transport_prepare_decode_ms = 0.0
    transport_prepare_request_build_ms = 0.0
    update_req = prepared_update_req
    if update_transport == "prepared_bytes_handle":
        apply_result = await apply_prepared_weight_update_handle_async(
            engine,
            update_handle=prepared_update_req["update_handle"],
            profile_output_dir=profile_output_dir,
            profile_prefix=profile_prefix,
            profile_activities=profile_activities,
            profile_with_stack=profile_with_stack,
            profile_record_shapes=profile_record_shapes,
        )
    elif update_transport in {"inline_bytes", "inline_bytes_to_thread"}:
        prepare_fn = prepare_weight_update
        prepare_args = ()
        prepare_kwargs = {
            "engine": engine,
            "weights_bytes": weights_bytes,
            "loader_metadata": loader_metadata,
            "flush_cache": flush_cache,
            "base_weight_version": base_weight_version,
            "weight_version": weight_version,
        }
        if update_transport == "inline_bytes_to_thread":
            prepared = await asyncio.to_thread(
                prepare_fn,
                *prepare_args,
                **prepare_kwargs,
            )
        else:
            prepared = prepare_fn(*prepare_args, **prepare_kwargs)
        transport_prepare_decode_ms = prepared["decode_ms"]
        transport_prepare_request_build_ms = prepared["request_build_ms"]
        update_req = prepared["update_req"]
    elif update_transport != "prepared_tensor":
        raise ValueError(f"Unsupported update_transport={update_transport!r}")
    else:
        apply_result = await apply_prepared_weight_update_async(
            engine,
            update_req=update_req,
            profile_output_dir=profile_output_dir,
            profile_prefix=profile_prefix,
            profile_activities=profile_activities,
            profile_with_stack=profile_with_stack,
            profile_record_shapes=profile_record_shapes,
        )
    task_end = time.perf_counter()
    return {
        **apply_result,
        "update_transport": update_transport,
        "transport_prepare_decode_ms": transport_prepare_decode_ms,
        "transport_prepare_request_build_ms": transport_prepare_request_build_ms,
        "task_wall_ms": (task_end - task_start) * 1000.0,
    }


def apply_prepared_weight_update(
    engine: sgl.Engine,
    *,
    update_req: Any,
    profile_output_dir: Optional[str] = None,
    profile_prefix: Optional[str] = None,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
) -> Dict[str, Any]:
    return engine.loop.run_until_complete(
        apply_prepared_weight_update_async(
            engine,
            update_req=update_req,
            profile_output_dir=profile_output_dir,
            profile_prefix=profile_prefix,
            profile_activities=profile_activities,
            profile_with_stack=profile_with_stack,
            profile_record_shapes=profile_record_shapes,
        )
    )


def _run_weight_update(
    engine: sgl.Engine,
    *,
    weights_bytes: bytes,
    loader_metadata: Dict[str, Any],
    flush_cache: bool,
    profile_output_dir: Optional[str] = None,
    profile_prefix: Optional[str] = None,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
) -> Dict[str, Any]:
    prepared_update = prepare_weight_update(
        engine,
        weights_bytes=weights_bytes,
        loader_metadata=loader_metadata,
        flush_cache=flush_cache,
    )
    applied_update = apply_prepared_weight_update(
        engine,
        update_req=prepared_update["update_req"],
        profile_output_dir=profile_output_dir,
        profile_prefix=profile_prefix,
        profile_activities=profile_activities,
        profile_with_stack=profile_with_stack,
        profile_record_shapes=profile_record_shapes,
    )
    return {
        "success": applied_update["success"],
        "message": applied_update["message"],
        "decode_ms": prepared_update["decode_ms"],
        "request_build_ms": prepared_update["request_build_ms"],
        "apply_ms": applied_update["apply_ms"],
        "total_ms": (
            prepared_update["decode_ms"]
            + prepared_update["request_build_ms"]
            + applied_update["apply_ms"]
        ),
        "trace_paths": applied_update["trace_paths"],
        "trace_phase_totals_ms": applied_update["trace_phase_totals_ms"],
        "phase_breakdown_ms": applied_update["phase_breakdown_ms"],
    }


def _build_engine(args: argparse.Namespace) -> sgl.Engine:
    return build_engine(
        model_path=args.model_path,
        tp_size=args.tp_size,
        attention_backend=args.attention_backend,
        prefill_attention_backend=args.prefill_attention_backend,
        decode_attention_backend=args.decode_attention_backend,
        moe_runner_backend=args.moe_runner_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        experts_shared_outer_loras=args.experts_shared_outer_loras,
        mem_fraction_static=args.mem_fraction_static,
        weight_sync_consistency_mode=args.weight_sync_consistency_mode,
    )


def run_sync_benchmark(
    engine: sgl.Engine,
    *,
    apply_bytes: bytes,
    revert_bytes: bytes,
    loader_metadata: Dict[str, Any],
    flush_cache: bool,
    warmup_cycles: int,
    cycles: int,
    profile_output_dir: Optional[str] = None,
    profile_cycle: int = 1,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
    compare_sample_path: Optional[str] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    baseline_logprobs: Optional[List[float]] = None
    if compare_sample_path and os.path.exists(compare_sample_path):
        sample = torch.load(compare_sample_path, weights_only=False)
        baseline_logprobs = _get_prompt_logprobs(engine, sample["tokens"])
        summary["correctness_tokens"] = len(sample["tokens"])

    apply_measurements: List[Dict[str, Any]] = []
    revert_measurements: List[Dict[str, Any]] = []
    profiled_trace_paths: List[str] = []
    profiled_phase_totals: Dict[str, float] = {}

    for cycle_idx in range(warmup_cycles + cycles):
        measured = cycle_idx >= warmup_cycles
        is_profile_cycle = (
            profile_output_dir
            and measured
            and (cycle_idx - warmup_cycles + 1) == profile_cycle
        )

        apply_result = _run_weight_update(
            engine,
            weights_bytes=apply_bytes,
            loader_metadata=loader_metadata,
            flush_cache=flush_cache,
            profile_output_dir=profile_output_dir if is_profile_cycle else None,
            profile_prefix="weight-sync-apply" if is_profile_cycle else None,
            profile_activities=profile_activities,
            profile_with_stack=profile_with_stack,
            profile_record_shapes=profile_record_shapes,
        )
        if not apply_result["success"]:
            raise RuntimeError(apply_result["message"])

        if measured:
            apply_measurements.append(apply_result)
            if is_profile_cycle:
                profiled_trace_paths = apply_result["trace_paths"]
                profiled_phase_totals = apply_result["trace_phase_totals_ms"]

        if baseline_logprobs is not None and measured and len(apply_measurements) == 1:
            sample = torch.load(compare_sample_path, weights_only=False)
            updated_logprobs = _get_prompt_logprobs(engine, sample["tokens"])
            deltas = [
                abs(base - updated)
                for base, updated in zip(baseline_logprobs, updated_logprobs)
            ]
            summary["correctness"] = {
                "mean_abs_logprob_delta": statistics.mean(deltas),
                "max_abs_logprob_delta": max(deltas),
                "num_logprobs_compared": len(deltas),
            }

        revert_result = _run_weight_update(
            engine,
            weights_bytes=revert_bytes,
            loader_metadata=loader_metadata,
            flush_cache=flush_cache,
        )
        if not revert_result["success"]:
            raise RuntimeError(revert_result["message"])
        if measured:
            revert_measurements.append(revert_result)

    summary["apply"] = {
        "decode_ms": _stats([m["decode_ms"] for m in apply_measurements]),
        "request_build_ms": _stats([m["request_build_ms"] for m in apply_measurements]),
        "apply_ms": _stats([m["apply_ms"] for m in apply_measurements]),
        "total_ms": _stats([m["total_ms"] for m in apply_measurements]),
    }
    summary["revert"] = {
        "decode_ms": _stats([m["decode_ms"] for m in revert_measurements]),
        "request_build_ms": _stats([m["request_build_ms"] for m in revert_measurements]),
        "apply_ms": _stats([m["apply_ms"] for m in revert_measurements]),
        "total_ms": _stats([m["total_ms"] for m in revert_measurements]),
    }
    summary["profile"] = {
        "trace_paths": profiled_trace_paths,
        "phase_totals_ms_max_across_ranks": profiled_phase_totals,
        "phase_breakdown_ms": _weight_sync_phase_breakdown(profiled_phase_totals),
    }
    return summary


async def _issue_streaming_service_request(
    engine: sgl.Engine,
    *,
    benchmark_start: float,
    request_idx: int,
    prompt: str,
    sampling_params: Dict[str, Any],
    token_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    request_start = time.perf_counter()
    response: Dict[str, Any] = {}
    last_completion_tokens = 0
    try:
        generator = await engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params,
            stream=True,
        )
        async for chunk in generator:
            if not isinstance(chunk, dict):
                continue
            response = chunk
            meta_info = response.get("meta_info", {})
            completion_tokens = int(meta_info.get("completion_tokens", 0) or 0)
            new_tokens = max(completion_tokens - last_completion_tokens, 0)
            if new_tokens > 0:
                token_events.append(
                    {
                        "request_idx": request_idx,
                        "time_s": time.perf_counter() - benchmark_start,
                        "new_tokens": new_tokens,
                        "weight_version_before": meta_info.get("weight_version_before"),
                        "weight_version_after": meta_info.get(
                            "weight_version_after", meta_info.get("weight_version")
                        ),
                        "weight_update_crossed": bool(
                            meta_info.get("weight_update_crossed", False)
                        ),
                    }
                )
                last_completion_tokens = completion_tokens
        success = True
        message = "Success"
    except Exception as exc:
        response = {}
        success = False
        message = str(exc)
    request_end = time.perf_counter()

    meta_info = response.get("meta_info", {}) if isinstance(response, dict) else {}
    return {
        "request_idx": request_idx,
        "start_s": request_start - benchmark_start,
        "end_s": request_end - benchmark_start,
        "latency_ms": (request_end - request_start) * 1000.0,
        "success": success,
        "message": message,
        "completion_tokens": int(meta_info.get("completion_tokens", 0) or 0),
        "weight_version_before": meta_info.get("weight_version_before"),
        "weight_version_after": meta_info.get(
            "weight_version_after", meta_info.get("weight_version")
        ),
        "weight_update_crossed": bool(meta_info.get("weight_update_crossed", False)),
        "weight_update_epoch_before": meta_info.get("weight_update_epoch_before"),
        "weight_update_epoch_after": meta_info.get("weight_update_epoch_after"),
    }


def _completion_silence_around_window_ms(
    completion_times_s: List[float],
    *,
    window_start_s: float,
    window_end_s: float,
) -> Optional[float]:
    last_before = max((t for t in completion_times_s if t <= window_start_s), default=None)
    first_after = min((t for t in completion_times_s if t >= window_end_s), default=None)
    if last_before is None or first_after is None:
        return None
    return (first_after - last_before) * 1000.0


async def _run_service_warmup_async(
    engine: sgl.Engine,
    *,
    total_requests: int,
    concurrency: int,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    if total_requests <= 0 or concurrency <= 0 or max_new_tokens <= 0:
        return {
            "total_requests": 0,
            "concurrency": 0,
            "max_new_tokens": 0,
            "successful_requests": 0,
            "elapsed_ms": 0.0,
        }

    warmup_start = time.perf_counter()
    request_tasks: set[asyncio.Task] = set()
    next_request_idx = 0
    successful_requests = 0

    while next_request_idx < total_requests or request_tasks:
        while next_request_idx < total_requests and len(request_tasks) < concurrency:
            request_tasks.add(
                asyncio.create_task(
                    _issue_streaming_service_request(
                        engine,
                        benchmark_start=warmup_start,
                        request_idx=next_request_idx,
                        prompt=prompt,
                        sampling_params={
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.0,
                        },
                        token_events=[],
                    )
                )
            )
            next_request_idx += 1

        done, _ = await asyncio.wait(request_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            request_tasks.remove(task)
            if task.result().get("success", False):
                successful_requests += 1

    warmup_end = time.perf_counter()
    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "max_new_tokens": max_new_tokens,
        "successful_requests": successful_requests,
        "elapsed_ms": (warmup_end - warmup_start) * 1000.0,
    }


async def _run_service_impact_benchmark_async(
    engine: sgl.Engine,
    *,
    apply_update_req: Any,
    revert_update_req: Any,
    apply_bytes: bytes,
    revert_bytes: bytes,
    loader_metadata: Dict[str, Any],
    flush_cache: bool,
    update_transport: str,
    base_weight_version: str,
    apply_weight_version: str,
    total_requests: int,
    concurrency: int,
    prompt: str,
    max_new_tokens: int,
    update_after_s: float,
    warmup_requests: int,
    warmup_concurrency: int,
    warmup_max_new_tokens: int,
    profile_output_dir: Optional[str] = None,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
) -> Dict[str, Any]:
    benchmark_start = time.perf_counter()
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0.0}

    request_tasks: set[asyncio.Task] = set()
    request_records: List[Dict[str, Any]] = []
    token_events: List[Dict[str, Any]] = []
    next_request_idx = 0

    update_task: Optional[asyncio.Task] = None
    update_result: Optional[Dict[str, Any]] = None
    update_requested_s: Optional[float] = None
    update_completed_s: Optional[float] = None
    update_triggered_under_load = True
    update_deadline_s = update_after_s

    while next_request_idx < total_requests or request_tasks or update_task is not None:
        if (
            update_requested_s is None
            and time.perf_counter() - benchmark_start >= update_deadline_s
        ):
            update_requested_s = time.perf_counter() - benchmark_start
            update_task = asyncio.create_task(
                apply_weight_update_with_transport_async(
                    engine,
                    update_transport=update_transport,
                    prepared_update_req=apply_update_req,
                    weights_bytes=apply_bytes,
                    loader_metadata=loader_metadata,
                    flush_cache=flush_cache,
                    base_weight_version=base_weight_version,
                    weight_version=apply_weight_version,
                    profile_output_dir=profile_output_dir,
                    profile_prefix="weight-sync-service-impact",
                    profile_activities=profile_activities,
                    profile_with_stack=profile_with_stack,
                    profile_record_shapes=profile_record_shapes,
                )
            )

        while next_request_idx < total_requests and len(request_tasks) < concurrency:
            request_tasks.add(
                asyncio.create_task(
                    _issue_streaming_service_request(
                        engine,
                        benchmark_start=benchmark_start,
                        request_idx=next_request_idx,
                        prompt=prompt,
                        sampling_params=sampling_params,
                        token_events=token_events,
                    )
                )
            )
            next_request_idx += 1

        wait_set = set(request_tasks)
        if update_task is not None:
            wait_set.add(update_task)
        if not wait_set:
            break

        wait_timeout: Optional[float] = None
        if update_requested_s is None:
            wait_timeout = max(
                0.0,
                update_deadline_s - (time.perf_counter() - benchmark_start),
            )

        done, _ = await asyncio.wait(
            wait_set,
            timeout=wait_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            continue
        completed_update_task = update_task if update_task in done else None

        for task in done:
            if task is completed_update_task:
                continue
            request_tasks.remove(task)
            request_records.append(task.result())

        if completed_update_task is not None:
            update_result = completed_update_task.result()
            update_completed_s = time.perf_counter() - benchmark_start
            update_task = None

    if update_requested_s is None:
        update_triggered_under_load = False
        update_requested_s = time.perf_counter() - benchmark_start
        update_result = await apply_weight_update_with_transport_async(
            engine,
            update_transport=update_transport,
            prepared_update_req=apply_update_req,
            weights_bytes=apply_bytes,
            loader_metadata=loader_metadata,
            flush_cache=flush_cache,
            base_weight_version=base_weight_version,
            weight_version=apply_weight_version,
            profile_output_dir=profile_output_dir,
            profile_prefix="weight-sync-service-impact",
            profile_activities=profile_activities,
            profile_with_stack=profile_with_stack,
            profile_record_shapes=profile_record_shapes,
        )
        update_completed_s = time.perf_counter() - benchmark_start
    elif update_task is not None:
        update_result = await update_task
        update_completed_s = time.perf_counter() - benchmark_start

    assert update_result is not None
    assert update_requested_s is not None
    assert update_completed_s is not None

    successful_requests = [record for record in request_records if record["success"]]
    completion_times_s = sorted(record["end_s"] for record in successful_requests)
    completion_silence_ms = _completion_silence_around_window_ms(
        completion_times_s,
        window_start_s=update_requested_s,
        window_end_s=update_completed_s,
    )
    token_event_times_s = sorted(event["time_s"] for event in token_events)
    token_silence_ms = _completion_silence_around_window_ms(
        token_event_times_s,
        window_start_s=update_requested_s,
        window_end_s=update_completed_s,
    )
    first_token_after_update_requested_s = _first_event_after_s(
        token_event_times_s, update_requested_s
    )
    first_token_after_update_completed_s = _first_event_after_s(
        token_event_times_s, update_completed_s
    )
    crossed_token_times_s = sorted(
        event["time_s"] for event in token_events if event["weight_update_crossed"]
    )
    first_crossed_token_after_update_requested_s = _first_event_after_s(
        crossed_token_times_s, update_requested_s
    )
    first_crossed_token_after_update_completed_s = _first_event_after_s(
        crossed_token_times_s, update_completed_s
    )

    revert_result: Dict[str, Any]
    if update_result["success"]:
        revert_result = await apply_weight_update_with_transport_async(
            engine,
            update_transport=update_transport,
            prepared_update_req=revert_update_req,
            weights_bytes=revert_bytes,
            loader_metadata=loader_metadata,
            flush_cache=flush_cache,
            base_weight_version=apply_weight_version,
            weight_version=base_weight_version,
        )
    else:
        revert_result = {
            "success": False,
            "message": "Skipped revert because apply failed.",
            "apply_ms": 0.0,
            "task_wall_ms": 0.0,
            "engine_timing": None,
            "trace_paths": [],
            "trace_phase_totals_ms": {},
            "phase_breakdown_ms": {},
            "update_transport": update_transport,
            "transport_prepare_decode_ms": 0.0,
            "transport_prepare_request_build_ms": 0.0,
        }

    benchmark_end_s = time.perf_counter() - benchmark_start
    request_latencies_ms = [record["latency_ms"] for record in successful_requests]
    total_completion_tokens = sum(
        record["completion_tokens"] for record in successful_requests
    )
    total_streamed_tokens = sum(
        int(event.get("new_tokens", 0) or 0) for event in token_events
    )

    return {
        "request_load": {
            "total_requests": total_requests,
            "concurrency": concurrency,
            "successful_requests": len(successful_requests),
            "failed_requests": len(request_records) - len(successful_requests),
            "latency_ms": _stats(request_latencies_ms),
            "completion_silence_around_update_ms": completion_silence_ms,
            "requests_dispatched_before_update": sum(
                record["start_s"] < update_requested_s for record in request_records
            ),
            "requests_dispatched_during_update": sum(
                update_requested_s <= record["start_s"] < update_completed_s
                for record in request_records
            ),
            "requests_completed_during_update": sum(
                update_requested_s <= record["end_s"] < update_completed_s
                for record in request_records
            ),
            "requests_crossing_update_window": sum(
                record["start_s"] < update_requested_s
                and record["end_s"] >= update_completed_s
                for record in request_records
            ),
            "requests_reporting_weight_update_crossed": sum(
                record.get("weight_update_crossed", False)
                for record in successful_requests
            ),
            "weight_version_transition_counts": _weight_version_transition_counts(
                successful_requests
            ),
            "completion_tokens": total_completion_tokens,
            "completion_tokens_per_s_overall": (
                total_completion_tokens / benchmark_end_s if benchmark_end_s > 0 else 0.0
            ),
            "benchmark_elapsed_ms": benchmark_end_s * 1000.0,
        },
        "token_stream": {
            "token_silence_around_update_ms": token_silence_ms,
            "total_token_events": len(token_events),
            "total_streamed_tokens": total_streamed_tokens,
            "token_events_during_update": sum(
                update_requested_s <= event["time_s"] < update_completed_s
                for event in token_events
            ),
            "tokens_emitted_during_update": sum(
                int(event.get("new_tokens", 0) or 0)
                for event in token_events
                if update_requested_s <= event["time_s"] < update_completed_s
            ),
            "tokens_reporting_weight_update_crossed": sum(
                int(event.get("new_tokens", 0) or 0)
                for event in token_events
                if event["weight_update_crossed"]
            ),
            "first_token_after_update_requested_ms": (
                (first_token_after_update_requested_s - update_requested_s) * 1000.0
                if first_token_after_update_requested_s is not None
                else None
            ),
            "first_token_after_update_completed_ms": (
                (first_token_after_update_completed_s - update_completed_s) * 1000.0
                if first_token_after_update_completed_s is not None
                else None
            ),
            "first_crossed_token_after_update_requested_ms": (
                (
                    first_crossed_token_after_update_requested_s - update_requested_s
                )
                * 1000.0
                if first_crossed_token_after_update_requested_s is not None
                else None
            ),
            "first_crossed_token_after_update_completed_ms": (
                (
                    first_crossed_token_after_update_completed_s - update_completed_s
                )
                * 1000.0
                if first_crossed_token_after_update_completed_s is not None
                else None
            ),
            "weight_version_transition_token_counts": _weight_version_transition_token_counts(
                token_events
            ),
        },
        "update": {
            "success": update_result["success"],
            "message": update_result["message"],
            "triggered_under_load": update_triggered_under_load,
            "requested_after_start_ms": update_requested_s * 1000.0,
            "completed_after_start_ms": update_completed_s * 1000.0,
            "update_transport": update_result["update_transport"],
            "task_wall_ms": update_result["task_wall_ms"],
            "request_wall_ms": update_result["apply_ms"],
            "engine_timing": update_result.get("engine_timing"),
            "transport_prepare_decode_ms": update_result["transport_prepare_decode_ms"],
            "transport_prepare_request_build_ms": update_result[
                "transport_prepare_request_build_ms"
            ],
            "trace_paths": update_result["trace_paths"],
            "trace_phase_totals_ms": update_result["trace_phase_totals_ms"],
            "phase_breakdown_ms": update_result["phase_breakdown_ms"],
        },
        "revert_after_update": {
            "success": revert_result["success"],
            "message": revert_result["message"],
            "request_wall_ms": revert_result["apply_ms"],
            "task_wall_ms": revert_result["task_wall_ms"],
            "engine_timing": revert_result.get("engine_timing"),
            "update_transport": revert_result["update_transport"],
            "transport_prepare_decode_ms": revert_result["transport_prepare_decode_ms"],
            "transport_prepare_request_build_ms": revert_result[
                "transport_prepare_request_build_ms"
            ],
        },
    }


def run_service_impact_benchmark(
    engine: sgl.Engine,
    *,
    apply_bytes: bytes,
    revert_bytes: bytes,
    loader_metadata: Dict[str, Any],
    flush_cache: bool,
    update_transport: str,
    total_requests: int,
    concurrency: int,
    prompt: str,
    max_new_tokens: int,
    update_after_s: float,
    warmup_requests: int,
    warmup_concurrency: int,
    warmup_max_new_tokens: int,
    prepared_bytes_transport_format: str,
    prepared_bytes_transport_bucket_bytes: Optional[int],
    profile_output_dir: Optional[str] = None,
    profile_activities: Optional[List[str]] = None,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
) -> Dict[str, Any]:
    base_weight_version = getattr(engine.server_args, "weight_version", "default")
    apply_weight_version = f"lora-sync-bench-{_sha256_hex(apply_bytes)[:8]}"
    prepared_handle_ids: List[str] = []
    try:
        if update_transport == "prepared_bytes_handle":
            prepared_apply = prepare_weight_update_handle_from_bytes(
                engine,
                weights_bytes=apply_bytes,
                loader_metadata=loader_metadata,
                flush_cache=flush_cache,
                base_weight_version=base_weight_version,
                weight_version=apply_weight_version,
                transport_format=prepared_bytes_transport_format,
                transport_bucket_bytes=prepared_bytes_transport_bucket_bytes,
            )
            prepared_handle_ids.append(prepared_apply["update_handle"])
            prepared_revert = prepare_weight_update_handle_from_bytes(
                engine,
                weights_bytes=revert_bytes,
                loader_metadata=loader_metadata,
                flush_cache=flush_cache,
                base_weight_version=apply_weight_version,
                weight_version=base_weight_version,
                transport_format=prepared_bytes_transport_format,
                transport_bucket_bytes=prepared_bytes_transport_bucket_bytes,
            )
            prepared_handle_ids.append(prepared_revert["update_handle"])
        else:
            prepared_apply = prepare_weight_update(
                engine,
                weights_bytes=apply_bytes,
                loader_metadata=loader_metadata,
                flush_cache=flush_cache,
                base_weight_version=base_weight_version,
                weight_version=apply_weight_version,
            )
            prepared_revert = prepare_weight_update(
                engine,
                weights_bytes=revert_bytes,
                loader_metadata=loader_metadata,
                flush_cache=flush_cache,
                base_weight_version=apply_weight_version,
                weight_version=base_weight_version,
            )
        warmup_summary = engine.loop.run_until_complete(
            _run_service_warmup_async(
                engine,
                total_requests=warmup_requests,
                concurrency=warmup_concurrency,
                prompt=prompt,
                max_new_tokens=warmup_max_new_tokens,
            )
        )
        summary = engine.loop.run_until_complete(
            _run_service_impact_benchmark_async(
                engine,
                apply_update_req=prepared_apply
                if update_transport == "prepared_bytes_handle"
                else prepared_apply["update_req"],
                revert_update_req=prepared_revert
                if update_transport == "prepared_bytes_handle"
                else prepared_revert["update_req"],
                apply_bytes=apply_bytes,
                revert_bytes=revert_bytes,
                loader_metadata=loader_metadata,
                flush_cache=flush_cache,
                update_transport=update_transport,
                base_weight_version=base_weight_version,
                apply_weight_version=apply_weight_version,
                total_requests=total_requests,
                concurrency=concurrency,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                update_after_s=update_after_s,
                warmup_requests=warmup_requests,
                warmup_concurrency=warmup_concurrency,
                warmup_max_new_tokens=warmup_max_new_tokens,
                profile_output_dir=profile_output_dir,
                profile_activities=profile_activities,
                profile_with_stack=profile_with_stack,
                profile_record_shapes=profile_record_shapes,
            )
        )
        summary["prepared_apply"] = prepared_apply.get("prepare_stats", prepared_apply)
        summary["prepared_revert"] = prepared_revert.get(
            "prepare_stats", prepared_revert
        )
        summary["warmup"] = warmup_summary
        summary["weight_versions"] = {
            "base_weight_version": base_weight_version,
            "apply_weight_version": apply_weight_version,
        }
        summary["service_impact_update_transport"] = update_transport
        return summary
    finally:
        for update_handle in prepared_handle_ids:
            engine.loop.run_until_complete(
                engine.tokenizer_manager.discard_prepared_weight_update(
                    update_handle, None
                )
            )


def main() -> None:
    args = _parse_args()
    adapter_path = _resolve_adapter_path(args)
    named_tensors, loader_metadata, apply_bytes, revert_bytes, skipped = _load_adapter_payload(
        adapter_path,
        adapter_safetensors=args.adapter_safetensors,
        adapter_config=args.adapter_config,
        allow_skipped=args.allow_skipped,
    )

    engine = _build_engine(args)
    summary: Dict[str, Any] = {
        "benchmark_mode": args.benchmark_mode,
        "model_path": args.model_path,
        "adapter_path": adapter_path,
        "tp_size": args.tp_size,
        "payload_bytes": len(apply_bytes),
        "payload_sha256": _sha256_hex(apply_bytes),
        "named_tensor_count": len(named_tensors),
        "target_count": len(loader_metadata.get("targets", [])),
        "skipped_tensor_count": len(skipped),
        "skipped_tensor_names": skipped,
        "flush_cache": args.flush_cache,
    }

    try:
        compare_sample_path = (
            os.path.join(adapter_path, "compare_sample_train_data.pt")
            if args.check_correctness
            else None
        )
        if args.benchmark_mode == "sync":
            summary.update(
                run_sync_benchmark(
                    engine,
                    apply_bytes=apply_bytes,
                    revert_bytes=revert_bytes,
                    loader_metadata=loader_metadata,
                    flush_cache=args.flush_cache,
                    warmup_cycles=args.warmup_cycles,
                    cycles=args.cycles,
                    profile_output_dir=args.profile_output_dir,
                    profile_cycle=args.profile_cycle,
                    profile_activities=args.profile_activities,
                    profile_with_stack=args.profile_with_stack,
                    profile_record_shapes=args.profile_record_shapes,
                    compare_sample_path=compare_sample_path,
                )
            )
        else:
            summary.update(
                run_service_impact_benchmark(
                    engine,
                    apply_bytes=apply_bytes,
                    revert_bytes=revert_bytes,
                    loader_metadata=loader_metadata,
                    flush_cache=args.flush_cache,
                    update_transport=args.service_impact_update_transport,
                    total_requests=args.service_impact_requests,
                    concurrency=args.service_impact_concurrency,
                    prompt=args.service_impact_prompt,
                    max_new_tokens=args.service_impact_max_new_tokens,
                    update_after_s=args.service_impact_update_after_s,
                    warmup_requests=args.service_impact_warmup_requests,
                    warmup_concurrency=args.service_impact_warmup_concurrency,
                    warmup_max_new_tokens=args.service_impact_warmup_max_new_tokens,
                    prepared_bytes_transport_format=args.prepared_bytes_transport_format,
                    prepared_bytes_transport_bucket_bytes=args.prepared_bytes_transport_bucket_bytes,
                    profile_output_dir=(
                        args.profile_output_dir
                        if args.service_impact_profile_update
                        else None
                    ),
                    profile_activities=args.profile_activities,
                    profile_with_stack=args.profile_with_stack,
                    profile_record_shapes=args.profile_record_shapes,
                )
            )
    finally:
        engine.shutdown()

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
