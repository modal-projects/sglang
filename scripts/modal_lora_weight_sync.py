import io
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

import modal
from modal_proto import api_pb2

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_ADAPTER_REPO = "yushengsu/lora-diff-Qwen3-VL-30B-A3B-Instruct"
DEFAULT_BASE_IMAGE = os.environ.get(
    "SGLANG_MODAL_BASE_IMAGE",
    "lmsysorg/sglang:nightly-dev-20260329-3ab9afd6",
)
DEFAULT_OUTPUT_VOLUME = os.environ.get(
    "SGLANG_MODAL_OUTPUT_VOLUME",
    "sglang-lora-weight-sync-artifacts-v2",
)
DEFAULT_HF_CACHE_VOLUME = os.environ.get(
    "SGLANG_MODAL_HF_CACHE_VOLUME",
    "sglang-huggingface-cache-v2",
)
DEFAULT_ENVIRONMENT_NAME = os.environ.get("MODAL_ENVIRONMENT", "")
DEFAULT_VOLUME_VERSION = int(os.environ.get("SGLANG_MODAL_VOLUME_VERSION", "2"))
ARTIFACTS_MOUNT_PATH = "/artifacts"
HF_CACHE_MOUNT_PATH = "/cache"
REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = "/sgl-workspace/sglang"
SANDBOX_TIMEOUT_S = 12 * 60 * 60
COMMAND_TIMEOUT_S = 10 * 60 * 60
SYNC_TIMEOUT_S = 30 * 60

OVERLAY_RELATIVE_PATHS = [
    Path("python/sglang/srt/configs/utils.py"),
    Path("python/sglang/srt/managers/scheduler_update_weights_mixin.py"),
    Path("python/sglang/srt/managers/tokenizer_communicator_mixin.py"),
    Path("python/sglang/srt/managers/tokenizer_manager.py"),
    Path("python/sglang/srt/managers/tp_worker.py"),
    Path("python/sglang/srt/model_loader/weight_utils.py"),
    Path("python/sglang/srt/models/transformers.py"),
    Path("python/sglang/srt/weight_sync/lora_merge_loader.py"),
    Path("python/sglang/srt/weight_sync/lora_payload_utils.py"),
    Path("python/sglang/srt/weight_sync/tensor_bucket.py"),
    Path("python/sglang/srt/weight_sync/update_bytes.py"),
    Path("python/sglang/test/weight_sync_lora_bench.py"),
]

app = modal.App("sglang-lora-weight-sync")


def _base_image_needs_transformers_upgrade(base_image: str) -> bool:
    return "nightly-dev-" not in base_image


def _build_image() -> modal.Image:
    built_image = modal.Image.from_registry(DEFAULT_BASE_IMAGE)
    if _base_image_needs_transformers_upgrade(DEFAULT_BASE_IMAGE):
        built_image = built_image.pip_install("transformers==5.3.0")
    built_image = built_image.env(
        {
            "HF_HOME": f"{HF_CACHE_MOUNT_PATH}/huggingface",
            "HF_HUB_CACHE": f"{HF_CACHE_MOUNT_PATH}/huggingface/hub",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    ).workdir(REMOTE_REPO_ROOT)
    for relative_path in OVERLAY_RELATIVE_PATHS:
        built_image = built_image.add_local_file(
            REPO_ROOT / relative_path,
            f"{REMOTE_REPO_ROOT}/{relative_path.as_posix()}",
            copy=False,
        )
    return built_image


image = _build_image()


def _volume_from_name(
    name: str,
    *,
    environment_name: str,
    version: int,
) -> modal.Volume:
    kwargs = {"create_if_missing": True}
    if environment_name:
        kwargs["environment_name"] = environment_name
    kwargs["version"] = version
    return modal.Volume.from_name(name, **kwargs)


def _default_run_name(*, tp_size: int, gpu: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    normalized_gpu = re.sub(r"[^a-zA-Z0-9]+", "-", gpu).strip("-").lower()
    return f"{timestamp}-tp{tp_size}-{normalized_gpu or 'gpu'}"


def _parse_region(region: str) -> Optional[str | list[str]]:
    if not region:
        return None
    parts = [item.strip() for item in region.split(",") if item.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


def _read_volume_text(volume: modal.Volume, path: str) -> str:
    buffer = io.BytesIO()
    volume.read_file_into_fileobj(path, buffer)
    return buffer.getvalue().decode("utf-8")


def _format_process_output(output: str, *, limit_lines: int = 120) -> str:
    lines = output.splitlines()
    if len(lines) <= limit_lines:
        return output
    return "\n".join(lines[-limit_lines:])


def _stream_subprocess_output(stream) -> str:
    output_chunks: list[str] = []
    while True:
        line = stream.readline()
        if line == "":
            break
        print(line, end="", flush=True)
        output_chunks.append(line)
    return "".join(output_chunks)


def _build_benchmark_command(
    *,
    benchmark_mode: str,
    model_path: str,
    adapter_path: str,
    adapter_repo: str,
    adapter_repo_type: str,
    adapter_safetensors: str,
    adapter_config: str,
    tp_size: int,
    warmup_cycles: int,
    cycles: int,
    flush_cache: bool,
    allow_skipped: bool,
    check_correctness: bool,
    profile: bool,
    profile_with_stack: bool,
    profile_record_shapes: bool,
    profile_activities: str,
    profile_cycle: int,
    disable_cuda_graph: bool,
    attention_backend: str,
    prefill_attention_backend: str,
    decode_attention_backend: str,
    moe_runner_backend: str,
    experts_shared_outer_loras: bool,
    mem_fraction_static: float,
    weight_sync_consistency_mode: str,
    service_impact_requests: int,
    service_impact_concurrency: int,
    service_impact_prompt: str,
    service_impact_max_new_tokens: int,
    service_impact_update_after_s: float,
    service_impact_profile_update: bool,
    service_impact_update_transport: str,
    service_impact_warmup_requests: int,
    service_impact_warmup_concurrency: int,
    service_impact_warmup_max_new_tokens: int,
    prepared_bytes_transport_format: str,
    prepared_bytes_transport_bucket_bytes: int,
    output_root: str,
    extra_bench_args: str,
) -> list[str]:
    summary_path = f"{output_root}/summary.json"
    command = [
        "python",
        "-m",
        "sglang.test.weight_sync_lora_bench",
        "--benchmark-mode",
        benchmark_mode,
        "--model-path",
        model_path,
        "--tp-size",
        str(tp_size),
        "--warmup-cycles",
        str(warmup_cycles),
        "--cycles",
        str(cycles),
        "--adapter-safetensors",
        adapter_safetensors,
        "--adapter-config",
        adapter_config,
        "--attention-backend",
        attention_backend,
        "--prefill-attention-backend",
        prefill_attention_backend,
        "--decode-attention-backend",
        decode_attention_backend,
        "--moe-runner-backend",
        moe_runner_backend,
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--profile-cycle",
        str(profile_cycle),
        "--output-json",
        summary_path,
    ]
    if adapter_path:
        command.extend(["--adapter-path", adapter_path])
    else:
        command.extend(["--adapter-repo", adapter_repo, "--adapter-repo-type", adapter_repo_type])
    if flush_cache:
        command.append("--flush-cache")
    if allow_skipped:
        command.append("--allow-skipped")
    if check_correctness:
        command.append("--check-correctness")
    command.append("--disable-cuda-graph" if disable_cuda_graph else "--no-disable-cuda-graph")
    command.append(
        "--experts-shared-outer-loras"
        if experts_shared_outer_loras
        else "--no-experts-shared-outer-loras"
    )
    if weight_sync_consistency_mode:
        command.extend(
            ["--weight-sync-consistency-mode", weight_sync_consistency_mode]
        )
    if benchmark_mode == "service-impact":
        command.extend(
            [
                "--service-impact-requests",
                str(service_impact_requests),
                "--service-impact-concurrency",
                str(service_impact_concurrency),
                "--service-impact-prompt",
                service_impact_prompt,
                "--service-impact-max-new-tokens",
                str(service_impact_max_new_tokens),
                "--service-impact-update-after-s",
                str(service_impact_update_after_s),
                "--service-impact-update-transport",
                service_impact_update_transport,
                "--service-impact-warmup-requests",
                str(service_impact_warmup_requests),
                "--service-impact-warmup-concurrency",
                str(service_impact_warmup_concurrency),
                "--service-impact-warmup-max-new-tokens",
                str(service_impact_warmup_max_new_tokens),
                "--prepared-bytes-transport-format",
                prepared_bytes_transport_format,
                "--prepared-bytes-transport-bucket-bytes",
                str(prepared_bytes_transport_bucket_bytes),
            ]
        )
        if service_impact_profile_update:
            command.append("--service-impact-profile-update")
    if profile:
        command.extend(
            [
                "--profile-output-dir",
                f"{output_root}/profile",
                "--profile-activities",
                *[activity.strip() for activity in profile_activities.split(",") if activity.strip()],
            ]
        )
        if profile_with_stack:
            command.append("--profile-with-stack")
        if profile_record_shapes:
            command.append("--profile-record-shapes")
    if extra_bench_args:
        command.extend(shlex.split(extra_bench_args))
    return command


def _artifact_listing(volume: modal.Volume, run_dir: str) -> list[str]:
    return sorted(entry.path for entry in volume.listdir(run_dir, recursive=True))


def _should_sync_volume(sync_requested: bool, volume_version: int) -> bool:
    return sync_requested and volume_version == api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_V2


def _profile_activities_list(profile_activities: str) -> list[str]:
    return [activity.strip() for activity in profile_activities.split(",") if activity.strip()]


def _write_summary_file(output_root: str, summary: dict[str, object]) -> Path:
    artifact_root = Path(output_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path


def _artifact_paths_for_output_root(output_root: str) -> list[str]:
    artifact_root = Path(output_root)
    return sorted(
        str(path.relative_to(ARTIFACTS_MOUNT_PATH))
        for path in artifact_root.rglob("*")
        if path.is_file()
    )


def _sync_volume_mount(sandbox: modal.Sandbox, mount_path: str) -> dict[str, object]:
    proc = sandbox.exec("sync", mount_path, timeout=SYNC_TIMEOUT_S)
    return_code = proc.wait()
    stdout = proc.stdout.read() or ""
    stderr = proc.stderr.read() or ""
    return {
        "mount_path": mount_path,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "ok": return_code == 0,
    }


DEFAULT_DEV_ARTIFACTS_VOLUME = _volume_from_name(
    DEFAULT_OUTPUT_VOLUME,
    environment_name=DEFAULT_ENVIRONMENT_NAME,
    version=DEFAULT_VOLUME_VERSION,
)
DEFAULT_DEV_HF_CACHE_VOLUME = _volume_from_name(
    DEFAULT_HF_CACHE_VOLUME,
    environment_name=DEFAULT_ENVIRONMENT_NAME,
    version=DEFAULT_VOLUME_VERSION,
)


@app.function(
    image=image,
    gpu="a100",
    cpu=8,
    memory=65536,
    timeout=SANDBOX_TIMEOUT_S,
    volumes={
        ARTIFACTS_MOUNT_PATH: DEFAULT_DEV_ARTIFACTS_VOLUME,
        HF_CACHE_MOUNT_PATH: DEFAULT_DEV_HF_CACHE_VOLUME,
    },
)
def dev_shell_spec() -> None:
    pass


@app.cls(
    image=image,
    timeout=SANDBOX_TIMEOUT_S,
)
class WeightSyncBenchRunner:
    @modal.method()
    def run(
        self,
        *,
        command: list[str],
        run_dir: str,
        output_root: str,
        volume_name: str,
        hf_cache_volume_name: str,
        environment_name: str,
        volume_version: int,
        sync_artifacts: bool,
        sync_hf_cache: bool,
        cloud: str,
        region: Optional[str | list[str]],
        gpu: str,
    ) -> dict[str, object]:
        print(f"Running: {' '.join(command)}", flush=True)
        proc = subprocess.Popen(
            command,
            cwd=REMOTE_REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None

        output = ""
        try:
            output = _stream_subprocess_output(proc.stdout)
            return_code = proc.wait()
        finally:
            if sync_artifacts:
                _volume_from_name(
                    volume_name,
                    environment_name=environment_name,
                    version=volume_version,
                ).commit()
            if sync_hf_cache:
                _volume_from_name(
                    hf_cache_volume_name,
                    environment_name=environment_name,
                    version=volume_version,
                ).commit()

        result: dict[str, object] = {
            "base_image": DEFAULT_BASE_IMAGE,
            "cloud": cloud,
            "command": command,
            "environment_name": environment_name or None,
            "gpu": gpu,
            "hf_cache_volume_name": hf_cache_volume_name,
            "region": region,
            "run_dir": run_dir,
            "summary_path": f"{run_dir}/summary.json",
            "volume_name": volume_name,
            "volume_version": volume_version,
        }
        if output.strip():
            result["stdout_tail"] = _format_process_output(output)

        if return_code != 0:
            result["return_code"] = return_code
            raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))

        summary_file = Path(output_root) / "summary.json"
        if not summary_file.exists():
            result["artifact_error"] = f"missing summary at {summary_file}"
            raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))

        artifact_root = Path(output_root)
        result["artifact_paths"] = sorted(
            str(path.relative_to(ARTIFACTS_MOUNT_PATH))
            for path in artifact_root.rglob("*")
            if path.is_file()
        )
        result["summary"] = json.loads(summary_file.read_text())
        return result


@app.cls(
    image=image,
    timeout=SANDBOX_TIMEOUT_S,
    scaledown_window=30 * 60,
    max_containers=1,
)
class PersistentWeightSyncBenchRunner:
    engine_config_json: str = modal.parameter()
    adapter_config_json: str = modal.parameter()

    @modal.enter()
    def startup(self) -> None:
        from sglang.test import weight_sync_lora_bench as bench_mod

        self._bench = bench_mod
        self.engine_config = json.loads(self.engine_config_json)
        self.adapter_config = json.loads(self.adapter_config_json)
        self.adapter_path = bench_mod.resolve_adapter_path(
            adapter_path=self.adapter_config["adapter_path"] or None,
            adapter_repo=self.adapter_config["adapter_repo"],
            adapter_repo_type=self.adapter_config["adapter_repo_type"],
        )
        (
            named_tensors,
            self.loader_metadata,
            self.apply_bytes,
            self.revert_bytes,
            self.skipped_tensor_names,
        ) = bench_mod._load_adapter_payload(
            self.adapter_path,
            adapter_safetensors=self.adapter_config["adapter_safetensors"],
            adapter_config=self.adapter_config["adapter_config"],
            allow_skipped=self.adapter_config["allow_skipped"],
        )
        self.engine = bench_mod.build_engine(**self.engine_config)
        self.compare_sample_path = os.path.join(
            self.adapter_path, "compare_sample_train_data.pt"
        )
        self.payload_metadata = {
            "payload_bytes": len(self.apply_bytes),
            "payload_sha256": bench_mod._sha256_hex(self.apply_bytes),
            "named_tensor_count": len(named_tensors),
            "target_count": len(self.loader_metadata.get("targets", [])),
            "skipped_tensor_count": len(self.skipped_tensor_names),
            "skipped_tensor_names": self.skipped_tensor_names,
        }
        print(
            json.dumps(
                {
                    "event": "persistent-runner-ready",
                    "model_path": self.engine_config["model_path"],
                    "adapter_path": self.adapter_path,
                    "tp_size": self.engine_config["tp_size"],
                    "weight_sync_consistency_mode": self.engine_config.get(
                        "weight_sync_consistency_mode"
                    ),
                    **self.payload_metadata,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    @modal.exit()
    def shutdown(self) -> None:
        if hasattr(self, "engine"):
            self.engine.shutdown()

    def _commit_volumes(
        self,
        *,
        volume_name: str,
        hf_cache_volume_name: str,
        environment_name: str,
        volume_version: int,
        sync_artifacts: bool,
        sync_hf_cache: bool,
    ) -> None:
        if sync_artifacts:
            _volume_from_name(
                volume_name,
                environment_name=environment_name,
                version=volume_version,
            ).commit()
        if sync_hf_cache:
            _volume_from_name(
                hf_cache_volume_name,
                environment_name=environment_name,
                version=volume_version,
            ).commit()

    def _base_summary(self, *, benchmark_mode: str, flush_cache: bool) -> dict[str, object]:
        return {
            "benchmark_mode": benchmark_mode,
            "model_path": self.engine_config["model_path"],
            "adapter_path": self.adapter_path,
            "tp_size": self.engine_config["tp_size"],
            "flush_cache": flush_cache,
            "weight_sync_consistency_mode": self.engine_config.get(
                "weight_sync_consistency_mode"
            ),
            **self.payload_metadata,
        }

    @modal.method()
    def run_benchmark(
        self,
        *,
        benchmark_mode: str,
        run_dir: str,
        output_root: str,
        volume_name: str,
        hf_cache_volume_name: str,
        environment_name: str,
        volume_version: int,
        sync_artifacts: bool,
        sync_hf_cache: bool,
        cloud: str,
        region: Optional[str | list[str]],
        gpu: str,
        flush_cache: bool,
        warmup_cycles: int,
        cycles: int,
        check_correctness: bool,
        profile: bool,
        profile_with_stack: bool,
        profile_record_shapes: bool,
        profile_activities: str,
        profile_cycle: int,
        service_impact_requests: int,
        service_impact_concurrency: int,
        service_impact_prompt: str,
        service_impact_max_new_tokens: int,
        service_impact_update_after_s: float,
        service_impact_profile_update: bool,
        service_impact_update_transport: str,
        service_impact_warmup_requests: int,
        service_impact_warmup_concurrency: int,
        service_impact_warmup_max_new_tokens: int,
        prepared_bytes_transport_format: str,
        prepared_bytes_transport_bucket_bytes: int,
    ) -> dict[str, object]:
        summary = self._base_summary(
            benchmark_mode=benchmark_mode,
            flush_cache=flush_cache,
        )
        profile_output_dir = f"{output_root}/profile" if profile else None
        try:
            if benchmark_mode == "sync":
                summary.update(
                    self._bench.run_sync_benchmark(
                        self.engine,
                        apply_bytes=self.apply_bytes,
                        revert_bytes=self.revert_bytes,
                        loader_metadata=self.loader_metadata,
                        flush_cache=flush_cache,
                        warmup_cycles=warmup_cycles,
                        cycles=cycles,
                        profile_output_dir=profile_output_dir,
                        profile_cycle=profile_cycle,
                        profile_activities=_profile_activities_list(profile_activities),
                        profile_with_stack=profile_with_stack,
                        profile_record_shapes=profile_record_shapes,
                        compare_sample_path=(
                            self.compare_sample_path if check_correctness else None
                        ),
                    )
                )
            elif benchmark_mode == "service-impact":
                summary.update(
                    self._bench.run_service_impact_benchmark(
                        self.engine,
                        apply_bytes=self.apply_bytes,
                        revert_bytes=self.revert_bytes,
                        loader_metadata=self.loader_metadata,
                        flush_cache=flush_cache,
                        total_requests=service_impact_requests,
                        concurrency=service_impact_concurrency,
                        prompt=service_impact_prompt,
                        max_new_tokens=service_impact_max_new_tokens,
                        update_after_s=service_impact_update_after_s,
                        warmup_requests=service_impact_warmup_requests,
                        warmup_concurrency=service_impact_warmup_concurrency,
                        warmup_max_new_tokens=service_impact_warmup_max_new_tokens,
                        prepared_bytes_transport_format=prepared_bytes_transport_format,
                        prepared_bytes_transport_bucket_bytes=prepared_bytes_transport_bucket_bytes,
                        update_transport=service_impact_update_transport,
                        profile_output_dir=(
                            profile_output_dir
                            if service_impact_profile_update
                            else None
                        ),
                        profile_activities=_profile_activities_list(profile_activities),
                        profile_with_stack=profile_with_stack,
                        profile_record_shapes=profile_record_shapes,
                    )
                )
            else:
                raise ValueError(f"Unsupported benchmark_mode={benchmark_mode!r}")

            _write_summary_file(output_root, summary)
            result: dict[str, object] = {
                "base_image": DEFAULT_BASE_IMAGE,
                "cloud": cloud,
                "environment_name": environment_name or None,
                "gpu": gpu,
                "hf_cache_volume_name": hf_cache_volume_name,
                "region": region,
                "run_dir": run_dir,
                "summary_path": f"{run_dir}/summary.json",
                "volume_name": volume_name,
                "volume_version": volume_version,
                "artifact_paths": _artifact_paths_for_output_root(output_root),
                "summary": summary,
            }
            return result
        finally:
            self._commit_volumes(
                volume_name=volume_name,
                hf_cache_volume_name=hf_cache_volume_name,
                environment_name=environment_name,
                volume_version=volume_version,
                sync_artifacts=sync_artifacts,
                sync_hf_cache=sync_hf_cache,
            )


@app.local_entrypoint()
def bench(
    runner_mode: str = "persistent",
    gpu: str = "a100",
    cloud: str = "auto",
    region: str = "",
    environment_name: str = DEFAULT_ENVIRONMENT_NAME,
    volume_name: str = DEFAULT_OUTPUT_VOLUME,
    hf_cache_volume_name: str = DEFAULT_HF_CACHE_VOLUME,
    volume_version: int = DEFAULT_VOLUME_VERSION,
    run_name: str = "",
    model_path: str = DEFAULT_MODEL,
    adapter_path: str = "",
    adapter_repo: str = DEFAULT_ADAPTER_REPO,
    adapter_repo_type: str = "dataset",
    adapter_safetensors: str = "adapter_model.safetensors",
    adapter_config: str = "adapter_config.json",
    tp_size: int = 1,
    benchmark_mode: str = "sync",
    warmup_cycles: int = 1,
    cycles: int = 3,
    flush_cache: bool = False,
    allow_skipped: bool = False,
    check_correctness: bool = False,
    profile: bool = False,
    profile_with_stack: bool = False,
    profile_record_shapes: bool = False,
    profile_activities: str = "CPU,GPU",
    profile_cycle: int = 1,
    disable_cuda_graph: bool = True,
    attention_backend: str = "flashinfer",
    prefill_attention_backend: str = "fa4",
    decode_attention_backend: str = "fa4",
    moe_runner_backend: str = "triton",
    experts_shared_outer_loras: bool = True,
    mem_fraction_static: float = 0.8,
    weight_sync_consistency_mode: str = "strict",
    service_impact_requests: int = 64,
    service_impact_concurrency: int = 8,
    service_impact_prompt: str = "Count from 1 to 64, separated by spaces.",
    service_impact_max_new_tokens: int = 96,
    service_impact_update_after_s: float = 0.5,
    service_impact_profile_update: bool = False,
    service_impact_update_transport: str = "prepared_tensor",
    service_impact_warmup_requests: int = 0,
    service_impact_warmup_concurrency: int = 0,
    service_impact_warmup_max_new_tokens: int = 0,
    prepared_bytes_transport_format: str = "flattened_bucket",
    prepared_bytes_transport_bucket_bytes: int = 512 * 1024 * 1024,
    cpu: int = 24,
    memory_mib: int = 131072,
    sync_artifacts: bool = True,
    sync_hf_cache: bool = False,
    extra_bench_args: str = "",
) -> None:
    if benchmark_mode not in {"sync", "service-impact"}:
        raise ValueError(f"Unsupported benchmark_mode={benchmark_mode!r}")
    if runner_mode not in {"persistent", "subprocess"}:
        raise ValueError(f"Unsupported runner_mode={runner_mode!r}")
    if runner_mode == "persistent" and extra_bench_args:
        raise ValueError(
            "extra_bench_args is only supported with runner_mode='subprocess'. "
            "Use the explicit top-level bench flags with the persistent runner."
        )

    artifacts_volume = _volume_from_name(
        volume_name,
        environment_name=environment_name,
        version=volume_version,
    )
    hf_cache_volume = _volume_from_name(
        hf_cache_volume_name,
        environment_name=environment_name,
        version=volume_version,
    )

    run_name = run_name or _default_run_name(tp_size=tp_size, gpu=gpu)
    run_dir = f"weight-sync-lora/{run_name}"
    output_root = f"{ARTIFACTS_MOUNT_PATH}/{run_dir}"
    region_config = _parse_region(region)
    runner_options = dict(
        gpu=gpu,
        cpu=cpu,
        memory=memory_mib,
        cloud=cloud,
        region=region_config,
        timeout=SANDBOX_TIMEOUT_S,
        volumes={
            ARTIFACTS_MOUNT_PATH: artifacts_volume,
            HF_CACHE_MOUNT_PATH: hf_cache_volume,
        },
    )
    sync_artifacts_enabled = _should_sync_volume(sync_artifacts, volume_version)
    sync_hf_cache_enabled = _should_sync_volume(sync_hf_cache, volume_version)

    if runner_mode == "persistent":
        engine_config_json = json.dumps(
            {
                "model_path": model_path,
                "tp_size": tp_size,
                "attention_backend": attention_backend,
                "prefill_attention_backend": prefill_attention_backend,
                "decode_attention_backend": decode_attention_backend,
                "moe_runner_backend": moe_runner_backend,
                "disable_cuda_graph": disable_cuda_graph,
                "experts_shared_outer_loras": experts_shared_outer_loras,
                "mem_fraction_static": mem_fraction_static,
                "weight_sync_consistency_mode": weight_sync_consistency_mode,
            },
            sort_keys=True,
        )
        adapter_config_json = json.dumps(
            {
                "adapter_path": adapter_path,
                "adapter_repo": adapter_repo,
                "adapter_repo_type": adapter_repo_type,
                "adapter_safetensors": adapter_safetensors,
                "adapter_config": adapter_config,
                "allow_skipped": allow_skipped,
            },
            sort_keys=True,
        )
        runner = PersistentWeightSyncBenchRunner.with_options(**runner_options)(
            engine_config_json=engine_config_json,
            adapter_config_json=adapter_config_json,
        )
        result = runner.run_benchmark.remote(
            benchmark_mode=benchmark_mode,
            run_dir=run_dir,
            output_root=output_root,
            volume_name=volume_name,
            hf_cache_volume_name=hf_cache_volume_name,
            environment_name=environment_name,
            volume_version=volume_version,
            sync_artifacts=sync_artifacts_enabled,
            sync_hf_cache=sync_hf_cache_enabled,
            cloud=cloud,
            region=region_config,
            gpu=gpu,
            flush_cache=flush_cache,
            warmup_cycles=warmup_cycles,
            cycles=cycles,
            check_correctness=check_correctness,
            profile=profile,
            profile_with_stack=profile_with_stack,
            profile_record_shapes=profile_record_shapes,
            profile_activities=profile_activities,
            profile_cycle=profile_cycle,
            service_impact_requests=service_impact_requests,
            service_impact_concurrency=service_impact_concurrency,
            service_impact_prompt=service_impact_prompt,
            service_impact_max_new_tokens=service_impact_max_new_tokens,
            service_impact_update_after_s=service_impact_update_after_s,
            service_impact_profile_update=service_impact_profile_update,
            service_impact_update_transport=service_impact_update_transport,
            service_impact_warmup_requests=service_impact_warmup_requests,
            service_impact_warmup_concurrency=service_impact_warmup_concurrency,
            service_impact_warmup_max_new_tokens=service_impact_warmup_max_new_tokens,
            prepared_bytes_transport_format=prepared_bytes_transport_format,
            prepared_bytes_transport_bucket_bytes=prepared_bytes_transport_bucket_bytes,
        )
    else:
        command = _build_benchmark_command(
            benchmark_mode=benchmark_mode,
            model_path=model_path,
            adapter_path=adapter_path,
            adapter_repo=adapter_repo,
            adapter_repo_type=adapter_repo_type,
            adapter_safetensors=adapter_safetensors,
            adapter_config=adapter_config,
            tp_size=tp_size,
            warmup_cycles=warmup_cycles,
            cycles=cycles,
            flush_cache=flush_cache,
            allow_skipped=allow_skipped,
            check_correctness=check_correctness,
            profile=profile,
            profile_with_stack=profile_with_stack,
            profile_record_shapes=profile_record_shapes,
            profile_activities=profile_activities,
            profile_cycle=profile_cycle,
            disable_cuda_graph=disable_cuda_graph,
            attention_backend=attention_backend,
            prefill_attention_backend=prefill_attention_backend,
            decode_attention_backend=decode_attention_backend,
            moe_runner_backend=moe_runner_backend,
            experts_shared_outer_loras=experts_shared_outer_loras,
            mem_fraction_static=mem_fraction_static,
            weight_sync_consistency_mode=weight_sync_consistency_mode,
            service_impact_requests=service_impact_requests,
            service_impact_concurrency=service_impact_concurrency,
            service_impact_prompt=service_impact_prompt,
            service_impact_max_new_tokens=service_impact_max_new_tokens,
            service_impact_update_after_s=service_impact_update_after_s,
            service_impact_profile_update=service_impact_profile_update,
            service_impact_update_transport=service_impact_update_transport,
            service_impact_warmup_requests=service_impact_warmup_requests,
            service_impact_warmup_concurrency=service_impact_warmup_concurrency,
            service_impact_warmup_max_new_tokens=service_impact_warmup_max_new_tokens,
            prepared_bytes_transport_format=prepared_bytes_transport_format,
            prepared_bytes_transport_bucket_bytes=prepared_bytes_transport_bucket_bytes,
            output_root=output_root,
            extra_bench_args=extra_bench_args,
        )
        runner = WeightSyncBenchRunner.with_options(**runner_options)()
        result = runner.run.remote(
            command=command,
            run_dir=run_dir,
            output_root=output_root,
            volume_name=volume_name,
            hf_cache_volume_name=hf_cache_volume_name,
            environment_name=environment_name,
            volume_version=volume_version,
            sync_artifacts=sync_artifacts_enabled,
            sync_hf_cache=sync_hf_cache_enabled,
            cloud=cloud,
            region=region_config,
            gpu=gpu,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
