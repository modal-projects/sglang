import os
import unittest
from dataclasses import dataclass

from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    _parse_int_list_env,
    is_blackwell_system,
)

DEFAULT_MODEL_PATH = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
BENCH_MODEL_PATH = (
    os.environ.get("FP8_ATTN_BENCH_MODEL_PATH")
    or os.environ.get("QWEN35_FP8_MODEL_PATH")
    or DEFAULT_MODEL_PATH
)
PROFILE_DIR = "performance_profiles_qwen35_fp8_attention_backends"
ATTENTION_BACKENDS = ("trtllm_mha", "fa4")
DEFAULT_TRTLLM_PAGE_SIZE = int(os.environ.get("FP8_ATTN_BENCH_TRTLLM_PAGE_SIZE", "64"))
DEFAULT_FA4_PAGE_SIZE = int(os.environ.get("FP8_ATTN_BENCH_FA4_PAGE_SIZE", "64"))

COMMON_ARGS = [
    "--trust-remote-code",
    "--ep=2",
    "--chunked-prefill-size=16384",
    "--max-prefill-tokens=16384",
    "--reasoning-parser=qwen3",
    "--tool-call-parser=qwen3_coder",
    "--enable-flashinfer-allreduce-fusion",
    "--kv-cache-dtype=fp8_e4m3",
    "--mem-fraction-static=0.8",
    "--enable-metrics",
]


@dataclass
class BackendPerfResult:
    variant: str
    passed: bool
    error: str | None = None
    latency: float | None = None
    input_throughput: float | None = None
    output_throughput: float | None = None
    overall_throughput: float | None = None


def get_page_size_for_backend(backend: str) -> int:
    if backend == "fa4":
        return DEFAULT_FA4_PAGE_SIZE
    if backend == "trtllm_mha":
        return DEFAULT_TRTLLM_PAGE_SIZE
    raise ValueError(f"Unsupported backend: {backend}")


def build_variants() -> list[ModelLaunchSettings]:
    return [
        ModelLaunchSettings(
            BENCH_MODEL_PATH,
            tp_size=8,
            extra_args=COMMON_ARGS
            + [
                f"--page-size={get_page_size_for_backend(backend)}",
                f"--attention-backend={backend}",
            ],
            variant=f"TP8-EP2-KVFP8-{backend}",
        )
        for backend in ATTENTION_BACKENDS
    ]


def run_backend_comparison(
    batch_sizes: list[int],
    input_lens: tuple[int, ...],
    output_lens: tuple[int, ...],
) -> tuple[list[BackendPerfResult], bool]:
    perf_runner = NightlyBenchmarkRunner(
        profile_dir=PROFILE_DIR,
        test_name="Qwen3.5-397B-A17B-FP8 Attention Backend Comparison",
        base_url=DEFAULT_URL_FOR_TEST,
    )
    perf_runner.setup_profile_directory()

    all_results: list[BackendPerfResult] = []
    all_passed = True

    for model in build_variants():
        print("\n" + "=" * 80)
        print(f"PERFORMANCE TEST: {model.model_path}")
        print(f"  Variant: {model.variant}")
        print(f"  Extra Args: {model.extra_args}")
        print("  Profiling: disabled")
        print("=" * 80)

        try:
            results, success, _ = perf_runner.run_benchmark_for_model(
                model_path=model.model_path,
                batch_sizes=batch_sizes,
                input_lens=input_lens,
                output_lens=output_lens,
                other_args=model.extra_args,
                variant=model.variant or "",
                extra_bench_args=["--trust-remote-code"],
                enable_profile=False,
                env=model.env,
            )

            if success and results:
                perf_runner.add_report(results, variant=model.variant)
                largest_batch_result = max(results, key=lambda r: r.batch_size)
                all_results.append(
                    BackendPerfResult(
                        variant=model.variant or model.model_path,
                        passed=True,
                        latency=largest_batch_result.latency,
                        input_throughput=largest_batch_result.input_throughput,
                        output_throughput=largest_batch_result.output_throughput,
                        overall_throughput=largest_batch_result.overall_throughput,
                    )
                )
            else:
                all_passed = False
                all_results.append(
                    BackendPerfResult(
                        variant=model.variant or model.model_path,
                        passed=False,
                        error="Benchmark failed",
                    )
                )
        except Exception as exc:
            all_passed = False
            all_results.append(
                BackendPerfResult(
                    variant=model.variant or model.model_path,
                    passed=False,
                    error=str(exc),
                )
            )

    perf_runner.write_final_report()
    return all_results, all_passed


def print_backend_summary(results: list[BackendPerfResult]) -> None:
    print("\n" + "=" * 72)
    print("Attention Backend Comparison")
    print("=" * 72)

    metrics_by_variant = {}
    for result in results:
        metrics_by_variant[result.variant] = result
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.variant}: {status}, "
            f"latency={result.latency}, "
            f"input_tput={result.input_throughput}, "
            f"output_tput={result.output_throughput}"
        )
        if result.error:
            print(f"  error={result.error}")

    fa4 = metrics_by_variant.get("TP8-EP2-KVFP8-fa4")
    trtllm = metrics_by_variant.get("TP8-EP2-KVFP8-trtllm_mha")
    if fa4 and trtllm and fa4.output_throughput and trtllm.output_throughput:
        speedup = fa4.output_throughput / trtllm.output_throughput
        print(f"FA4 vs TRTLLM output throughput speedup: {speedup:.3f}x")
    if fa4 and trtllm and fa4.input_throughput and trtllm.input_throughput:
        speedup = fa4.input_throughput / trtllm.input_throughput
        print(f"FA4 vs TRTLLM input throughput speedup: {speedup:.3f}x")
    print("=" * 72 + "\n")


@unittest.skipUnless(is_blackwell_system(), "Requires a Blackwell GPU system")
class TestQwen35FP8AttentionBackends(unittest.TestCase):
    """Local perf-only A/B for FA4-compatible FP8 models on 8x Blackwell.

    This keeps the workload fixed and only changes `--attention-backend`
    so FA4 FP8 can be compared directly against TRTLLM MHA.
    """

    def test_qwen35_fp8_attention_backends(self):
        batch_sizes = _parse_int_list_env("QWEN35_FP8_BENCH_BATCH_SIZES", "1,8,16,64")
        input_lens = tuple(_parse_int_list_env("QWEN35_FP8_BENCH_INPUT_LENS", "4096"))
        output_lens = tuple(_parse_int_list_env("QWEN35_FP8_BENCH_OUTPUT_LENS", "512"))

        results, all_passed = run_backend_comparison(
            batch_sizes=batch_sizes,
            input_lens=input_lens,
            output_lens=output_lens,
        )

        print_backend_summary(results)
        self.assertTrue(all_passed, "One or more backend perf runs failed")


if __name__ == "__main__":
    unittest.main()
