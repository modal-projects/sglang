"""End-to-end DFlash checkpoint reuse on a hybrid GDN model."""

import json
import unittest
import uuid

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")

TARGET_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
DRAFT_MODEL = "modal-labs/Qwen3.6-35B-A3B-DFlash"
TRACK_INTERVAL = 256
FRESH_REPEAT_SCALE = 3.0
FRESH_REPEAT_MARGIN = 0.5


class TestDFlashMambaRadixCheckpoint(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(True):
            cls.process = popen_launch_server(
                TARGET_MODEL,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
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
                    "--mamba-radix-cache-strategy",
                    "extra_buffer",
                    "--mamba-track-interval",
                    str(TRACK_INTERVAL),
                    "--mamba-ssm-dtype",
                    "bfloat16",
                    "--max-running-requests",
                    "2",
                    "--context-length",
                    "2048",
                    "--max-total-tokens",
                    "4096",
                    "--max-mamba-cache-size",
                    "10",
                    "--mem-fraction-static",
                    "0.65",
                    "--cuda-graph-backend-decode",
                    "disabled",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _generate(self, payload):
        response = requests.post(
            self.base_url + "/generate",
            json=payload,
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def _compare_top_logprobs(lhs_meta, rhs_meta):
        def top_map(meta):
            return {entry[1]: entry[0] for entry in meta["output_top_logprobs"][0]}

        lhs_top = top_map(lhs_meta)
        rhs_top = top_map(rhs_meta)
        common_tokens = lhs_top.keys() & rhs_top.keys()
        max_abs_diff = max(
            (abs(lhs_top[token] - rhs_top[token]) for token in common_tokens),
            default=float("inf"),
        )
        return len(common_tokens), max_abs_diff

    def test_cached_decode_checkpoint_matches_fresh_recompute(self):
        cache_key = f"dflash-mamba-radix-{uuid.uuid4().hex}"
        first = self._generate(
            {
                "text": (
                    "Give a detailed proof that there are infinitely many prime "
                    "numbers, then discuss two variants."
                ),
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_new_tokens": 270,
                    "ignore_eos": True,
                    "sampling_seed": 0,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
                "extra_key": cache_key,
            }
        )
        prompt_ids = [entry[1] for entry in first["meta_info"]["input_token_logprobs"]]
        full_prefix = prompt_ids + first["output_ids"]

        def score(extra_key):
            return self._generate(
                {
                    "input_ids": full_prefix,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                    "return_logprob": True,
                    "logprob_start_len": len(full_prefix) - 1,
                    "top_logprobs_num": 20,
                    "extra_key": extra_key,
                }
            )

        cached = score(cache_key)
        fresh = score(cache_key + "-fresh")
        fresh_repeat = score(cache_key + "-fresh-repeat")

        # Exercise the non-speculative prefill checkpoint path independently of
        # the checkpoint produced while DFlash advances through the boundary.
        prefill_cache_key = cache_key + "-prefill-checkpoint"
        self._generate(
            {
                "input_ids": full_prefix[:TRACK_INTERVAL],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "extra_key": prefill_cache_key,
            }
        )
        prefill_cached = score(prefill_cache_key)

        cached_meta = cached["meta_info"]
        fresh_meta = fresh["meta_info"]
        fresh_repeat_meta = fresh_repeat["meta_info"]
        prefill_cached_meta = prefill_cached["meta_info"]
        cached_common, cached_diff = self._compare_top_logprobs(cached_meta, fresh_meta)
        prefill_common, prefill_diff = self._compare_top_logprobs(
            prefill_cached_meta, fresh_meta
        )
        repeat_common, repeat_diff = self._compare_top_logprobs(
            fresh_meta, fresh_repeat_meta
        )
        cache_diff_limit = FRESH_REPEAT_SCALE * repeat_diff + FRESH_REPEAT_MARGIN
        metrics = {
            "cached_tokens": cached_meta["cached_tokens"],
            "fresh_cached_tokens": fresh_meta["cached_tokens"],
            "fresh_repeat_cached_tokens": fresh_repeat_meta["cached_tokens"],
            "prefill_cached_tokens": prefill_cached_meta["cached_tokens"],
            "cached_common_top20": cached_common,
            "cached_max_abs_logprob_diff": cached_diff,
            "prefill_common_top20": prefill_common,
            "prefill_max_abs_logprob_diff": prefill_diff,
            "fresh_repeat_common_top20": repeat_common,
            "fresh_repeat_max_abs_logprob_diff": repeat_diff,
            "cache_diff_limit_from_fresh_repeat": cache_diff_limit,
        }
        passed = (
            metrics["cached_tokens"] >= TRACK_INTERVAL
            and metrics["fresh_cached_tokens"] == 0
            and metrics["fresh_repeat_cached_tokens"] == 0
            and metrics["prefill_cached_tokens"] >= TRACK_INTERVAL
            and metrics["cached_common_top20"] >= 18
            and metrics["prefill_common_top20"] >= 18
            and metrics["fresh_repeat_common_top20"] >= 18
            and metrics["cached_max_abs_logprob_diff"] <= cache_diff_limit
            and metrics["prefill_max_abs_logprob_diff"] <= cache_diff_limit
        )
        print(
            "VERDICT dflash_mamba_radix "
            + ("PASS " if passed else "FAIL ")
            + json.dumps(metrics, sort_keys=True),
            flush=True,
        )

        self.assertGreaterEqual(metrics["cached_tokens"], TRACK_INTERVAL)
        self.assertEqual(metrics["fresh_cached_tokens"], 0)
        self.assertEqual(metrics["fresh_repeat_cached_tokens"], 0)
        self.assertGreaterEqual(metrics["prefill_cached_tokens"], TRACK_INTERVAL)
        self.assertGreaterEqual(metrics["cached_common_top20"], 18)
        self.assertGreaterEqual(metrics["prefill_common_top20"], 18)
        self.assertGreaterEqual(metrics["fresh_repeat_common_top20"], 18)
        self.assertLessEqual(metrics["cached_max_abs_logprob_diff"], cache_diff_limit)
        self.assertLessEqual(metrics["prefill_max_abs_logprob_diff"], cache_diff_limit)


if __name__ == "__main__":
    unittest.main()
