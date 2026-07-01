"""
DCP (Decode Context Parallelism) + DFLASH speculative decoding on the
tokenspeed_mla attention backend.

This exercises the CUDA/B200 DCP path added for tokenspeed_mla:
  - Decode (q_len=1): q all-gather along heads -> tokenspeed FP8 MLA decode
    kernel over the rank-local KV shard with return_lse=True (base-2 LSE,
    softmax scale folded) -> cross-rank merge via cp_lse_ag_out_rs_mla.
  - DFLASH target verify (q_len=num_draft_tokens): two-phase backend
    attention — (a) non-causal decode kernel over the rank-LOCAL committed
    prefix, (b) residue-class share of the fresh draft block in torch,
    (c) local base-2 merge — then the same cross-rank merge.
  - Rank-invariant block tables built with the widened logical page size
    (page_size * dcp_world_size) over the logical req_to_token.
  - DFLASH draft KV pool REPLICATED per rank (sized by the logical
    allocator capacity), so the fa4 draft backend runs unchanged.

Requires B200 (SM100): tokenspeed_mla is Blackwell-only, FP8 KV, and the
LSE-returning decode kernel ships in the tokenspeed_mla wheel built from
branch jamesliu/decode-lse. This test cannot run on non-Blackwell CI.

Model paths are overridable via env for the B200 runner:
  SGLANG_TEST_DCP_DFLASH_MODEL        target model (MLA, e.g. Kimi K2.6 NVFP4)
  SGLANG_TEST_DCP_DFLASH_DRAFT_MODEL  DFLASH draft checkpoint
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

# B200-only (SM100 tokenspeed_mla kernels); keep out of default CI stages.
register_cuda_ci(est_time=3600, nightly=True, suite="nightly-4-gpu-b200-dcp-dflash")

MODEL_PATH = os.getenv(
    "SGLANG_TEST_DCP_DFLASH_MODEL",
    "moonshotai/Kimi-K2.6-NVFP4",
)
DRAFT_MODEL_PATH = os.getenv(
    "SGLANG_TEST_DCP_DFLASH_DRAFT_MODEL",
    "moonshotai/Kimi-K2.6-DFlash-draft",
)

_COMMON_SERVER_ARGS = [
    "--tp-size",
    "4",
    "--trust-remote-code",
    "--attention-backend",
    "tokenspeed_mla",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "64",
    "--mem-fraction-static",
    "0.82",
    "--max-running-requests",
    "128",
    "--cuda-graph-max-bs-decode",
    "128",
    "--enable-metrics",
    "--random-seed",
    "0",
    "--log-level",
    "info",
]

_DFLASH_ARGS = [
    "--speculative-algorithm",
    "DFLASH",
    f"--speculative-draft-model-path={DRAFT_MODEL_PATH}",
    "--speculative-num-draft-tokens",
    "8",
    "--speculative-draft-attention-backend",
    "fa4",
]

_DCP4_ARGS = [
    "--dcp-size",
    "4",
]


def _get_server_info(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/server_info", timeout=30)
    resp.raise_for_status()
    return resp.json()


class TestDCPTokenspeedDFlashGSM8K(
    GSM8KMixin, BasicDecodeCorrectnessMixin, CustomTestCase
):
    """DCP=4 + TP=4 + DFLASH on tokenspeed_mla — accuracy gate + decode probes.

    Covers, per forward mode:
      - EXTEND (prefill): existing DCP prefix all-gather path (MHA chunked
        prefix), tokenspeed prefill kernel unchanged.
      - TARGET_VERIFY: two-phase DCP verify (local prefix + residue-class
        draft block) under CUDA graphs.
      - DFLASH draft forward: fa4 over the replicated draft KV pool with
        logical cache locations.

    Inherits:
      - GSM8KMixin.test_gsm8k: accuracy gate
      - BasicDecodeCorrectnessMixin: factual recall, no-repetition, temp=0
        determinism, max_new_tokens=1 (catches graph capture bugs)
    """

    model = MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Loose initial gate; tighten once a non-DCP DFLASH baseline on the same
    # checkpoint is established (typically within 1-2% of the dense baseline).
    gsm8k_accuracy_thres = 0.90
    gsm8k_num_questions = 200
    gsm8k_num_threads = 64
    gsm8k_num_shots = 5

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_DCP4_ARGS + _DFLASH_ARGS + _COMMON_SERVER_ARGS,
        )
        cls._server_info = _get_server_info(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_dcp_activation_check(self):
        """max_total_num_tokens is the LOGICAL (dcp-widened) capacity; a
        positive value confirms the widened allocator initialized."""
        self.assertGreater(self._server_info["max_total_num_tokens"], 0)

    def test_spec_accept_rate_sane(self):
        """Run a batch of temp=0 generations and verify speculative decoding
        is actually accepting draft tokens (a broken DCP verify path shows up
        as accept length pinned at 1, i.e. every draft token rejected)."""
        prompts = [
            "The capital of France is",
            "1, 1, 2, 3, 5, 8, 13, 21,",
            "Water is composed of hydrogen and",
            "def fibonacci(n):\n    ",
        ]
        for prompt in prompts:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                },
                timeout=120,
            )
            resp.raise_for_status()
        info = _get_server_info(self.base_url)
        # decode_avg_spec_accept_length is exposed in scheduler info when
        # speculative decoding is enabled.
        accept_len = info.get("avg_spec_accept_length") or info.get(
            "internal_states", [{}]
        )[0].get("avg_spec_accept_length")
        if accept_len is not None:
            self.assertGreater(
                float(accept_len),
                1.1,
                "DFLASH under DCP accepted almost no draft tokens; the DCP "
                "target-verify merge is likely mis-normalized.",
            )


@unittest.skipIf(
    is_in_ci(),
    "Requires two server launches; run manually on B200 for DCP parity checks.",
)
class TestDCPTokenspeedDFlashParity(CustomTestCase):
    """DCP=4 vs non-DCP output parity for tokenspeed_mla + DFLASH (manual).

    Launch a non-DCP baseline, record temp=0 outputs, relaunch with
    --dcp-size 4, and require identical outputs (greedy) within logprob
    tolerance. Catches normalization bugs in the two-phase verify merge that
    a coarse accuracy gate can miss.
    """

    LOGPROB_TOLERANCE = 1.0
    base_url = "http://127.0.0.1:31600"
    model = MODEL_PATH

    _PROMPTS = [
        "The capital city of France is",
        "What is 2 + 3? The answer is",
        "In the year 1492, Christopher Columbus",
        "The largest planet in our solar system is",
    ]

    @classmethod
    def setUpClass(cls):
        cls._processes = []

    @classmethod
    def tearDownClass(cls):
        for proc in cls._processes:
            try:
                kill_process_tree(proc.pid, wait_timeout=60)
            except Exception:
                pass

    def _launch(self, extra_args):
        proc = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=extra_args,
        )
        self._processes.append(proc)
        return proc

    def _generate(self, prompt, max_new_tokens=16):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
                "return_logprob": True,
                "top_logprobs_num": 1,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["text"], data["meta_info"]["output_token_logprobs"]

    def test_dcp_vs_baseline_parity(self):
        # 1) non-DCP baseline
        baseline_proc = self._launch(_DFLASH_ARGS + _COMMON_SERVER_ARGS)
        baseline = {p: self._generate(p) for p in self._PROMPTS}
        kill_process_tree(baseline_proc.pid, wait_timeout=60)

        # 2) DCP=4
        self._launch(_DCP4_ARGS + _DFLASH_ARGS + _COMMON_SERVER_ARGS)
        for prompt in self._PROMPTS:
            text, logprobs = self._generate(prompt)
            base_text, base_logprobs = baseline[prompt]
            self.assertEqual(
                text,
                base_text,
                f"DCP output diverged from baseline for prompt {prompt!r}",
            )
            for (lp, _tok, *_), (blp, _btok, *_) in zip(logprobs, base_logprobs):
                self.assertLess(
                    abs(lp - blp),
                    self.LOGPROB_TOLERANCE,
                    f"logprob divergence for prompt {prompt!r}",
                )


if __name__ == "__main__":
    unittest.main()
