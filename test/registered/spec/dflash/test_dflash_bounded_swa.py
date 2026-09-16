"""End-to-end test for the bounded, request-owned DFLASH draft KV cache.

Pairs a hybrid linear/full-attention target with an all-SWA DFLASH drafter:

* target ``Qwen/Qwen3.8-27B`` (64 layers, linear_attention + full_attention)
* draft ``z-lab/Qwen3.8-27B-DFlash2`` (5 ``sliding_attention`` layers,
  ``sliding_window`` 2048, block size 8)

With an all-SWA drafter the draft KV pool is a fixed per-request ring instead
of one draft row per target token, and radix prefix hits hold back one draft
window so the ring is re-prefilled. This test checks that the bounded pool is
selected and sized as planned, that greedy output is stable across a prefix
hit, and that accuracy and accept length hold on GSM8K. A second class runs
the same server with the soft hold-back policy, where a prefix hit resumes at
the cache's natural resume point and the draft attends only to the part of its
ring written since then; it checks that the hit is no longer capped and reports
the accept length over the first tokens after a hit against a cold prefill.
"""

import os
import tempfile
import time
import unittest
from contextlib import ExitStack

import openai
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1800, stage="nightly", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3.8-27B"
DRAFT_MODEL = "z-lab/Qwen3.8-27B-DFlash2"
# From the draft config: sliding_window and dflash_config.block_size.
DRAFT_WINDOW_TOKENS = 2048
DRAFT_BLOCK_SIZE = 8

BOUNDED_LOG_MARKER = "DFLASH bounded all-SWA draft KV cache"
FALLBACK_LOG_MARKER = "bounded draft KV cache is disabled"

# A prompt several times longer than the draft window, so a repeated request
# is a partial prefix hit whose visible draft window lies inside cached territory.
_LONG_PROMPT = " ".join(
    f"Entry {i}: the quick brown fox number {i} jumps over the lazy dog number {i + 1}."
    for i in range(700)
)
# An open-ended instruction after the filler, so a 512-token greedy continuation
# is prose rather than a repetition of the filler pattern.
_ESSAY_QUESTION = (
    "\n\nWrite a detailed essay on how foxes and dogs are portrayed in folklore, "
    "with examples from at least three cultures."
)


class TestDFlashBoundedSwa(CustomTestCase, GSM8KMixin):
    model = TARGET_MODEL
    draft_model = DRAFT_MODEL
    page_size = 1
    max_running_requests = 32
    other_launch_args: list = []
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 3.0
    gsm8k_num_questions = 200

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_dir = os.environ.get("DFLASH_BOUNDED_SWA_LOG_DIR") or tempfile.mkdtemp(
            prefix="dflash_bounded_swa_"
        )
        os.makedirs(cls.log_dir, exist_ok=True)
        cls.stdout = open(os.path.join(cls.log_dir, "server_stdout.log"), "w")
        cls.stderr = open(os.path.join(cls.log_dir, "server_stderr.log"), "w")
        launch_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-draft-tokens",
            str(DRAFT_BLOCK_SIZE),
            "--page-size",
            str(cls.page_size),
            # The bounded pool is sized from this, so it must be explicit.
            "--max-running-requests",
            str(cls.max_running_requests),
            "--mem-fraction-static",
            "0.8",
            *cls.other_launch_args,
        ]
        with ExitStack() as stack:
            for env, value in (
                (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
                (envs.SGLANG_ENABLE_ASYNC_ASSERT, True),
            ):
                stack.enter_context(env.override(value))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1800),
                other_args=launch_args,
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        for f in (getattr(cls, "stdout", None), getattr(cls, "stderr", None)):
            if f is not None:
                f.close()

    def _server_log(self) -> str:
        self.stdout.flush()
        self.stderr.flush()
        text = ""
        for name in ("server_stdout.log", "server_stderr.log"):
            with open(os.path.join(self.log_dir, name)) as f:
                text += f.read()
        return text

    def _generate(
        self, prompt: str, max_new_tokens: int, ignore_eos: bool = False
    ) -> dict:
        res = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "ignore_eos": ignore_eos,
                },
            },
        )
        res.raise_for_status()
        return res.json()

    def _repeat_until_hit(self, prompt: str, max_new_tokens: int, **kw) -> dict:
        """Re-send a prompt until the radix tree reports a hit.

        A finished request is inserted into the tree asynchronously, so an
        immediate repeat can still miss.
        """
        warm = None
        for _ in range(10):
            warm = self._generate(prompt, max_new_tokens, **kw)
            if warm["meta_info"]["cached_tokens"] > 0:
                return warm
            time.sleep(1)
        return warm

    def _prefix_hit(self) -> tuple:
        """Cold prefill of the long prompt, then a repeat that hits the cache."""
        requests.post(self.base_url + "/flush_cache").raise_for_status()
        cold = self._generate(_LONG_PROMPT, max_new_tokens=64)
        prompt_tokens = cold["meta_info"]["prompt_tokens"]
        self.assertGreater(prompt_tokens, DRAFT_WINDOW_TOKENS + 512)
        self.assertTrue(cold["text"].strip())
        warm = self._repeat_until_hit(_LONG_PROMPT, max_new_tokens=64)
        cached_tokens = warm["meta_info"]["cached_tokens"]
        print(
            f"prefix hit: prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}"
        )
        self.assertGreater(cached_tokens, 0, "repeated prompt never hit the cache")
        return cold, warm

    def test_accept_length_after_hit(self):
        """Report the draft's accept length over the first 512 generated tokens
        after a prefix hit, against the same span after a cold prefill. Under the
        hard hold-back the draft window is fully re-prefilled; under the soft
        hold-back the draft starts with a partial window. This is a measurement,
        so only sanity bounds are asserted: greedy output after a hit can differ
        from the cold output (prefill over cached KV vs cold chunked prefill flip
        near-tie argmaxes), and accept length depends on the text generated.
        """
        prompt = _LONG_PROMPT + _ESSAY_QUESTION
        requests.post(self.base_url + "/flush_cache").raise_for_status()
        cold = self._generate(prompt, max_new_tokens=512, ignore_eos=True)
        # Two repeats: the first may resume at a coarser checkpoint than the
        # second (a Mamba cache records a branch point on the first capped hit).
        self._repeat_until_hit(prompt, max_new_tokens=512, ignore_eos=True)
        time.sleep(1)
        warm = self._generate(prompt, max_new_tokens=512, ignore_eos=True)
        cold_al = cold["meta_info"].get("spec_accept_length")
        warm_al = warm["meta_info"].get("spec_accept_length")
        print(
            "accept length over 512 tokens: "
            f"cold={cold_al}, after hit={warm_al}, "
            f"hit cached_tokens={warm['meta_info']['cached_tokens']} "
            f"of prompt_tokens={warm['meta_info']['prompt_tokens']}"
        )
        self.assertIsNotNone(cold_al)
        self.assertIsNotNone(warm_al)
        self.assertGreater(cold_al, 1.0)
        self.assertGreater(warm_al, 1.0)
        assert self.process.poll() is None

    def test_bounded_pool_selected(self):
        """The gate picked the bounded pool and sized it from the plan."""
        log = self._server_log()
        self.assertNotIn(FALLBACK_LOG_MARKER, log)
        self.assertIn(BOUNDED_LOG_MARKER, log)
        # Ring slots per request: the visible window, page alignment slack, and
        # two speculative blocks (one in flight, one reserved by overlap
        # scheduling), rounded up to a whole page. Spelled out rather than
        # imported so the test does not use the code under test as its oracle.
        minimum = DRAFT_WINDOW_TOKENS + (self.page_size - 1) + 2 * DRAFT_BLOCK_SIZE
        capacity = (minimum + self.page_size - 1) // self.page_size * self.page_size
        self.assertIn(
            f"attention_window={DRAFT_WINDOW_TOKENS}, capacity_per_request={capacity}",
            log,
        )
        self.assertIn(f"pool_tokens={capacity * self.max_running_requests}", log)

    def test_prefix_hit_holds_back_draft_window(self):
        """A repeated long prompt is a prefix hit. The match must be a real hit
        and must be capped so at least one draft window is re-prefilled.

        Greedy text after a prefix hit is not compared with the cold run: a
        prefill over cached KV and a cold chunked prefill differ numerically
        and can flip a near-tie argmax, on the target-indexed pool as much as
        on the bounded one. Determinism is asserted between two requests that
        see the same cache state instead.
        """
        cold, warm = self._prefix_hit()
        prompt_tokens = cold["meta_info"]["prompt_tokens"]
        cached_tokens = warm["meta_info"]["cached_tokens"]
        # Capped by the hold-back: at least one draft window is re-prefilled.
        # (A radix cache may round the match down further to a node boundary.)
        self.assertLessEqual(cached_tokens, prompt_tokens - DRAFT_WINDOW_TOKENS)

        # Same prompt, same cache state: greedy output must be identical.
        warm_again = self._generate(_LONG_PROMPT, max_new_tokens=64)
        self.assertEqual(warm["text"], warm_again["text"])

        # Branch off the shared prefix with a different suffix.
        branched = self._generate(
            _LONG_PROMPT + " In one sentence, what do all entries have in common?",
            max_new_tokens=32,
        )
        self.assertTrue(branched["text"].strip())
        assert self.process.poll() is None

    def test_greedy_determinism(self):
        client = openai.Client(base_url=self.base_url + "/v1", api_key="EMPTY")
        prompt = "The capital of France is"
        outputs = []
        for _ in range(2):
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=32,
                temperature=0,
            )
            outputs.append(response.choices[0].text)
        print(f"determinism: {outputs=}")
        self.assertEqual(outputs[0], outputs[1])
        assert self.process.poll() is None

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        assert self.process.poll() is None


class TestDFlashBoundedSwaSoftHoldback(TestDFlashBoundedSwa):
    """Same server with the soft hold-back policy at threshold 0: any prefix
    hit whose cap would cost extra prefill resumes uncapped and the draft
    attends only to the ring positions written since the resume point."""

    other_launch_args = ["--speculative-draft-soft-holdback-threshold", "0"]

    def test_prefix_hit_holds_back_draft_window(self):
        """The repeat resumes past the hold-back cap: more than
        prompt_len - window tokens are reused (the Mamba cache resumes at the
        cold run's prompt-end checkpoint, 64-aligned)."""
        cold, warm = self._prefix_hit()
        prompt_tokens = cold["meta_info"]["prompt_tokens"]
        cached_tokens = warm["meta_info"]["cached_tokens"]
        self.assertGreater(cached_tokens, prompt_tokens - DRAFT_WINDOW_TOKENS)

        # Same prompt, same cache state: greedy output must be identical.
        warm_again = self._generate(_LONG_PROMPT, max_new_tokens=64)
        self.assertEqual(warm["text"], warm_again["text"])
        assert self.process.poll() is None


if __name__ == "__main__":
    unittest.main()
