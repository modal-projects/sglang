"""Greedy logprob A/B probe for SGLANG_OPT_USE_INKLING_SHEARED_BIAS.

Kernel-level tests (test_shear_bias_precision.py) show shear == score_mod to
bf16 noise vs an fp64 reference in every config. This probe checks the full
serving stack (TP8, cuda graphs, radix cache, chunked prefill, overlap sched):
send identical prompts greedy, record output tokens + logprobs, diff runs.

Usage:
  1. launch server with SGLANG_OPT_USE_INKLING_SHEARED_BIAS=0  -> python ab_logprob_probe.py --out /tmp/ab_scoremod.json
  2. relaunch with SGLANG_OPT_USE_INKLING_SHEARED_BIAS=1       -> python ab_logprob_probe.py --out /tmp/ab_shear.json
  3. python ab_logprob_probe.py --compare /tmp/ab_scoremod.json /tmp/ab_shear.json

Identical outputs + logprob deltas ~1e-3 => paths equivalent e2e; the AIME gap
is eval variance. Diverging tokens with growing logprob gap => real e2e bug.
"""

import argparse
import json

import requests

PROMPTS = [
    "Solve: what is the sum of the first 100 positive integers? Explain step by step.",
    "Prove that sqrt(2) is irrational.",
    ("The following is a long technical document. " + "In numerical analysis, floating point rounding interacts with reduction order. " * 400)[:12000]
    + " Summarize the key point in one sentence.",
    "Compute the number of ways to tile a 2x10 board with 2x1 dominoes. Show your reasoning.",
    "x" * 5 + " Repeat the previous token exactly 3 times.",
]


def run(port: int, out_path: str):
    results = []
    for i, p in enumerate(PROMPTS):
        r = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json={
                "text": p,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 256},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=600,
        )
        r.raise_for_status()
        d = r.json()
        mi = d["meta_info"]
        results.append(
            {
                "prompt_idx": i,
                "output_ids": [t[1] for t in mi["output_token_logprobs"]],
                "output_logprobs": [t[0] for t in mi["output_token_logprobs"]],
                "input_logprobs": [t[0] for t in mi.get("input_token_logprobs", []) if t[0] is not None],
            }
        )
        print(f"prompt {i}: {len(results[-1]['output_ids'])} tokens")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"saved {out_path}")


def compare(a_path: str, b_path: str):
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    for ra, rb in zip(a, b):
        i = ra["prompt_idx"]
        ta, tb = ra["output_ids"], rb["output_ids"]
        n = min(len(ta), len(tb))
        div = next((k for k in range(n) if ta[k] != tb[k]), None)
        common = div if div is not None else n
        dlp = max(
            (abs(x - y) for x, y in zip(ra["output_logprobs"][:common], rb["output_logprobs"][:common])),
            default=0.0,
        )
        dlp_in = max(
            (abs(x - y) for x, y in zip(ra["input_logprobs"], rb["input_logprobs"])),
            default=0.0,
        )
        status = "MATCH" if div is None and len(ta) == len(tb) else f"DIVERGE@{div}"
        print(
            f"prompt {i}: {status} len={len(ta)}/{len(tb)} "
            f"max|dlogprob| out={dlp:.3e} in={dlp_in:.3e}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        assert args.out
        run(args.port, args.out)
