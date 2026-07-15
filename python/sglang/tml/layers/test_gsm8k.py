      
#!/usr/bin/env python3
"""Self-contained GSM8K accept-length (acc-len) benchmark for d66 chain-MTP speculative decoding.

Measures the speculative-decode *accept length* of a d66 MTP checkpoint on the GSM8K test set
(1319 questions) against a running SGLang server. Two prompt variants matching the production eval:

  * ``nosys`` : the bare user question.
  * ``sys``   : a ``Thinking effort level: <effort>`` system turn followed by the user question.

Prompts are rendered to ``input_ids`` with the Inkling tokenizer and chat framing,
with ``add_generation_prompt=False`` so the prompt ends at
``<|end_message|>`` (200010) and the model emits the assistant turn itself -- byte-identical to
the production eval prompts. Because ids are sent directly, this works against a
``--skip-tokenizer-init`` server (which only accepts token ids).

Accept length is the server's own speculative-decode metric, aggregated over all requests:

    acc_length = sum(completion_tokens) / sum(spec_verify_ct)

i.e. mean accepted tokens per target-model verify step. It is invariant to client batch size
(``--concurrency``) given correctly-logged per-request stats, so raise concurrency for speed.

The stop token ``<|content_model_end_sampling|>`` (200006) is REQUIRED: a skip-tokenizer-init
server does not auto-stop on it, so without a stop the generation runs to ``--max-new-tokens``
and then loops, which (being trivially predictable) inflates acc_length.

Launch a server first (see README.md), then e.g.:

    python benchmark/gsm8k_mtp_acclen/drive_gsm8k_acclen.py \
        --url http://HOST:PORT --variant both --temperature 0 --concurrency 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)


def build_inkling_tokenizer():
    """Build the Inkling tokenizer and chat-framing overlay."""
    import sglang.tml
    from sglang.tml.tokenizer import TmlTokenizer
    from transformers import AutoTokenizer

    hf_dir = os.path.join(os.path.dirname(sglang.tml.__file__), "huggingface")
    base = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    return TmlTokenizer(tokenizer=base)


def eot_token_id() -> int:
    """d66 assistant-turn end: ``<|content_model_end_sampling|>`` == 200006."""
    from sglang.tml.tokenizer import TML_SPECIAL_TOKEN_IDS

    return int(TML_SPECIAL_TOKEN_IDS["<|content_model_end_sampling|>"])


def load_questions(num_questions: int, cache_path: str) -> list[str]:
    if not os.path.isfile(cache_path):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(GSM8K_TEST_URL, cache_path)
    with open(cache_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["question"] for r in rows[:num_questions]]


def render_prompts(
    questions: list[str], variant: str, effort: float, tokenizer
) -> list[list[int]]:
    from sglang.tml.renderer import render_tml_messages

    out: list[list[int]] = []
    for q in questions:
        messages: list[dict] = []
        if variant == "sys":
            messages.append({"role": "system", "content": f"Thinking effort level: {effort}"})
        messages.append({"role": "user", "content": q})
        # add_generation_prompt=False -> end at <|end_message|>; model emits the assistant turn.
        out.append(
            render_tml_messages(messages, tokenizer, add_generation_prompt=False)
        )
    return out


def drive_one(url: str, ids: list[int], temperature: float, max_new_tokens: int, eot: int):
    """One /generate request. Returns (completion_tokens, spec_verify_ct) or None on error."""
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop_token_ids": [eot],
            "ignore_eos": False,
        },
        "stream": False,
    }
    try:
        r = requests.post(f"{url}/generate", json=payload, timeout=2400)
        r.raise_for_status()
        body = r.json()
        obj = body[0] if isinstance(body, list) else body
        mi = obj["meta_info"]
        return int(mi.get("completion_tokens", 0)), int(mi.get("spec_verify_ct", 0) or 0)
    except Exception as exc:  # noqa: BLE001 -- one bad request must not kill the sweep
        print(f"  [warn] request failed: {exc!r}", flush=True)
        return None


def run_variant(url, prompts, temperature, max_new_tokens, concurrency, eot, label):
    results: list[tuple[int, int] | None] = [None] * len(prompts)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut2i = {ex.submit(drive_one, url, ids, temperature, max_new_tokens, eot): i
                 for i, ids in enumerate(prompts)}
        done = 0
        for fut in as_completed(fut2i):
            results[fut2i[fut]] = fut.result()
            done += 1
            if done % 100 == 0 or done == len(prompts):
                print(f"  [{label}] {done}/{len(prompts)}", flush=True)
    wall = time.perf_counter() - t0
    ok = [r for r in results if r is not None]
    tot_ct = sum(c for c, _ in ok)
    tot_sv = sum(s for _, s in ok if s)
    return {
        "variant": label,
        "n": len(ok),
        "n_failed": len(prompts) - len(ok),
        "acc_length": (tot_ct / tot_sv) if tot_sv else 1.0,
        "total_tokens": tot_ct,
        "total_verify_steps": tot_sv,
        "wall_s": round(wall, 1),
        "tok_per_s": round(tot_ct / wall, 1) if wall else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--url", help="http://HOST:PORT of the SGLang server / router")
    grp.add_argument("--host", help="server host (used with --port)")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--variant", choices=["nosys", "sys", "both"], default="sys")
    ap.add_argument("--num-questions", type=int, default=1319)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="generation cap; stop token usually fires first (median gsm8k gen ~400 tok)")
    ap.add_argument("--effort", type=float, default=0.5,
                    help="sys-variant 'Thinking effort level: <effort>' system turn")
    ap.add_argument("--concurrency", type=int, default=128,
                    help="in-flight requests; 1 = faithful bs=1. acc_length is bs-invariant -> raise for speed")
    ap.add_argument("--data-cache", default=os.path.expanduser("~/.cache/gsm8k_test.jsonl"))
    ap.add_argument("--output", default=None, help="optional JSONL path to append result rows")
    args = ap.parse_args()

    url = (args.url or f"http://{args.host}:{args.port}").rstrip("/")
    variants = ["nosys", "sys"] if args.variant == "both" else [args.variant]
    print(f"[gsm8k-acclen] url={url} variants={variants} n={args.num_questions} "
          f"T={args.temperature} concurrency={args.concurrency} max_new_tokens={args.max_new_tokens}")

    tokenizer = build_inkling_tokenizer()
    eot = eot_token_id()
    questions = load_questions(args.num_questions, args.data_cache)
    print(f"[gsm8k-acclen] {len(questions)} questions; "
          f"stop=<|content_model_end_sampling|> ({eot})", flush=True)

    rows = []
    for v in variants:
        prompts = render_prompts(questions, v, args.effort, tokenizer)
        med = sorted(len(p) for p in prompts)[len(prompts) // 2]
        print(f"[gsm8k-acclen] variant={v}: {len(prompts)} prompts (median {med} tok) -> driving",
              flush=True)
        rows.append(run_variant(url, prompts, args.temperature, args.max_new_tokens,
                                args.concurrency, eot, v))

    try:
        import tabulate
        headers = ["variant", "N", "fail", "Acc Length", "gen tokens", "verify steps", "tok/s"]
        table = [[r["variant"], r["n"], r["n_failed"], f"{r['acc_length']:.3f}",
                  r["total_tokens"], r["total_verify_steps"], r["tok_per_s"]] for r in rows]
        print("\n" + tabulate.tabulate(table, headers=headers, tablefmt="pretty"))
    except ImportError:
        for r in rows:
            print(f"  {r['variant']}: acc_length={r['acc_length']:.3f} N={r['n']}")

    if args.output:
        with open(args.output, "a") as f:
            for r in rows:
                f.write(json.dumps({**r, "url": url, "temperature": args.temperature,
                                    "num_questions": args.num_questions}) + "\n")


if __name__ == "__main__":
    main()
