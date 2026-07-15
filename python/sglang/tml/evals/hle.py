"""Standalone HLE (Humanity's Last Exam) eval against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, TextIO

from openai import AsyncOpenAI, OpenAIError

HLE_JUDGE_PROMPT_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {model_response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {reference_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

_CORRECT_RE = re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)
DEFAULT_DATA_PATH = Path(__file__).with_name("hle_text.jsonl")
LOG_EVERY = 25
_REASONING_EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")


def _parse_reasoning_effort(value: str) -> float | str:
    if value in _REASONING_EFFORT_LEVELS:
        return value
    try:
        effort = float(value)
    except ValueError as e:
        choices = ", ".join(_REASONING_EFFORT_LEVELS)
        raise argparse.ArgumentTypeError(
            f"expected a number in [0, 1] or one of: {choices}"
        ) from e
    if not 0.0 <= effort <= 1.0:
        raise argparse.ArgumentTypeError("reasoning effort must be in [0, 1]")
    return effort


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def score_from_judge(judge_text: str) -> float:
    """Return 1.0 for ``correct: yes`` and 0.0 for ``correct: no``."""
    m = _CORRECT_RE.search(judge_text or "")
    if m is None:
        raise ValueError(
            f"no 'correct: yes|no' verdict in judge response: {judge_text!r}"
        )
    return 1.0 if m.group(1).lower() == "yes" else 0.0


def _content(resp) -> str:
    """Visible final text of a chat completion (excludes the reasoning channel)."""
    msg = resp.choices[0].message
    return (msg.content or "").strip()


async def run_one(
    ex: dict,
    model_client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    args,
) -> dict:
    out = {
        "id": ex.get("id"),
        "score": 0.0,
        "error": None,
        "model_tokens": 0,
        "finish": None,
    }
    model_kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": ex["messages"],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
    }
    if args.reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = args.reasoning_effort
    try:
        mr = await model_client.chat.completions.create(**model_kwargs)
    except (OpenAIError, asyncio.TimeoutError) as e:
        out["error"] = f"model: {type(e).__name__}: {str(e)[:160]}"
        return out

    model_response = _content(mr)
    out["finish"] = mr.choices[0].finish_reason
    if mr.usage is not None:
        out["model_tokens"] = mr.usage.completion_tokens

    judge_prompt = HLE_JUDGE_PROMPT_TEMPLATE.format(
        question=ex["question"],
        model_response=model_response,
        reference_answer=ex["reference_answer"],
    )
    try:
        jr = await judge_client.chat.completions.create(
            model=args.judge,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=args.judge_max_tokens,
            timeout=args.request_timeout,
        )
    except (OpenAIError, asyncio.TimeoutError) as e:
        out["error"] = f"judge: {type(e).__name__}: {str(e)[:160]}"
        return out

    try:
        out["score"] = score_from_judge(_content(jr))
    except ValueError as e:
        out["error"] = f"judge: {type(e).__name__}: {str(e)[:160]}"
    return out


async def amain(args) -> None:
    with DEFAULT_DATA_PATH.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise SystemExit("No HLE examples loaded.")
    print(f"Loaded {len(data)} HLE examples from {DEFAULT_DATA_PATH}", flush=True)
    print(
        f"temperature={args.temperature} max_tokens={args.max_tokens} "
        f"reasoning_effort={args.reasoning_effort} concurrency={args.concurrency}",
        flush=True,
    )

    # Do not retry long-running generations, which would duplicate requests.
    model_client = AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key, max_retries=0
    )
    judge_client = AsyncOpenAI(
        base_url=args.judge_base_url or args.base_url,
        api_key=args.api_key,
        max_retries=0,
    )
    out_path = Path(args.output or f"hle_results_{_safe_model_name(args.model)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tic = time.perf_counter()
    done = 0
    correct = 0.0
    errors = 0
    results: list[dict[str, Any]] = []
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    async def drain_one(future: asyncio.Task[dict[str, Any]], out_f: TextIO) -> None:
        nonlocal done, correct, errors
        r = await future
        results.append(r)
        done += 1
        correct += r["score"]
        errors += int(bool(r["error"]))
        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()
        if done % LOG_EVERY == 0 or done == len(data) or r["error"]:
            el = time.perf_counter() - tic
            tag = f"  ERR {r['error']}" if r["error"] else ""
            print(
                f"[{done}/{len(data)}] acc={correct / done:.4f} "
                f"errs={errors} last(id={str(r['id'])[:8]} score={r['score']:.0f} "
                f"toks={r['model_tokens']} finish={r['finish']}) {el:.0f}s{tag}",
                flush=True,
            )

    with out_path.open("w", encoding="utf-8") as out_f:
        for ex in data:
            pending.add(
                asyncio.create_task(run_one(ex, model_client, judge_client, args))
            )
            while len(pending) >= args.concurrency:
                done_tasks, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    await drain_one(task, out_f)

        while pending:
            done_tasks, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done_tasks:
                await drain_one(task, out_f)

    latency = time.perf_counter() - tic

    n = len(results)
    score_sum = sum(r["score"] for r in results)
    acc = score_sum / n if n else 0.0
    truncated = sum(1 for r in results if r["finish"] == "length")

    print("\n==== HLE results ====", flush=True)
    print(f"model:     {args.model}")
    print(f"judge:     {args.judge}")
    print(f"n:         {n}")
    print(f"correct:   {int(score_sum)}")
    print(f"accuracy:  {acc:.4f}")
    print(f"errors:    {errors}")
    print(f"truncated (finish=length): {truncated}")
    print(f"latency:   {latency:.1f}s")
    if errors:
        for r in results:
            if r["error"]:
                print(f"  e.g. error: {r['error']}")
                break

    print(f"\nPer-sample results: {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone HLE eval (OpenAI-compatible).")
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--judge-base-url", default=None, help="defaults to --base-url")
    p.add_argument("--judge", default=None, help="defaults to --model")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--temperature", type=float, default=1.0, help="HLE default is 1.0")
    p.add_argument(
        "--reasoning-effort",
        type=_parse_reasoning_effort,
        default=None,
        help=f"optional scalar in [0, 1] or one of: {', '.join(_REASONING_EFFORT_LEVELS)}",
    )
    p.add_argument("--max-tokens", type=int, default=131072)
    p.add_argument("--judge-max-tokens", type=int, default=2048)
    p.add_argument(
        "--request-timeout", type=float, default=3600.0, help="per-call timeout (s)"
    )
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument(
        "--limit", type=int, default=None, help="total sample cap for smoke runs"
    )
    p.add_argument("--output", default=None, help="JSONL path for per-sample results")
    args = p.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.judge is None:
        args.judge = args.model
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
