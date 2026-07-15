"""Standalone 1M Needle-in-a-Haystack eval against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from random import Random
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from transformers import AutoTokenizer

NEEDLE_TEMPLATE = "One of the special magic uuids for {key} is: {value}."

KEY_TO_VALUE_QUERY_TEMPLATE = (
    "A special magic uuid is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the uuid afterwards.\n"
    "{context}\n"
    "What is the special magic uuid for {query_key} mentioned in the provided text? "
    "The special magic uuid for {query_key} mentioned in the provided text is"
)

DEFAULT_CONTEXT_MIN = 2_000
DEFAULT_CONTEXT_MAX = 1_024_000
DEFAULT_CONTEXT_INTERVALS = 15
DEFAULT_DEPTHS = tuple(range(0, 101, 10))
DEFAULT_SAMPLES_PER_CELL = 10
DEFAULT_MAX_TOKENS = 128
DEFAULT_SEED = 42
TOKENS_PER_LINE_SAMPLES = 20
LOG_EVERY = 10
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


def generate_uuid(rng: Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def make_context_lengths(context_lengths: str | None) -> list[int]:
    if context_lengths:
        return parse_int_list(context_lengths)
    step = (DEFAULT_CONTEXT_MAX - DEFAULT_CONTEXT_MIN) / (DEFAULT_CONTEXT_INTERVALS - 1)
    return [
        round(DEFAULT_CONTEXT_MIN + i * step) for i in range(DEFAULT_CONTEXT_INTERVALS)
    ]


def build_token_counter(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def measure_tokens_per_line(rng: Random, count_tokens, num_samples: int) -> float:
    total = 0
    for _ in range(num_samples):
        line = NEEDLE_TEMPLATE.format(key=generate_uuid(rng), value=generate_uuid(rng))
        total += count_tokens(line + "\n")
    return total / max(num_samples, 1)


def build_prompt(
    key_value_pairs: list[tuple[str, str]],
    target_key: str,
) -> str:
    context = "\n".join(
        NEEDLE_TEMPLATE.format(key=key, value=value) for key, value in key_value_pairs
    )
    return KEY_TO_VALUE_QUERY_TEMPLATE.format(context=context, query_key=target_key)


def generate_sample(
    *,
    context_length: int,
    depth_percent: int,
    sample_index: int,
    rng: Random,
    count_tokens,
    tokens_per_line: float,
    max_tokens: int,
) -> dict[str, Any]:
    dummy_key = generate_uuid(rng)
    # Preserve the generator's deterministic RNG sequence.
    generate_uuid(rng)
    template_overhead = count_tokens(
        KEY_TO_VALUE_QUERY_TEMPLATE.format(context="", query_key=dummy_key)
    )
    available_tokens = context_length - template_overhead - max_tokens
    estimated_lines = max(3, int(available_tokens / tokens_per_line))

    key_value_pairs = [
        (generate_uuid(rng), generate_uuid(rng)) for _ in range(estimated_lines)
    ]
    target_index = int((depth_percent / 100) * max(len(key_value_pairs) - 1, 0))
    target_key, target_value = key_value_pairs[target_index]

    prompt = build_prompt(key_value_pairs, target_key)
    prompt_tokens = count_tokens(prompt)
    total_tokens = prompt_tokens + max_tokens

    remaining_adjustments = 50
    while (
        total_tokens > context_length
        and len(key_value_pairs) > 3
        and remaining_adjustments > 0
    ):
        if len(key_value_pairs) - 1 != target_index:
            key_value_pairs.pop()
        else:
            key_value_pairs.pop(-2)
            target_index = len(key_value_pairs) - 1
        prompt = build_prompt(key_value_pairs, target_key)
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_tokens
        remaining_adjustments -= 1

    while total_tokens < context_length - tokens_per_line and remaining_adjustments > 0:
        key_value_pairs.append((generate_uuid(rng), generate_uuid(rng)))
        prompt = build_prompt(key_value_pairs, target_key)
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_tokens
        remaining_adjustments -= 1

    if total_tokens > context_length and len(key_value_pairs) > 3:
        if len(key_value_pairs) - 1 != target_index:
            key_value_pairs.pop()
        prompt = build_prompt(key_value_pairs, target_key)
        prompt_tokens = count_tokens(prompt)
        total_tokens = prompt_tokens + max_tokens

    sample_id = (
        f"niah_uuid_key_to_value_{context_length}_{depth_percent}_{rng.getrandbits(32)}"
    )
    return {
        "id": sample_id,
        "context_length": context_length,
        "depth_percent": depth_percent,
        "sample_index": sample_index,
        "prompt": prompt,
        "expected": target_value,
        "target_key": target_key,
        "prompt_tokens": prompt_tokens,
        "budgeted_total_tokens": total_tokens,
        "num_haystack_lines": len(key_value_pairs),
    }


def iter_samples(args: argparse.Namespace, count_tokens):
    context_lengths = make_context_lengths(args.context_lengths)
    depths = parse_int_list(args.depths)
    measure_rng = Random(DEFAULT_SEED + 1)
    tokens_per_line = measure_tokens_per_line(
        measure_rng, count_tokens, TOKENS_PER_LINE_SAMPLES
    )
    rng = Random(DEFAULT_SEED)
    emitted = 0

    for context_length in context_lengths:
        for depth_percent in depths:
            for sample_index in range(args.samples_per_cell):
                if args.limit is not None and emitted >= args.limit:
                    return
                yield generate_sample(
                    context_length=context_length,
                    depth_percent=depth_percent,
                    sample_index=sample_index,
                    rng=rng,
                    count_tokens=count_tokens,
                    tokens_per_line=tokens_per_line,
                    max_tokens=args.max_tokens,
                )
                emitted += 1


def extract_message_text(message: Any) -> tuple[str, int]:
    content = message.content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                text_parts.append(str(item.text))
        content_text = "".join(text_parts)
    else:
        content_text = content or ""

    reasoning = getattr(message, "reasoning_content", None) or getattr(
        message, "reasoning", None
    )
    reasoning_chars = len(str(reasoning)) if reasoning is not None else 0
    return content_text.strip(), reasoning_chars


def grade_response(expected: str, response_text: str) -> tuple[bool, str]:
    text = response_text.strip()
    hit = expected.lower() in text.lower()
    # Surface the line that carries the answer for debugging; the model often
    # emits a preamble line before the value, so scoring can't key off line 0.
    matched_line = next(
        (
            line.strip()
            for line in text.splitlines()
            if expected.lower() in line.lower()
        ),
        text.splitlines()[0].strip() if text else "",
    )
    return hit, matched_line


async def run_one(
    sample: dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": sample["id"],
        "context_length": sample["context_length"],
        "depth_percent": sample["depth_percent"],
        "sample_index": sample["sample_index"],
        "target_key": sample["target_key"],
        "expected": sample["expected"],
        "prompt_tokens": sample["prompt_tokens"],
        "budgeted_total_tokens": sample["budgeted_total_tokens"],
        "num_haystack_lines": sample["num_haystack_lines"],
        "response": "",
        "matched_line": "",
        "correct": False,
        "error": None,
        "finish": None,
        "completion_tokens": None,
        "total_tokens": None,
        "reasoning_chars": 0,
    }
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": sample["prompt"]}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
    }
    if args.reasoning_effort is not None:
        kwargs["reasoning_effort"] = args.reasoning_effort

    try:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        response_text, reasoning_chars = extract_message_text(choice.message)
        correct, extracted = grade_response(sample["expected"], response_text)
        out["response"] = response_text
        out["matched_line"] = extracted
        out["correct"] = bool(correct)
        out["finish"] = choice.finish_reason
        out["reasoning_chars"] = reasoning_chars
        if response.usage is not None:
            out["completion_tokens"] = response.usage.completion_tokens
            out["total_tokens"] = response.usage.total_tokens
    except (OpenAIError, asyncio.TimeoutError) as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:240]}"
    return out


def summarize(results: list[dict[str, Any]], elapsed: float, model: str) -> None:
    print("\n==== NIAH 1M key-to-value results ====", flush=True)
    print(f"model:    {model}")
    print(f"n:        {len(results)}")
    print(f"errors:   {sum(1 for row in results if row['error'])}")
    print(f"latency:  {elapsed:.1f}s")
    if not results:
        return

    total_correct = sum(1 for row in results if row["correct"])
    print(
        f"overall:  acc={total_correct / len(results):.4f} ({total_correct}/{len(results)})"
    )

    for context_length in sorted({row["context_length"] for row in results}):
        rows = [row for row in results if row["context_length"] == context_length]
        correct = sum(1 for row in rows if row["correct"])
        print(
            f"  ctx={context_length:7d}  acc={correct / len(rows):.4f} ({correct}/{len(rows)})"
        )

    print("\nBy depth:")
    for depth_percent in sorted({row["depth_percent"] for row in results}):
        rows = [row for row in results if row["depth_percent"] == depth_percent]
        correct = sum(1 for row in rows if row["correct"])
        print(
            f"  depth={depth_percent:3d}%  acc={correct / len(rows):.4f} ({correct}/{len(rows)})"
        )


async def amain(args: argparse.Namespace) -> None:
    tokenizer_name = args.tokenizer or args.model
    count_tokens = build_token_counter(tokenizer_name)
    context_lengths = make_context_lengths(args.context_lengths)
    depths = parse_int_list(args.depths)
    if any(length <= 0 for length in context_lengths):
        raise SystemExit("--context-lengths values must be positive")
    if any(not 0 <= depth <= 100 for depth in depths):
        raise SystemExit("--depths values must be between 0 and 100")
    total = len(context_lengths) * len(depths) * args.samples_per_cell
    if args.limit is not None:
        total = min(total, args.limit)

    print(
        f"Running NIAH key-to-value: {len(context_lengths)} context lengths, "
        f"{len(depths)} depths, n={total}",
        flush=True,
    )
    print(
        f"contexts={context_lengths[0]}..{context_lengths[-1]} "
        f"max_tokens={args.max_tokens} tokenizer={tokenizer_name}",
        flush=True,
    )

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    out_path = Path(
        args.output or f"niah_1m_results_{_safe_model_name(args.model)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tic = time.perf_counter()
    done = 0
    correct = 0
    errors = 0
    results: list[dict[str, Any]] = []
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    async def drain_one(f) -> None:
        nonlocal done, correct, errors
        row = await f
        results.append(row)
        done += 1
        correct += int(row["correct"])
        errors += int(bool(row["error"]))
        with out_path.open("a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
        if done % LOG_EVERY == 0 or done == total or row["error"]:
            elapsed = time.perf_counter() - tic
            tag = f" ERR {row['error']}" if row["error"] else ""
            print(
                f"[{done}/{total}] acc={correct / done:.4f} errs={errors} "
                f"last(ctx={row['context_length']} depth={row['depth_percent']} "
                f"correct={int(row['correct'])} finish={row['finish']}) {elapsed:.0f}s{tag}",
                flush=True,
            )

    out_path.write_text("", encoding="utf-8")
    samples = iter(iter_samples(args, count_tokens))
    # Max-context sample construction is CPU-heavy; keep in-flight HTTP requests responsive.
    while (sample := await asyncio.to_thread(next, samples, None)) is not None:
        pending.add(asyncio.create_task(run_one(sample, client, args)))
        while len(pending) >= args.concurrency:
            done_tasks, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done_tasks:
                await drain_one(task)

    while pending:
        done_tasks, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done_tasks:
            await drain_one(task)

    summarize(results, time.perf_counter() - tic, args.model)
    print(f"\nPer-sample results: {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--context-lengths",
        default=None,
        help="comma-separated context lengths; defaults to the 2k-to-1,024k grid",
    )
    parser.add_argument(
        "--depths",
        default=",".join(str(x) for x in DEFAULT_DEPTHS),
        help="comma-separated depth percentages",
    )
    parser.add_argument(
        "--samples-per-cell", type=int, default=DEFAULT_SAMPLES_PER_CELL
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Hugging Face tokenizer path or model ID; defaults to --model",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        type=_parse_reasoning_effort,
        default=None,
        help=f"optional scalar in [0, 1] or one of: {', '.join(_REASONING_EFFORT_LEVELS)}",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--limit", type=int, default=None, help="total sample cap for smoke runs"
    )
    parser.add_argument(
        "--output", default=None, help="JSONL path for per-sample results"
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.samples_per_cell < 1:
        raise SystemExit("--samples-per-cell must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
