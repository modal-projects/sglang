"""Standalone BFCL eval against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from datasets import disable_progress_bars, load_dataset
from openai import AsyncOpenAI, OpenAIError

DATASET_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 256 * 1024
LOG_EVERY = 25
_REASONING_EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")

disable_progress_bars()

CATEGORY_FILES = {
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
    "live_parallel": "BFCL_v3_live_parallel.json",
    "live_parallel_multiple": "BFCL_v3_live_parallel_multiple.json",
    "live_irrelevance": "BFCL_v3_live_irrelevance.json",
    "live_relevance": "BFCL_v3_live_relevance.json",
}

ANSWER_FILES = {
    "live_simple": "possible_answer/BFCL_v3_live_simple.json",
    "live_multiple": "possible_answer/BFCL_v3_live_multiple.json",
    "live_parallel": "possible_answer/BFCL_v3_live_parallel.json",
    "live_parallel_multiple": "possible_answer/BFCL_v3_live_parallel_multiple.json",
}

EXACT_CATEGORIES = tuple(ANSWER_FILES)
ALL_CATEGORIES = tuple(CATEGORY_FILES)
SAFE_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")

# Upstream has one BFCL_v3_live_multiple prompt/answer ID typo at the final row.
KNOWN_ROW_ORDER_ID_MISMATCHES = {
    ("live_multiple", "live_multiple_1052-79-0", "live_multiple_1052-279-0"),
}


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


def _read_jsonl_from_hf(filename: str) -> list[dict[str, Any]]:
    dataset = load_dataset(
        DATASET_REPO,
        data_files=filename,
        split="train",
    )
    return [dict(row) for row in dataset]


def parse_categories(value: str) -> list[str]:
    if value == "all":
        return list(ALL_CATEGORIES)
    categories = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(categories) - set(ALL_CATEGORIES))
    if unknown:
        raise ValueError(
            f"Unknown BFCL categories: {unknown}. Available: {list(ALL_CATEGORIES)}"
        )
    return categories


def _messages_from_bfcl_question(question: Any) -> list[dict[str, Any]]:
    # BFCL single-turn rows store question as [[{"role": "user", ...}]].
    if isinstance(question, list) and question and isinstance(question[0], list):
        return question[0]
    if isinstance(question, list):
        return question
    return [{"role": "user", "content": str(question)}]


def load_samples(
    categories: list[str], offset: int, limit: int | None
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for category in categories:
        prompts = _read_jsonl_from_hf(CATEGORY_FILES[category])
        answers: list[dict[str, Any]] = []
        answers_by_id: dict[str, Any] = {}
        if category in ANSWER_FILES:
            answers = _read_jsonl_from_hf(ANSWER_FILES[category])
            if len(answers) != len(prompts):
                raise RuntimeError(
                    f"{category}: prompt/answer length mismatch "
                    f"({len(prompts)} prompts, {len(answers)} answers)"
                )
            answers_by_id = {row["id"]: row["ground_truth"] for row in answers}
            if len(answers_by_id) != len(answers):
                raise RuntimeError(
                    f"{category}: duplicate answer IDs in {ANSWER_FILES[category]}"
                )

        for idx, row in enumerate(prompts):
            expected = None
            if category in ANSWER_FILES:
                if row["id"] in answers_by_id:
                    expected = answers_by_id[row["id"]]
                else:
                    answer_row = answers[idx]
                    mismatch = (category, row["id"], answer_row["id"])
                    if mismatch not in KNOWN_ROW_ORDER_ID_MISMATCHES:
                        raise RuntimeError(
                            f"Missing possible_answer for {category} id={row['id']} "
                            f"at index={idx}; row-order answer id={answer_row['id']}"
                        )
                    expected = answer_row["ground_truth"]
            samples.append(
                {
                    "id": row["id"],
                    "category": category,
                    "messages": _messages_from_bfcl_question(row["question"]),
                    "functions": row["function"],
                    "expected": expected,
                }
            )

    return samples[offset : None if limit is None else offset + limit]


def safe_tool_name(name: str, used: dict[str, str]) -> str:
    base = SAFE_TOOL_NAME_RE.sub("_", name)
    if not base:
        base = "tool"
    if len(base) > 64:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        base = f"{base[:55]}_{digest}"

    candidate = base
    suffix = 1
    while candidate in used and used[candidate] != name:
        suffix_text = f"_{suffix}"
        candidate = f"{base[: 64 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    used[candidate] = name
    return candidate


def _json_schema_type(bfcl_type: Any) -> str | list[str] | None:
    if bfcl_type == "dict":
        return "object"
    if bfcl_type in ("array", "tuple"):
        return "array"
    if bfcl_type == "integer":
        return "integer"
    if bfcl_type == "float":
        return "number"
    if bfcl_type == "boolean":
        return "boolean"
    if bfcl_type == "string":
        return "string"
    if bfcl_type == "any":
        return None
    return bfcl_type if isinstance(bfcl_type, str) else None


def bfcl_schema_to_json_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}

    out: dict[str, Any] = {}
    json_type = _json_schema_type(schema.get("type"))
    if json_type is not None:
        out["type"] = json_type
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if "default" in schema:
        out["default"] = schema["default"]
    if "properties" in schema and isinstance(schema["properties"], dict):
        out["type"] = "object"
        out["properties"] = {
            str(name): bfcl_schema_to_json_schema(prop)
            for name, prop in schema["properties"].items()
        }
    if "required" in schema and isinstance(schema["required"], list):
        out["required"] = [str(item) for item in schema["required"]]
    if "items" in schema:
        out["items"] = bfcl_schema_to_json_schema(schema["items"])
    return out


def make_tools(
    functions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    used: dict[str, str] = {}
    tools: list[dict[str, Any]] = []
    for fn in functions:
        original_name = fn["name"]
        name = safe_tool_name(original_name, used)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": fn["description"],
                    "parameters": bfcl_schema_to_json_schema(fn["parameters"]),
                },
            }
        )
    return tools, used


def normalize_name(name: str) -> str:
    name = name.strip()
    if name.startswith("functions."):
        name = name[len("functions.") :]
    return name.replace(".", "_").lower()


def standardize_string(value: str) -> str:
    return re.sub(r"[ ,./\\_*^-]", "", value).lower().replace("'", '"')


def values_equal(predicted: Any, expected: Any) -> bool:
    if expected == "" or expected is None:
        return predicted == expected
    if predicted == expected:
        return True
    try:
        if float(predicted) == float(expected):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(predicted, str) and isinstance(expected, str):
        return standardize_string(predicted) == standardize_string(expected)
    if isinstance(predicted, list) and isinstance(expected, list):
        return len(predicted) == len(expected) and all(
            value_matches(p_item, e_item) for p_item, e_item in zip(predicted, expected)
        )
    if isinstance(predicted, dict) and isinstance(expected, dict):
        return dict_matches(predicted, expected)
    return str(predicted).lower() == str(expected).lower()


def value_matches(predicted: Any, acceptable: Any) -> bool:
    if isinstance(acceptable, list):
        if not acceptable:
            return predicted == []
        return any(values_equal(predicted, item) for item in acceptable)
    if isinstance(acceptable, dict):
        return isinstance(predicted, dict) and dict_matches(predicted, acceptable)
    return values_equal(predicted, acceptable)


def _missing_is_optional(acceptable: Any) -> bool:
    if acceptable == "" or acceptable is None:
        return True
    if isinstance(acceptable, list):
        return any(item == "" or item is None for item in acceptable)
    return False


def dict_matches(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key in predicted:
        if key not in expected:
            return False
    for key, acceptable in expected.items():
        if key not in predicted:
            if _missing_is_optional(acceptable):
                continue
            return False
        if not value_matches(predicted[key], acceptable):
            return False
    return True


def call_matches_expected(
    predicted: dict[str, Any],
    expected_call: dict[str, Any],
) -> bool:
    if len(expected_call) != 1:
        return False

    expected_name = next(iter(expected_call))
    expected_args = expected_call[expected_name]
    if normalize_name(predicted["name"]) != normalize_name(expected_name):
        return False
    if not isinstance(predicted["args"], dict) or not isinstance(expected_args, dict):
        return False

    predicted_args = predicted["args"]

    for param in predicted_args:
        if param not in expected_args:
            return False
    for param, acceptable in expected_args.items():
        if param not in predicted_args:
            if _missing_is_optional(acceptable):
                continue
            return False
        if not value_matches(predicted_args[param], acceptable):
            return False
    return True


def exact_calls_match(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    category: str,
) -> bool:
    if len(predicted) != len(expected):
        return False

    if "parallel" not in category:
        return all(
            call_matches_expected(pred, exp) for pred, exp in zip(predicted, expected)
        )

    matched_pred: set[int] = set()
    for exp in expected:
        found = False
        for i, pred in enumerate(predicted):
            if i in matched_pred:
                continue
            if call_matches_expected(pred, exp):
                matched_pred.add(i)
                found = True
                break
        if not found:
            return False
    return True


def extract_tool_calls(
    message: Any,
    safe_to_original: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    tool_calls = (
        message["tool_calls"] if isinstance(message, dict) else message.tool_calls
    )
    if tool_calls is None:
        tool_calls = []
    for tool_call in tool_calls:
        function = (
            tool_call["function"] if isinstance(tool_call, dict) else tool_call.function
        )
        safe_name = function["name"] if isinstance(function, dict) else function.name
        arguments = (
            function["arguments"] if isinstance(function, dict) else function.arguments
        )
        if arguments is None:
            arguments = "{}"
        original_name = safe_to_original.get(safe_name, safe_name)
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                errors.append(
                    f"{safe_name}: arguments decoded to {type(args).__name__}, not dict"
                )
                args = {}
        except json.JSONDecodeError as e:
            errors.append(f"{safe_name}: JSONDecodeError: {str(e)[:120]}")
            args = {}
        calls.append({"name": original_name, "safe_name": safe_name, "args": args})
    return calls, errors


def grade_sample(sample: dict[str, Any], predicted: list[dict[str, Any]]) -> bool:
    category = sample["category"]
    if category == "live_irrelevance":
        return len(predicted) == 0
    if category == "live_relevance":
        declared = {normalize_name(fn["name"]) for fn in sample["functions"]}
        return any(normalize_name(call["name"]) in declared for call in predicted)
    expected = sample["expected"]
    return exact_calls_match(predicted, expected, category)


async def run_one(
    sample: dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tools, safe_to_original = make_tools(sample["functions"])
    out: dict[str, Any] = {
        "id": sample["id"],
        "category": sample["category"],
        "question": sample["messages"],
        "expected": sample["expected"],
        "predicted": [],
        "correct": False,
        "error": None,
        "finish": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": sample["messages"],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
    }
    if args.reasoning_effort is not None:
        kwargs["reasoning_effort"] = args.reasoning_effort
    try:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        predicted, parse_errors = extract_tool_calls(message, safe_to_original)
        out["predicted"] = predicted
        out["tool_parse_errors"] = parse_errors
        out["content"] = message.content or ""
        out["finish"] = choice.finish_reason
        if response.usage is not None:
            out["prompt_tokens"] = response.usage.prompt_tokens
            out["completion_tokens"] = response.usage.completion_tokens
            out["total_tokens"] = response.usage.total_tokens
        out["correct"] = bool(not parse_errors and grade_sample(sample, predicted))
    except (OpenAIError, asyncio.TimeoutError) as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:240]}"
    return out


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def print_summary(
    args: argparse.Namespace, results: list[dict[str, Any]], elapsed: float
) -> None:
    per_category: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        per_category.setdefault(result["category"], []).append(result)

    print("\n==== BFCL results ====", flush=True)
    print(f"model:    {args.model}")
    print(f"n:        {len(results)}")
    print(f"errors:   {sum(1 for r in results if r['error'])}")
    print(f"latency:  {elapsed:.1f}s")

    category_scores: list[float] = []
    exact_correct = 0
    exact_total = 0
    for category in sorted(per_category):
        rows = per_category[category]
        correct = sum(1 for row in rows if row["correct"])
        acc = correct / len(rows) if rows else 0.0
        category_scores.append(acc)
        if category in EXACT_CATEGORIES:
            exact_correct += correct
            exact_total += len(rows)
        label = "relevance_call_rate" if category == "live_relevance" else "acc"
        print(f"  {category:24s} {label}={acc:.4f} ({correct}/{len(rows)})")

    if exact_total:
        print(
            f"  {'EXACT_CALLS':24s} acc={exact_correct / exact_total:.4f} "
            f"({exact_correct}/{exact_total})"
        )
    if category_scores:
        print(
            f"  {'ALL_LIVE_MACRO':24s} acc={sum(category_scores) / len(category_scores):.4f}"
        )


async def amain(args: argparse.Namespace) -> None:
    categories = parse_categories(args.categories)
    print(
        f"Loading BFCL categories={categories} offset={args.offset} limit={args.limit} ...",
        flush=True,
    )
    samples = load_samples(categories, args.offset, args.limit)
    print(f"  {len(samples)} samples", flush=True)
    if not samples:
        raise SystemExit("No samples loaded.")
    print(
        f"temperature={args.temperature} max_tokens={args.max_tokens} "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=0,
    )
    out_path = Path(args.output or f"bfcl_results_{_safe_model_name(args.model)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tic = time.perf_counter()
    done = 0
    correct = 0
    errors = 0
    results: list[dict[str, Any]] = []
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    with out_path.open("w", encoding="utf-8") as f:

        async def drain_one(task: asyncio.Task[dict[str, Any]]) -> None:
            nonlocal done, correct, errors
            row = await task
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            correct += int(row["correct"])
            errors += int(bool(row["error"]))
            if done % LOG_EVERY == 0 or done == len(samples) or row["error"]:
                elapsed = time.perf_counter() - tic
                print(
                    f"[{done}/{len(samples)}] acc={correct / done:.4f} "
                    f"errs={errors} last(id={str(row['id'])[:24]} "
                    f"cat={row['category']} correct={int(row['correct'])}) {elapsed:.0f}s",
                    flush=True,
                )

        for sample in samples:
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

    print_summary(args, results, time.perf_counter() - tic)
    print(f"\nPer-sample results: {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone BFCL eval (OpenAI-compatible native tools)."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--categories",
        default="all",
        help="all or comma-separated BFCL live category names",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="total sample cap for smoke runs"
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        type=_parse_reasoning_effort,
        default=None,
        help=f"optional scalar in [0, 1] or one of: {', '.join(_REASONING_EFFORT_LEVELS)}",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--output", default=None, help="JSONL path for per-sample results"
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.offset < 0:
        raise SystemExit("--offset must be >= 0")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
