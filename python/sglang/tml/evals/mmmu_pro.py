"""Standalone MMMU-Pro eval against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import ast
import base64
import hashlib
import io
import json
import os
import random
import re
import string
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from PIL import Image

CHOICE_LETTERS = string.ascii_uppercase
DEFAULT_DATASET = "MMMU/MMMU_Pro"
DEFAULT_SPLIT = "all"
DEFAULT_MAX_TOKENS = 128 * 1024
DEFAULT_TEMPERATURE = 1.0
LOG_EVERY = 25
_REASONING_EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")

ALL_SPLITS = ("standard10", "vision", "standard4")
HEADLINE_SPLITS = ("standard10", "vision")

SPLIT_TO_HF_CONFIG = {
    "standard10": "standard (10 options)",
    "standard4": "standard (4 options)",
    "vision": "vision",
}

SPLIT_GROUPS = {
    "standard10": ("standard10",),
    "standard4": ("standard4",),
    "vision": ("vision",),
    "headline": HEADLINE_SPLITS,
    "all": ALL_SPLITS,
}

SPLIT_TO_EVAL_NAME = {
    "standard10": "mmmu-pro-standard10",
    "standard4": "mmmu-pro-standard4",
    "vision": "mmmu-pro-vision",
}

STANDARD_PROMPT = (
    "Answer the following multiple-choice question. The last line of your response should be of "
    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. "
    "Think step by step before answering."
)

VISION_PROMPT = (
    "Write out the multiple-choice question in the image and then solve it. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER "
    "is one of options. Think step by step before answering."
)


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


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _completion_seed(sample_id: str, split: str) -> int:
    task_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{DEFAULT_DATASET}/{SPLIT_TO_HF_CONFIG[split]}/test/{sample_id}",
    )
    return _stable_seed(f"{task_id}/0")


def _load_dataset(split: str):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on the target environment
        raise SystemExit("MMMU-Pro loading requires `datasets` and `Pillow`.") from e

    return load_dataset(DEFAULT_DATASET, SPLIT_TO_HF_CONFIG[split], split="test")


def _iter_dataset_rows(
    datasets: list[tuple[str, Any]], limit: int | None
) -> Iterator[tuple[str, Any, int]]:
    if limit is None:
        for split, dataset in datasets:
            for idx in range(len(dataset)):
                yield split, dataset[idx], idx
        return

    emitted = 0
    for idx in range(max(len(dataset) for _, dataset in datasets)):
        for split, dataset in datasets:
            if idx >= len(dataset):
                continue
            yield split, dataset[idx], idx
            emitted += 1
            if emitted >= limit:
                return


def parse_options(row: dict[str, Any]) -> list[str]:
    options = ast.literal_eval(row["options"])
    if not isinstance(options, list):
        raise ValueError(
            f"expected options to parse to list, got {type(options).__name__}"
        )
    if not 2 <= len(options) <= len(CHOICE_LETTERS):
        raise ValueError(
            f"expected 2-{len(CHOICE_LETTERS)} options, got {len(options)}"
        )
    return [str(option) for option in options]


def parse_answer(row: dict[str, Any], option_count: int) -> str:
    answer = str(row["answer"]).strip().upper()
    if answer not in CHOICE_LETTERS[:option_count]:
        raise ValueError(f"answer {answer!r} is not valid for {option_count} options")
    return answer


def pil_image_to_data_url(image: Image.Image) -> str:
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image)
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def image_to_data_url(image: Any) -> str:
    if image is None:
        raise ValueError("image is missing")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            raw = image["bytes"]
            if not isinstance(raw, bytes):
                raise TypeError(f"image bytes must be bytes, got {type(raw).__name__}")
            with Image.open(io.BytesIO(raw)) as decoded:
                return pil_image_to_data_url(decoded)
        if image.get("path") is not None:
            with Image.open(image["path"]) as decoded:
                return pil_image_to_data_url(decoded)
        if image.get("src") is not None:
            return str(image["src"])
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as decoded:
            return pil_image_to_data_url(decoded)
    if isinstance(image, str):
        return image
    if isinstance(image, Image.Image):
        return pil_image_to_data_url(image)
    raise ValueError(f"unsupported image type: {type(image).__name__}")


def image_content(image: Any) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}


def text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def make_standard_sample(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    options = parse_options(row)
    answer = parse_answer(row, len(options))
    all_choices = list(CHOICE_LETTERS[: len(options)])
    index2ans = dict(zip(all_choices, options, strict=True))
    options_prompt = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(all_choices, options, strict=True)
    )
    prompt = f"{row['question']}\n{options_prompt}\n\n{STANDARD_PROMPT}"

    content: list[dict[str, Any]] = []
    for image_idx in range(1, 8):
        image = row.get(f"image_{image_idx}")
        if image is not None:
            content.append(image_content(image))
    content.append(text_content(prompt))

    return {
        "id": str(row.get("id") or idx),
        "eval_name": SPLIT_TO_EVAL_NAME[split],
        "split": split,
        "subject": str(row.get("subject", "")),
        "question": str(row.get("question", "")),
        "options": options,
        "answer": answer,
        "all_choices": all_choices,
        "index2ans": index2ans,
        "image_count": len(content) - 1,
        "messages": [{"role": "user", "content": content}],
    }


def make_vision_sample(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    options = parse_options(row)
    answer = parse_answer(row, len(options))
    all_choices = list(CHOICE_LETTERS[: len(options)])
    index2ans = dict(zip(all_choices, options, strict=True))
    content = [image_content(row["image"]), text_content(VISION_PROMPT)]
    return {
        "id": str(row.get("id") or idx),
        "eval_name": SPLIT_TO_EVAL_NAME[split],
        "split": split,
        "subject": str(row.get("subject", "")),
        "question": "",
        "options": options,
        "answer": answer,
        "all_choices": all_choices,
        "index2ans": index2ans,
        "image_count": 1,
        "messages": [{"role": "user", "content": content}],
    }


def make_sample(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    if split == "vision":
        return make_vision_sample(row, split, idx)
    return make_standard_sample(row, split, idx)


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


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    *,
    seed: int,
) -> tuple[str, str]:
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0], "answer_prefix"

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if not candidates and len(response.split()) > 5:
        for index, answer_text in index2ans.items():
            if answer_text.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if not candidates:
        return random.Random(seed).choice(all_choices), "random_fallback"

    if len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for candidate in candidates:
                    start_indexes.append(response.rfind(f"({candidate})"))
            else:
                for candidate in candidates:
                    start_indexes.append(response.rfind(f" {candidate} "))
        else:
            for candidate in candidates:
                start_indexes.append(
                    response.lower().rfind(index2ans[candidate].lower())
                )
        return candidates[
            max(range(len(start_indexes)), key=start_indexes.__getitem__)
        ], "last_match"

    return candidates[0], "single_match"


async def run_one(
    sample: dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": sample["id"],
        "eval_name": sample["eval_name"],
        "split": sample["split"],
        "subject": sample["subject"],
        "question": sample["question"],
        "options": sample["options"],
        "answer": sample["answer"],
        "image_count": sample["image_count"],
        "response": "",
        "predicted": None,
        "parse_method": None,
        "correct": False,
        "correct_format": 0.0,
        "exact_match": 0.0,
        "scalar_reward": 0.0,
        "answer_is_valid_choice": 0.0,
        "answer_prefix_match": 0.0,
        "fallback_parse": 0.0,
        "random_fallback": 0.0,
        "error": None,
        "finish": None,
        "completion_tokens": None,
        "total_tokens": None,
        "reasoning_chars": 0,
    }
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": sample["messages"],
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
        predicted, parse_method = parse_multi_choice_response(
            response_text,
            sample["all_choices"],
            sample["index2ans"],
            seed=_completion_seed(sample["id"], sample["split"]),
        )
        out["response"] = response_text
        out["predicted"] = predicted
        out["parse_method"] = parse_method
        out["correct"] = predicted == sample["answer"]
        out["correct_format"] = 1.0 if parse_method == "answer_prefix" else 0.0
        out["exact_match"] = 1.0 if out["correct"] else 0.0
        out["scalar_reward"] = out["exact_match"]
        out["answer_is_valid_choice"] = 1.0
        out["answer_prefix_match"] = out["correct_format"]
        out["fallback_parse"] = 1.0 - out["answer_prefix_match"]
        out["random_fallback"] = 1.0 if parse_method == "random_fallback" else 0.0
        out["finish"] = choice.finish_reason
        out["reasoning_chars"] = reasoning_chars
        if response.usage is not None:
            out["completion_tokens"] = response.usage.completion_tokens
            out["total_tokens"] = response.usage.total_tokens
    except (OpenAIError, asyncio.TimeoutError) as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:240]}"
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for eval_name in sorted({row["eval_name"] for row in results}):
        rows = [row for row in results if row["eval_name"] == eval_name]
        metrics[f"{eval_name}/exact_match:mean"] = _mean_field(rows, "exact_match")
        metrics[f"{eval_name}/scalar_reward:mean"] = _mean_field(rows, "scalar_reward")
        metrics[f"{eval_name}/correct_format:mean"] = _mean_field(
            rows, "correct_format"
        )
        metrics[f"{eval_name}/answer_is_valid_choice:mean"] = _mean_field(
            rows, "answer_is_valid_choice"
        )
        metrics[f"{eval_name}/random_fallback:mean"] = _mean_field(
            rows, "random_fallback"
        )

        total_tokens = _mean(
            [
                float(row["total_tokens"])
                for row in rows
                if row["total_tokens"] is not None
            ]
        )
        completion_tokens = _mean(
            [
                float(row["completion_tokens"])
                for row in rows
                if row["completion_tokens"] is not None
            ]
        )
        if total_tokens is not None:
            metrics[f"{eval_name}/n_total_tokens:mean"] = total_tokens
        if completion_tokens is not None:
            metrics[f"{eval_name}/n_action_tokens:mean"] = completion_tokens

    standard10 = SPLIT_TO_EVAL_NAME["standard10"]
    vision = SPLIT_TO_EVAL_NAME["vision"]
    for metric in ("exact_match:mean", "n_total_tokens:mean"):
        values = [
            metrics[f"{eval_name}/{metric}"]
            for eval_name in (standard10, vision)
            if f"{eval_name}/{metric}" in metrics
        ]
        if len(values) == 2:
            metrics[f"mmmu-pro/{metric}"] = sum(values) / 2
    return metrics


def summarize(
    results: list[dict[str, Any]], elapsed: float, model: str, split: str
) -> None:
    print(f"\n==== MMMU-Pro {split} results ====", flush=True)
    print(f"model:    {model}")
    print(f"n:        {len(results)}")
    print(f"errors:   {sum(1 for row in results if row['error'])}")
    print(f"latency:  {elapsed:.1f}s")
    if not results:
        return

    total_correct = sum(1 for row in results if row["correct"])
    truncated = sum(1 for row in results if row["finish"] == "length")
    random_fallback = sum(
        1 for row in results if row["parse_method"] == "random_fallback"
    )
    answer_prefix = sum(1 for row in results if row["parse_method"] == "answer_prefix")
    print(
        f"overall:  acc={total_correct / len(results):.4f} ({total_correct}/{len(results)})"
    )
    print(f"finish=length: {truncated}")
    print(f"answer_prefix: {answer_prefix}")
    print(f"random_fallback: {random_fallback}")

    print("\nBy split:")
    for eval_name in sorted({row["eval_name"] for row in results}):
        rows = [row for row in results if row["eval_name"] == eval_name]
        correct = sum(1 for row in rows if row["correct"])
        length = sum(1 for row in rows if row["finish"] == "length")
        print(
            f"  {eval_name:32s} acc={correct / len(rows):.4f} "
            f"({correct}/{len(rows)}) length={length}"
        )

    metrics = aggregate_metrics(results)
    if metrics:
        print("\nBenchmark metrics:")
        for key in sorted(metrics):
            print(f"  {key}: {metrics[key]:.6f}")

    print("\nBy split / subject:")
    for split_name in sorted({row["split"] for row in results}):
        split_rows = [row for row in results if row["split"] == split_name]
        for subject in sorted({row["subject"] for row in split_rows}):
            rows = [row for row in split_rows if row["subject"] == subject]
            correct = sum(1 for row in rows if row["correct"])
            print(
                f"  {split_name:10s} {subject:32s} "
                f"acc={correct / len(rows):.4f} ({correct}/{len(rows)})"
            )


async def amain(args: argparse.Namespace) -> None:
    split_names = SPLIT_GROUPS[args.split]
    datasets = [(split, _load_dataset(split)) for split in split_names]
    available = sum(len(dataset) for _, dataset in datasets)
    total = min(available, args.limit) if args.limit is not None else available

    print(
        f"Running MMMU-Pro {args.split}: n={total} max_tokens={args.max_tokens} "
        f"temperature={args.temperature}",
        flush=True,
    )

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    out_path = Path(
        args.output
        or f"mmmu_pro_{args.split}_results_{_safe_model_name(args.model)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    tic = time.perf_counter()
    done = 0
    correct = 0
    errors = 0
    results: list[dict[str, Any]] = []
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    async def drain_one(future: asyncio.Task[dict[str, Any]]) -> None:
        nonlocal done, correct, errors
        row = await future
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
                f"last(id={row['id']} pred={row['predicted']} target={row['answer']} "
                f"correct={int(row['correct'])} finish={row['finish']}) {elapsed:.0f}s{tag}",
                flush=True,
            )

    for split, row, idx in _iter_dataset_rows(datasets, args.limit):
        sample = make_sample(row, split, idx)
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

    summarize(results, time.perf_counter() - tic, args.model, args.split)
    print(f"\nPer-sample results: {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--split", choices=tuple(SPLIT_GROUPS), default=DEFAULT_SPLIT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        type=_parse_reasoning_effort,
        default=None,
        help=f"optional scalar in [0, 1] or one of: {', '.join(_REASONING_EFFORT_LEVELS)}",
    )
    parser.add_argument("--concurrency", type=int, default=8)
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
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
