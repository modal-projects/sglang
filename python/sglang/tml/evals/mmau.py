"""Standalone MMAU audio-understanding eval against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from openai import AsyncOpenAI, OpenAIError

DEFAULT_SAMPLE_RATE = 16000
TASKS = ("sound", "music", "speech")
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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def mmau_string_match(answer: str, prediction: str, choices: list[str]) -> bool:
    """Official MMAU token matcher."""
    prediction_tokens = _tokenize(prediction)
    answer_tokens = _tokenize(answer)

    if not prediction_tokens:
        return False

    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = _tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(
        incorrect_tokens
    )


def strip_letter_prefix(s: str) -> str:
    """Strip a leading ``"(A) "`` / ``"(B) "`` / ... prefix if present."""
    if len(s) >= 4 and s[0] == "(" and s[2] == ")" and s[3] == " " and s[1].isalpha():
        return s[4:]
    return s


def to_mono_16k(audio_array: np.ndarray, audio_sr: int) -> np.ndarray:
    """Down-mix to mono FIRST, then resample to 16 kHz."""
    audio_array = np.asarray(audio_array, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if audio_sr != DEFAULT_SAMPLE_RATE:
        audio_array = librosa.resample(
            audio_array, orig_sr=audio_sr, target_sr=DEFAULT_SAMPLE_RATE
        )
    return audio_array.astype(np.float32)


def audio_to_data_url(raw_bytes: bytes) -> str:
    audio_array, audio_sr = sf.read(io.BytesIO(raw_bytes))
    audio_array = to_mono_16k(audio_array, audio_sr)
    buf = io.BytesIO()
    sf.write(buf, audio_array, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def build_prompt(question: str, choices: list[str]) -> str:
    """Build the MMAU multiple-choice instruction."""
    choices_text = "\n".join(choices)
    out = f"{question}\n\nChoice: \n{choices_text}\n"
    out += (
        f"Choose a choices from the given {len(choices)} choices. Do not provide any "
        "additional explanations or content. Output must match exactly one of the listed choices."
    )
    return out


def load_samples(limit: int | None, task: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id="gamma-lab-umd/MMAU-test-mini",
        filename="test_mini.parquet",
        repo_type="dataset",
    )
    rows = pq.ParquetFile(local_path).read().to_pylist()

    samples: list[dict] = []
    for row in rows:
        row_task = json.loads(row["other_attributes"])["task"]
        if task != "all" and row_task != task:
            continue
        choices = [strip_letter_prefix(c) for c in row["choices"]]
        samples.append(
            {
                "id": row.get("id"),
                "task": row_task,
                "question": row["instruction"],
                "choices": choices,
                "answer": strip_letter_prefix(row["answer"]),
                "audio_bytes": row["context"]["bytes"],
            }
        )
    if limit is not None:
        # The parquet is grouped by task; for an all-task smoke, round-robin across
        # tasks so a small --limit still exercises sound/music/speech.
        if task == "all":
            by_task: dict[str, list[dict]] = {}
            for s in samples:
                by_task.setdefault(s["task"], []).append(s)
            queues = list(by_task.values())
            interleaved: list[dict] = []
            i = 0
            while len(interleaved) < limit and any(i < len(q) for q in queues):
                for q in queues:
                    if i < len(q):
                        interleaved.append(q[i])
                        if len(interleaved) >= limit:
                            break
                i += 1
            samples = interleaved
        else:
            samples = samples[:limit]
    return samples


async def grade_one(
    sample: dict,
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> dict:
    prompt = build_prompt(sample["question"], sample["choices"])
    out = {
        "id": sample["id"],
        "task": sample["task"],
        "question": sample["question"],
        "choices": sample["choices"],
        "answer": sample["answer"],
        "prediction": "",
        "reasoning_len": 0,
        "correct": False,
        "error": None,
    }

    try:
        data_url = await asyncio.to_thread(audio_to_data_url, sample["audio_bytes"])
    except Exception as e:  # noqa: BLE001 - third-party audio decoding failure
        out["error"] = f"{type(e).__name__}: {str(e)[:240]}"
        return out

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio_url", "audio_url": {"url": data_url}},
            ],
        }
    ]
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
    }
    if args.reasoning_effort is not None:
        kwargs["reasoning_effort"] = args.reasoning_effort

    try:
        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        prediction = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""
        out["prediction"] = prediction
        out["reasoning_len"] = len(reasoning)
        out["correct"] = mmau_string_match(
            sample["answer"], prediction, sample["choices"]
        )
        return out
    except (OpenAIError, asyncio.TimeoutError) as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:240]}"

    return out


async def amain(args: argparse.Namespace) -> int:
    print(f"Loading MMAU v05.15.25 ({args.task}, limit={args.limit}) ...", flush=True)
    samples = load_samples(args.limit, args.task)
    print(f"  {len(samples)} samples", flush=True)
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1
    print(
        f"temperature={args.temperature} max_tokens={args.max_tokens} "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    out_path = Path(
        args.output or f"mmau_results_{args.model.replace('/', '_')}_{args.task}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.time()
    pending: set[asyncio.Task[dict]] = set()
    with out_path.open("w", encoding="utf-8") as out_f:

        async def drain_one(task: asyncio.Task[dict]) -> None:
            row = await task
            results.append(row)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            done = len(results)
            if done % LOG_EVERY == 0 or done == len(samples) or row["error"]:
                acc = sum(result["correct"] for result in results) / done
                print(
                    f"  {done}/{len(samples)}  running_acc={acc:.4f} "
                    f"errors={sum(1 for result in results if result['error'])}",
                    flush=True,
                )

        for sample in samples:
            pending.add(asyncio.create_task(grade_one(sample, client, args)))
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

    per_task: dict[str, list[int]] = {}
    n_err = 0
    for result in results:
        per_task.setdefault(result["task"], []).append(int(result["correct"]))
        n_err += int(bool(result["error"]))

    print("\n=== MMAU v05.15.25 results ===")
    print(
        f"model={args.model}  n={len(results)}  errors={n_err}  elapsed={time.time() - t0:.1f}s"
    )
    for task in sorted(per_task):
        values = per_task[task]
        print(
            f"  {task:8s}  acc={sum(values) / len(values):.4f}  ({sum(values)}/{len(values)})"
        )
    correct = sum(result["correct"] for result in results)
    overall = correct / len(results)
    print(f"  {'OVERALL':8s}  acc={overall:.4f}  ({correct}/{len(results)})")
    print(f"\nPer-sample results: {out_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--task", default="all", choices=("all", *TASKS))
    ap.add_argument(
        "--limit", type=int, default=None, help="total sample cap for smoke runs"
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument(
        "--reasoning-effort",
        type=_parse_reasoning_effort,
        default=None,
        help=f"optional scalar in [0, 1] or one of: {', '.join(_REASONING_EFFORT_LEVELS)}",
    )
    ap.add_argument("--output", default=None, help="JSONL path for per-sample results")
    args = ap.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
