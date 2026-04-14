from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import time

import modal

APP_NAME = "sglang-tinker-ppo-smoke-test"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_ENDPOINT_URL = os.getenv("SGLANG_ENDPOINT_URL", "")
DEFAULT_TINKER_MODAL_SRC = pathlib.Path(
    os.getenv("TINKER_MODAL_SRC", "/Users/jm/tinker-modal/src")
)
REMOTE_TINKER_MODAL_SRC = pathlib.Path("/root/tinker_modal_src")
TINKER_SECRET = modal.Secret.from_name(
    "tinker-modal-labs-init",
    environment_name=os.getenv("MODAL_ENVIRONMENT", "jason-dev"),
)

image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "httpx>=0.28.0",
    "jinja2>=3.1.0",
    "orjson>=3.10.0",
    "tinker>=0.9.0",
)
if modal.is_local() and DEFAULT_TINKER_MODAL_SRC.exists():
    image = image.add_local_dir(
        DEFAULT_TINKER_MODAL_SRC / "tinker_modal",
        str(REMOTE_TINKER_MODAL_SRC / "tinker_modal"),
        copy=False,
    )

app = modal.App(APP_NAME)

with image.imports():
    import httpx


def _require_tinker_api_key() -> str:
    api_key = os.getenv("TINKER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is missing from the Modal secret.")
    return api_key


def _load_tinker_modal_ppo():
    if str(REMOTE_TINKER_MODAL_SRC) not in sys.path:
        sys.path.insert(0, str(REMOTE_TINKER_MODAL_SRC))
    try:
        from tinker_modal.ppo_example import PpoConfig, run_ppo
    except Exception as exc:
        raise RuntimeError(
            "No tinker_path was provided and the sibling tinker-modal PPO helper "
            "could not be imported. Set TINKER_MODAL_SRC to a checkout with "
            "src/tinker_modal/ppo_example.py or provide --tinker-path."
        ) from exc
    return PpoConfig, run_ppo


async def _run_smoke(
    *,
    endpoint_url: str,
    model: str,
    lora_rank: int,
    batch_size: int,
    max_tokens: int,
    sample_temperature: float,
    tinker_path: str | None,
    flush_cache: bool,
    atomic_pause_mode: str | None,
    post_merge_sleep_seconds: float,
    chat_attempts: int,
    chat_retry_delay_seconds: float,
) -> dict[str, object]:
    checkpoint_prefix = f"ppo_smoke_{int(time.time())}"
    training_model_id: str | None
    if tinker_path:
        sampler_path = tinker_path
        training_model_id = None
        print(f"[smoke] reusing sampler checkpoint {sampler_path}")
    else:
        PpoConfig, run_ppo = _load_tinker_modal_ppo()
        config = PpoConfig(
            model=model,
            lora_rank=lora_rank,
            batch_size=batch_size,
            num_iterations=1,
            max_tokens=max_tokens,
            sample_temperature=sample_temperature,
            checkpoint_prefix=checkpoint_prefix,
        )
        print(f"[smoke] starting one-step PPO run with checkpoint prefix {checkpoint_prefix}")
        ppo_result = await run_ppo(config)
        sampler_path = ppo_result.final_sampling_path
        training_model_id = ppo_result.training_model_id
        print(f"[smoke] produced sampler checkpoint {sampler_path}")

    api_key = _require_tinker_api_key()
    base_url = endpoint_url.rstrip("/")
    timeout = httpx.Timeout(1800.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        merge_response = await client.post(
            f"{base_url}/admin/update_merged_lora_from_tinker",
            json={
                "tinker_path": sampler_path,
                "tinker_api_key": api_key,
                "strict": True,
                "flush_cache": flush_cache,
                "atomic_pause_mode": atomic_pause_mode,
                "weight_version": sampler_path,
            },
        )
        merge_response.raise_for_status()
        merge_json = merge_response.json()
        if not merge_json.get("success"):
            raise RuntimeError(f"merge route returned failure: {merge_json}")
        print(f"[smoke] merge route accepted {sampler_path}")

        model_info_response = await client.get(f"{base_url}/get_model_info")
        model_info_response.raise_for_status()
        model_info = model_info_response.json()
        if model_info.get("weight_version") != sampler_path:
            raise RuntimeError(
                "weight version mismatch after merge: "
                f"expected={sampler_path!r} actual={model_info.get('weight_version')!r}"
            )
        print(f"[smoke] model info shows weight_version={model_info['weight_version']}")
        if post_merge_sleep_seconds > 0:
            print(f"[smoke] waiting {post_merge_sleep_seconds:.1f}s before probing chat")
            await asyncio.sleep(post_merge_sleep_seconds)

        chat_json: dict[str, object] | None = None
        completion_text: str | None = None
        chat_weight_version: str | None = None
        chat_metadata: dict[str, object] | None = None
        chat_error: dict[str, object] | None = None
        for attempt in range(1, chat_attempts + 1):
            chat_response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                    "temperature": 0,
                    "max_tokens": 8,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            if chat_response.is_success:
                chat_json = chat_response.json()
                chat_metadata = chat_json.get("metadata") or chat_json.get("meta_info") or {}
                chat_weight_version = chat_metadata.get("weight_version")
                completion_text = chat_json["choices"][0]["message"]["content"]
                print(
                    f"[smoke] chat completion succeeded on attempt {attempt} "
                    f"with content={completion_text!r}"
                )
                break

            chat_error = {
                "attempt": attempt,
                "status_code": chat_response.status_code,
                "body": chat_response.text,
            }
            print(
                f"[smoke] chat attempt {attempt} failed with status "
                f"{chat_response.status_code}: {chat_response.text[:400]!r}"
            )
            if attempt < chat_attempts:
                await asyncio.sleep(chat_retry_delay_seconds)

        if chat_json is not None and chat_weight_version != sampler_path:
            raise RuntimeError(
                "chat response weight version mismatch: "
                f"expected={sampler_path!r} actual={chat_weight_version!r}"
            )

    return {
        "success": chat_json is not None,
        "training_model_id": training_model_id,
        "checkpoint_prefix": checkpoint_prefix,
        "sampler_path": sampler_path,
        "merge_response": merge_json,
        "model_info": model_info,
        "chat_completion": completion_text,
        "chat_weight_version": chat_weight_version,
        "chat_metadata": chat_metadata,
        "chat_error": chat_error,
        "tinker_modal_src": (
            str(DEFAULT_TINKER_MODAL_SRC) if DEFAULT_TINKER_MODAL_SRC.exists() else None
        ),
    }


@app.function(
    image=image,
    secrets=[TINKER_SECRET],
    timeout=90 * 60,
)
def ppo_one_step_merge_smoke(
    endpoint_url: str,
    model: str = DEFAULT_MODEL,
    lora_rank: int = 8,
    batch_size: int = 1,
    max_tokens: int = 32,
    sample_temperature: float = 0.7,
    tinker_path: str = "",
    flush_cache: bool = False,
    atomic_pause_mode: str = "in_place",
    post_merge_sleep_seconds: float = 0.0,
    chat_attempts: int = 3,
    chat_retry_delay_seconds: float = 5.0,
) -> dict[str, object]:
    return asyncio.run(
        _run_smoke(
            endpoint_url=endpoint_url,
            model=model,
            lora_rank=lora_rank,
            batch_size=batch_size,
            max_tokens=max_tokens,
            sample_temperature=sample_temperature,
            tinker_path=tinker_path or None,
            flush_cache=flush_cache,
            atomic_pause_mode=atomic_pause_mode,
            post_merge_sleep_seconds=post_merge_sleep_seconds,
            chat_attempts=chat_attempts,
            chat_retry_delay_seconds=chat_retry_delay_seconds,
        )
    )


@app.local_entrypoint()
def main(
    endpoint_url: str = DEFAULT_ENDPOINT_URL,
    model: str = DEFAULT_MODEL,
    lora_rank: int = 8,
    batch_size: int = 1,
    max_tokens: int = 32,
    sample_temperature: float = 0.7,
    tinker_path: str = "",
    flush_cache: bool = False,
    atomic_pause_mode: str = "in_place",
    post_merge_sleep_seconds: float = 0.0,
    chat_attempts: int = 3,
    chat_retry_delay_seconds: float = 5.0,
) -> None:
    if not endpoint_url:
        raise RuntimeError(
            "Pass --endpoint-url or set SGLANG_ENDPOINT_URL to the deployed SGLang server URL."
        )
    result = ppo_one_step_merge_smoke.remote(
        endpoint_url=endpoint_url,
        model=model,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_tokens=max_tokens,
        sample_temperature=sample_temperature,
        tinker_path=tinker_path,
        flush_cache=flush_cache,
        atomic_pause_mode=atomic_pause_mode,
        post_merge_sleep_seconds=post_merge_sleep_seconds,
        chat_attempts=chat_attempts,
        chat_retry_delay_seconds=chat_retry_delay_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
