import contextlib
import gc
import importlib.metadata
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_LOADER = "sglang.srt.model_loader.lora_merge_loader.merge_lora_tensors_inplace"

BASE_MODEL = os.getenv("QWEN35_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
ADAPTER_DIR = Path(os.getenv("QWEN35_LORA_DIR", str(REPO_ROOT)))
ADAPTER_CONFIG_PATH = Path(
    os.getenv("QWEN35_LORA_CONFIG", str(ADAPTER_DIR / "adapter_config.json"))
)
ADAPTER_WEIGHTS_PATH = Path(
    os.getenv(
        "QWEN35_LORA_WEIGHTS",
        str(ADAPTER_DIR / "sampler_weights_init.safetensors"),
    )
)

MAX_NEW_TOKENS = int(os.getenv("QWEN35_MERGE_MAX_NEW_TOKENS", "48"))
MERGE_MAX_ABS_THRESHOLD = float(
    os.getenv("QWEN35_MERGE_MAX_ABS_THRESHOLD", "5e-2")
)
MERGE_MEAN_ABS_THRESHOLD = float(
    os.getenv("QWEN35_MERGE_MEAN_ABS_THRESHOLD", "5e-3")
)
TORCH_DTYPE = getattr(torch, os.getenv("QWEN35_MERGE_DTYPE", "bfloat16"))
ADAPTER_SUBSET = os.getenv("QWEN35_ADAPTER_SUBSET", "all")
TRACE_PROMPT_INDEX = int(os.getenv("QWEN35_TRACE_PROMPT_INDEX", "2"))
TRACE_TOP_K = int(os.getenv("QWEN35_TRACE_TOP_K", "5"))
SGLANG_MOE_RUNNER_BACKEND = os.getenv("QWEN35_SGLANG_MOE_RUNNER_BACKEND", "").strip()
HF_ATTN_IMPLEMENTATION = os.getenv("QWEN35_HF_ATTN_IMPLEMENTATION", "").strip()
SGLANG_ENABLE_DETERMINISTIC_INFERENCE = os.getenv(
    "QWEN35_SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1"
).strip().lower() in ("1", "true", "yes", "on")

PROMPTS = [
    "Summarize why mixture-of-experts models can improve serving efficiency, but mention at least one systems downside.",
    "Write a compact release note explaining an online RL rollout service that updates model weights without restarting inference workers.",
    "Given a customer support chat, produce a calm two-sentence reply that acknowledges the issue and asks for the minimum extra information needed.",
]


def _load_adapter_assets():
    with open(ADAPTER_CONFIG_PATH, "r") as f:
        adapter_config = json.load(f)
    adapter_tensors = _filter_adapter_tensors(
        list(load_file(str(ADAPTER_WEIGHTS_PATH)).items()),
        ADAPTER_SUBSET,
    )
    return adapter_config, adapter_tensors


def _normalize_adapter_subset_tokens(subset_spec: str) -> List[str]:
    tokens = [token.strip().lower() for token in subset_spec.split(",") if token.strip()]
    if not tokens:
        return ["all"]

    aliases = {
        "attn": "attention",
        "lmhead": "lm_head",
        "shared": "shared_expert",
        "routed": "routed_experts",
        "moe": "experts",
        "routed_gate": "routed_w1",
        "routed_up": "routed_w3",
        "routed_down": "routed_w2",
    }
    normalized = [aliases.get(token, token) for token in tokens]
    if "all" in normalized:
        return ["all"]

    valid = {
        "lm_head",
        "linear_attn",
        "self_attn",
        "attention",
        "shared_expert",
        "routed_experts",
        "routed_w1",
        "routed_w2",
        "routed_w3",
        "experts",
    }
    invalid = [token for token in normalized if token not in valid]
    if invalid:
        raise ValueError(
            f"Unsupported QWEN35_ADAPTER_SUBSET token(s): {', '.join(sorted(invalid))}"
        )
    return normalized


def _matches_adapter_subset_group(name: str, group: str) -> bool:
    if group == "all":
        return True
    if group == "lm_head":
        return "unembed_tokens" in name
    if group == "linear_attn":
        return ".linear_attn." in name
    if group == "self_attn":
        return ".self_attn." in name
    if group == "attention":
        return ".linear_attn." in name or ".self_attn." in name
    if group == "shared_expert":
        return ".shared_expert" in name
    if group == "routed_experts":
        return ".mlp.experts." in name
    if group == "routed_w1":
        return ".mlp.experts.w1." in name or ".mlp.experts.gate_proj." in name
    if group == "routed_w2":
        return ".mlp.experts.w2." in name or ".mlp.experts.down_proj." in name
    if group == "routed_w3":
        return ".mlp.experts.w3." in name or ".mlp.experts.up_proj." in name
    if group == "experts":
        return ".shared_expert" in name or ".mlp.experts." in name
    raise ValueError(f"Unknown adapter subset group: {group}")


def _filter_adapter_tensors(
    adapter_tensors: Iterable[tuple[str, torch.Tensor]],
    subset_spec: str,
) -> List[tuple[str, torch.Tensor]]:
    groups = _normalize_adapter_subset_tokens(subset_spec)
    if groups == ["all"]:
        return list(adapter_tensors)

    filtered = [
        (name, tensor)
        for name, tensor in adapter_tensors
        if any(_matches_adapter_subset_group(name, group) for group in groups)
    ]
    if not filtered:
        raise ValueError(f"No adapter tensors matched QWEN35_ADAPTER_SUBSET={subset_spec!r}")
    return filtered


def _write_adapter_dir(
    adapter_config: Dict[str, Any],
    adapter_tensors: List[tuple[str, torch.Tensor]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "adapter_config.json"
    weights_path = output_dir / "adapter_model.safetensors"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)
    save_file(dict(adapter_tensors), str(weights_path))
    return output_dir


@contextmanager
def _adapter_dir_for_subset(
    adapter_config: Dict[str, Any],
    adapter_tensors: List[tuple[str, torch.Tensor]],
):
    if _normalize_adapter_subset_tokens(ADAPTER_SUBSET) == ["all"]:
        yield ADAPTER_DIR
        return

    with tempfile.TemporaryDirectory(prefix="qwen35-lora-subset-") as tmpdir:
        yield _write_adapter_dir(adapter_config, adapter_tensors, Path(tmpdir))


def _cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _serialize_topk_from_logprobs(
    logprobs: torch.Tensor,
    tokenizer,
    top_k: int,
) -> List[List[Dict[str, Any]]]:
    values, indices = torch.topk(logprobs, top_k, dim=-1)
    serialized: List[List[Dict[str, Any]]] = []
    for row_vals, row_indices in zip(values, indices):
        serialized.append(
            [
                {
                    "logprob": float(value.item()),
                    "token_id": int(token_id.item()),
                    "token_text": _decode_token(tokenizer, int(token_id.item())),
                }
                for value, token_id in zip(row_vals, row_indices)
            ]
        )
    return serialized


def _compute_token_logprobs_and_topk(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    tokenizer,
    *,
    top_k: int = 0,
) -> tuple[torch.Tensor, List[List[Dict[str, Any]]] | None]:
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    token_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    topk = None
    if top_k > 0:
        topk = _serialize_topk_from_logprobs(logprobs, tokenizer, top_k)
    return token_logprobs, topk


def _serialize_generation_scores_topk(
    scores: Iterable[torch.Tensor],
    tokenizer,
    top_k: int,
) -> List[List[Dict[str, Any]]]:
    serialized: List[List[Dict[str, Any]]] = []
    for score in scores:
        logprobs = F.log_softmax(score[0], dim=-1, dtype=torch.float32)
        serialized.extend(_serialize_topk_from_logprobs(logprobs.unsqueeze(0), tokenizer, top_k))
    return serialized


def _hf_model_load_kwargs(torch_dtype: torch.dtype) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if HF_ATTN_IMPLEMENTATION:
        kwargs["attn_implementation"] = HF_ATTN_IMPLEMENTATION
    return kwargs


def _hf_backend_report(
    model,
    *,
    merge_mode: str,
    output_text: str = "",
) -> Dict[str, Any]:
    def _module_report(module_name: str, package_name: str) -> Dict[str, Any]:
        available = importlib.util.find_spec(module_name) is not None
        version = None
        if available:
            try:
                version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                version = None
        return {
            "module": module_name,
            "package": package_name,
            "available": available,
            "version": version,
        }

    config = getattr(model, "config", None)
    return {
        "merge_mode": merge_mode,
        "requested_attn_implementation": HF_ATTN_IMPLEMENTATION or None,
        "resolved_attn_implementation": getattr(
            config, "_attn_implementation", None
        ),
        "model_class": type(model).__name__,
        "config_class": type(config).__name__ if config is not None else None,
        "fast_path_fallback_detected": "The fast path is not available" in output_text,
        "optional_packages": {
            "fla": _module_report("fla", "flash-linear-attention"),
            "causal_conv1d": _module_report("causal_conv1d", "causal-conv1d"),
        },
    }


def _hf_generate_and_score(
    model_path: str,
    adapter_dir: Path,
    prompts: List[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from peft import PeftModel

    stream = io.StringIO()
    model = None
    results: List[Dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **_hf_model_load_kwargs(torch_dtype),
            )
            model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
                torch_dtype=torch_dtype,
                is_trainable=False,
            )
            model.eval()
            for prompt in prompts:
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    model.device
                )
                prompt_attention_mask = torch.ones_like(prompt_ids)
                outputs = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

                full_ids = outputs.sequences[0]
                prompt_len = prompt_ids.shape[1]
                completion_ids = full_ids[prompt_len:]
                completion_text = tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )

                full_ids_batched = full_ids.unsqueeze(0)
                full_attention_mask = torch.ones_like(full_ids_batched)
                logits = model(
                    full_ids_batched,
                    attention_mask=full_attention_mask,
                ).logits[0, :-1]
                target_ids = full_ids[1:]
                token_logprobs, _ = _compute_token_logprobs_and_topk(
                    logits,
                    target_ids,
                    tokenizer,
                )
                prefill_logprobs = token_logprobs[: prompt_len - 1].cpu()
                completion_logprobs = token_logprobs[prompt_len - 1 :].cpu()

                results.append(
                    {
                        "prompt": prompt,
                        "prompt_len": prompt_len,
                        "full_ids": full_ids.cpu().tolist(),
                        "completion_ids": completion_ids.cpu().tolist(),
                        "completion_text": completion_text,
                        "hf_prefill_logprobs": prefill_logprobs,
                        "hf_completion_logprobs": completion_logprobs,
                    }
                )
        backend_report = _hf_backend_report(
            model,
            merge_mode="peft_runtime",
            output_text=stream.getvalue(),
        )
    finally:
        if model is not None:
            del model
        _cleanup_torch()

    return results, backend_report


def _hf_generate_and_score_with_materialized_comparison(
    model_path: str,
    adapter_dir: Path,
    prompts: List[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    from peft import PeftModel

    stream = io.StringIO()
    model = None
    runtime_results: List[Dict[str, Any]] = []
    materialized_results: List[Dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **_hf_model_load_kwargs(torch_dtype),
            )
            model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
                torch_dtype=torch_dtype,
                is_trainable=False,
            )
            model.eval()
            for prompt_index, prompt in enumerate(prompts):
                trace_this_prompt = (
                    TRACE_TOP_K > 0 and prompt_index == TRACE_PROMPT_INDEX
                )
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    model.device
                )
                prompt_attention_mask = torch.ones_like(prompt_ids)
                outputs = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=trace_this_prompt,
                )

                full_ids = outputs.sequences[0]
                prompt_len = prompt_ids.shape[1]
                completion_ids = full_ids[prompt_len:]
                completion_text = tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )

                full_ids_batched = full_ids.unsqueeze(0)
                full_attention_mask = torch.ones_like(full_ids_batched)
                logits = model(
                    full_ids_batched,
                    attention_mask=full_attention_mask,
                ).logits[0, :-1]
                target_ids = full_ids[1:]
                token_logprobs, topk = _compute_token_logprobs_and_topk(
                    logits,
                    target_ids,
                    tokenizer,
                    top_k=TRACE_TOP_K if trace_this_prompt else 0,
                )
                runtime_results.append(
                    {
                        "prompt": prompt,
                        "prompt_len": prompt_len,
                        "full_ids": full_ids.cpu().tolist(),
                        "completion_ids": completion_ids.cpu().tolist(),
                        "completion_text": completion_text,
                        "hf_prefill_logprobs": token_logprobs[: prompt_len - 1].cpu(),
                        "hf_completion_logprobs": token_logprobs[
                            prompt_len - 1 :
                        ].cpu(),
                        **(
                            {
                                "trace_prompt_data": {
                                    "prompt_index": prompt_index,
                                    "prompt": prompt,
                                    "prompt_len": prompt_len,
                                    "full_ids": full_ids.cpu().tolist(),
                                    "full_token_texts": [
                                        _decode_token(tokenizer, token_id)
                                        for token_id in full_ids.cpu().tolist()
                                    ],
                                    "completion_ids": completion_ids.cpu().tolist(),
                                    "completion_token_texts": [
                                        _decode_token(tokenizer, token_id)
                                        for token_id in completion_ids.cpu().tolist()
                                    ],
                                    "hf_runtime_prefill_top_logprobs": topk[
                                        : prompt_len - 1
                                    ],
                                    "hf_runtime_completion_top_logprobs": topk[
                                        prompt_len - 1 :
                                    ],
                                    "hf_runtime_free_run_top_logprobs": _serialize_generation_scores_topk(
                                        outputs.scores,
                                        tokenizer,
                                        TRACE_TOP_K,
                                    ),
                                }
                            }
                            if trace_this_prompt and topk is not None
                            else {}
                        ),
                    }
                )
            runtime_backend_report = _hf_backend_report(
                model,
                merge_mode="peft_runtime",
                output_text=stream.getvalue(),
            )

            model = model.merge_and_unload()
            model.eval()
            for item in runtime_results:
                trace_this_prompt = "trace_prompt_data" in item
                full_ids = torch.tensor(
                    item["full_ids"],
                    dtype=torch.long,
                    device=model.device,
                )
                full_ids_batched = full_ids.unsqueeze(0)
                full_attention_mask = torch.ones_like(full_ids_batched)
                logits = model(
                    full_ids_batched,
                    attention_mask=full_attention_mask,
                ).logits[0, :-1]
                target_ids = full_ids[1:]
                token_logprobs, topk = _compute_token_logprobs_and_topk(
                    logits,
                    target_ids,
                    tokenizer,
                    top_k=TRACE_TOP_K if trace_this_prompt else 0,
                )
                prompt_len = item["prompt_len"]
                materialized_results.append(
                    {
                        "prompt": item["prompt"],
                        "hf_materialized_prefill_logprobs": token_logprobs[
                            : prompt_len - 1
                        ].cpu(),
                        "hf_materialized_completion_logprobs": token_logprobs[
                            prompt_len - 1 :
                        ].cpu(),
                        **(
                            {
                                "trace_prompt_materialized_data": {
                                    "hf_materialized_prefill_top_logprobs": topk[
                                        : prompt_len - 1
                                    ],
                                    "hf_materialized_completion_top_logprobs": topk[
                                        prompt_len - 1 :
                                    ],
                                }
                            }
                            if trace_this_prompt and topk is not None
                            else {}
                        ),
                    }
                )
            materialized_backend_report = _hf_backend_report(
                model,
                merge_mode="merged_materialized",
                output_text=stream.getvalue(),
            )
    finally:
        if model is not None:
            del model
        _cleanup_torch()

    return (
        runtime_results,
        materialized_results,
        runtime_backend_report,
        materialized_backend_report,
    )


def _hf_score_sequences_base(
    model_path: str,
    prompt_results: List[Dict[str, Any]],
    torch_dtype: torch.dtype,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stream = io.StringIO()
    model = None
    results: List[Dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **_hf_model_load_kwargs(torch_dtype),
            )
            model.eval()
            for prompt_index, item in enumerate(prompt_results):
                trace_this_prompt = (
                    TRACE_TOP_K > 0 and prompt_index == TRACE_PROMPT_INDEX
                )
                full_ids = torch.tensor(
                    item["full_ids"],
                    dtype=torch.long,
                    device=model.device,
                )
                full_ids_batched = full_ids.unsqueeze(0)
                full_attention_mask = torch.ones_like(full_ids_batched)
                logits = model(
                    full_ids_batched,
                    attention_mask=full_attention_mask,
                ).logits[0, :-1]
                target_ids = full_ids[1:]
                token_logprobs, topk = _compute_token_logprobs_and_topk(
                    logits,
                    target_ids,
                    tokenizer,
                    top_k=TRACE_TOP_K if trace_this_prompt else 0,
                )
                prompt_len = item["prompt_len"]
                results.append(
                    {
                        "prompt": item["prompt"],
                        "hf_base_prefill_logprobs": token_logprobs[
                            : prompt_len - 1
                        ].cpu(),
                        "hf_base_completion_logprobs": token_logprobs[
                            prompt_len - 1 :
                        ].cpu(),
                        **(
                            {
                                "trace_prompt_base_data": {
                                    "hf_base_prefill_top_logprobs": topk[
                                        : prompt_len - 1
                                    ],
                                    "hf_base_completion_top_logprobs": topk[
                                        prompt_len - 1 :
                                    ],
                                }
                            }
                            if trace_this_prompt and topk is not None
                            else {}
                        ),
                    }
                )
        backend_report = _hf_backend_report(
            model,
            merge_mode="base",
            output_text=stream.getvalue(),
        )
    finally:
        if model is not None:
            del model
        _cleanup_torch()

    return results, backend_report


def _hf_score_sequences_materialized_merged(
    model_path: str,
    adapter_dir: Path,
    prompt_results: List[Dict[str, Any]],
    torch_dtype: torch.dtype,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from peft import PeftModel

    stream = io.StringIO()
    model = None
    results: List[Dict[str, Any]] = []
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **_hf_model_load_kwargs(torch_dtype),
            )
            model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
                torch_dtype=torch_dtype,
                is_trainable=False,
            )
            model = model.merge_and_unload()
            model.eval()
            for prompt_index, item in enumerate(prompt_results):
                trace_this_prompt = (
                    TRACE_TOP_K > 0 and prompt_index == TRACE_PROMPT_INDEX
                )
                full_ids = torch.tensor(
                    item["full_ids"],
                    dtype=torch.long,
                    device=model.device,
                )
                logits = model(full_ids.unsqueeze(0)).logits[0, :-1]
                target_ids = full_ids[1:]
                token_logprobs, topk = _compute_token_logprobs_and_topk(
                    logits,
                    target_ids,
                    tokenizer,
                    top_k=TRACE_TOP_K if trace_this_prompt else 0,
                )
                prompt_len = item["prompt_len"]
                results.append(
                    {
                        "prompt": item["prompt"],
                        "hf_materialized_prefill_logprobs": token_logprobs[
                            : prompt_len - 1
                        ].cpu(),
                        "hf_materialized_completion_logprobs": token_logprobs[
                            prompt_len - 1 :
                        ].cpu(),
                        **(
                            {
                                "trace_prompt_materialized_data": {
                                    "hf_materialized_prefill_top_logprobs": topk[
                                        : prompt_len - 1
                                    ],
                                    "hf_materialized_completion_top_logprobs": topk[
                                        prompt_len - 1 :
                                    ],
                                }
                            }
                            if trace_this_prompt and topk is not None
                            else {}
                        ),
                    }
                )
        backend_report = _hf_backend_report(
            model,
            merge_mode="merged_materialized",
            output_text=stream.getvalue(),
        )
    finally:
        if model is not None:
            del model
        _cleanup_torch()

    return results, backend_report


def _unwrap_single_response(response):
    if isinstance(response, list):
        assert len(response) == 1
        return response[0]
    return response


def _sglang_score_sequences_with_engine(
    engine,
    prompt_results: List[Dict[str, Any]],
    max_new_tokens: int,
    *,
    include_free_run: bool,
    completion_key: str,
    prefill_key: str,
) -> List[Dict[str, Any]]:
    scored_results: List[Dict[str, Any]] = []
    for item in prompt_results:
        score_out = _unwrap_single_response(
            engine.generate(
                input_ids=item["full_ids"],
                sampling_params={"max_new_tokens": 0, "temperature": 0.0},
                return_logprob=True,
                logprob_start_len=0,
            )
        )
        input_token_logprobs = score_out["meta_info"]["input_token_logprobs"]
        sglang_logprobs = torch.tensor(
            [logprob for logprob, _, _ in input_token_logprobs],
            dtype=torch.float32,
        )
        prompt_len = item["prompt_len"]
        result = {
            **item,
            prefill_key: sglang_logprobs[1:prompt_len],
            completion_key: sglang_logprobs[prompt_len:],
        }

        if include_free_run:
            gen_out = _unwrap_single_response(
                engine.generate(
                    prompt=item["prompt"],
                    sampling_params={
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                )
            )
            result["sglang_completion_text"] = gen_out["text"]

        scored_results.append(result)
    return scored_results


def _sglang_score_base(
    model_path: str,
    prompt_results: List[Dict[str, Any]],
    torch_dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    engine_kwargs = dict(
        model_path=model_path,
        dtype=str(torch_dtype).replace("torch.", ""),
        disable_radix_cache=True,
        enable_deterministic_inference=SGLANG_ENABLE_DETERMINISTIC_INFERENCE,
        log_level="error",
    )
    if SGLANG_MOE_RUNNER_BACKEND:
        engine_kwargs["moe_runner_backend"] = SGLANG_MOE_RUNNER_BACKEND
    engine = sgl.Engine(**engine_kwargs)

    try:
        return _sglang_score_sequences_with_engine(
            engine,
            prompt_results,
            max_new_tokens=0,
            include_free_run=False,
            completion_key="sglang_base_completion_logprobs",
            prefill_key="sglang_base_prefill_logprobs",
        )
    finally:
        engine.shutdown()
        _cleanup_torch()


def _sglang_score_merged(
    model_path: str,
    adapter_tensors: List,
    adapter_config: Dict[str, Any],
    prompt_results: List[Dict[str, Any]],
    torch_dtype: torch.dtype,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    engine_kwargs = dict(
        model_path=model_path,
        dtype=str(torch_dtype).replace("torch.", ""),
        custom_weight_loader=[MERGE_LOADER],
        disable_radix_cache=True,
        enable_deterministic_inference=SGLANG_ENABLE_DETERMINISTIC_INFERENCE,
        log_level="error",
    )
    if SGLANG_MOE_RUNNER_BACKEND:
        engine_kwargs["moe_runner_backend"] = SGLANG_MOE_RUNNER_BACKEND
    engine = sgl.Engine(**engine_kwargs)

    try:
        success, message = engine.update_weights_from_tensor(
            named_tensors=adapter_tensors,
            manifest={"adapter_config": adapter_config},
            load_format=MERGE_LOADER,
        )
        if not success:
            raise RuntimeError(f"Merged LoRA update failed: {message}")

        merged_results = _sglang_score_sequences_with_engine(
            engine,
            prompt_results,
            max_new_tokens=max_new_tokens,
            include_free_run=True,
            completion_key="sglang_completion_logprobs",
            prefill_key="sglang_prefill_logprobs",
        )
    finally:
        engine.shutdown()
        _cleanup_torch()

    return merged_results


class TestQwen35MergedLoRALogprobDiff(unittest.TestCase):
    def test_qwen35_merged_lora_matches_hf_peft_logprobs(self):
        self.assertTrue(
            ADAPTER_CONFIG_PATH.exists(),
            f"Missing adapter config: {ADAPTER_CONFIG_PATH}",
        )
        self.assertTrue(
            ADAPTER_WEIGHTS_PATH.exists(),
            f"Missing adapter weights: {ADAPTER_WEIGHTS_PATH}",
        )

        adapter_config, adapter_tensors = _load_adapter_assets()
        self.assertEqual(adapter_config["peft_type"].lower(), "lora")
        self.assertFalse(adapter_config.get("use_dora", False))
        self.assertEqual(adapter_config.get("bias"), "none")
        self.assertFalse(adapter_config.get("lora_bias", False))

        with _adapter_dir_for_subset(adapter_config, adapter_tensors) as adapter_dir:
            hf_results, _, _, _ = _hf_generate_and_score_with_materialized_comparison(
                model_path=BASE_MODEL,
                adapter_dir=adapter_dir,
                prompts=PROMPTS,
                max_new_tokens=MAX_NEW_TOKENS,
                torch_dtype=TORCH_DTYPE,
            )
        merged_results = _sglang_score_merged(
            model_path=BASE_MODEL,
            adapter_tensors=adapter_tensors,
            adapter_config=adapter_config,
            prompt_results=hf_results,
            torch_dtype=TORCH_DTYPE,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        max_abs_values = []
        mean_abs_values = []
        for item in merged_results:
            hf_logprobs = item["hf_completion_logprobs"]
            sglang_logprobs = item["sglang_completion_logprobs"]
            self.assertEqual(
                len(hf_logprobs),
                len(sglang_logprobs),
                f"Completion length mismatch for prompt: {item['prompt']}",
            )

            diff = (hf_logprobs - sglang_logprobs).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            max_abs_values.append(max_abs)
            mean_abs_values.append(mean_abs)

            print(f"\nPrompt: {item['prompt']}")
            print(f"HF completion:      {item['completion_text']}")
            print(f"SGLang completion:  {item['sglang_completion_text']}")
            print(f"Completion tokens:  {len(item['completion_ids'])}")
            print(f"Abs diff max:       {max_abs:.6e}")
            print(f"Abs diff mean:      {mean_abs:.6e}")

        overall_max_abs = max(max_abs_values)
        overall_mean_abs = sum(mean_abs_values) / len(mean_abs_values)
        print("\nOverall merged-vs-PEFT completion logprob diff:")
        print(f"  max_abs  = {overall_max_abs:.6e}")
        print(f"  mean_abs = {overall_mean_abs:.6e}")

        self.assertLessEqual(
            overall_max_abs,
            MERGE_MAX_ABS_THRESHOLD,
            f"overall_max_abs={overall_max_abs:.6e} exceeds threshold {MERGE_MAX_ABS_THRESHOLD:.6e}",
        )
        self.assertLessEqual(
            overall_mean_abs,
            MERGE_MEAN_ABS_THRESHOLD,
            f"overall_mean_abs={overall_mean_abs:.6e} exceeds threshold {MERGE_MEAN_ABS_THRESHOLD:.6e}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
