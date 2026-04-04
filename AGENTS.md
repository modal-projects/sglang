# AGENTS.md

## Purpose

This folder is for repairing and extending live online-RL weight sync for Qwen3.5 / Qwen3.5-VL with a strong bias toward shared infrastructure, correctness, and Modal-based validation.

## Lessons Learned From The Previous Session

### 1. Modal Is A Hard Requirement

- Do not run Python locally for development or testing in this workspace.
- All execution and validation must go through Modal.
- Local shell inspection and code editing are fine.
- If a workflow needs Python execution, move it into a Modal function, a Modal smoke harness, or the established `~/flash-fde` patch-test flow.

### 2. Do Not Solve This With A Bespoke Qwen-Only Side Path

- The failed approach was to write a dedicated Qwen3.5 LoRA payload translator that duplicated model naming and packing knowledge.
- That is brittle and does not advance the real goal.
- The correct place to improve is the shared live weight-delta sync path:
  1. transport bytes safely
  2. compile logical deltas on the server
  3. localize deltas to runtime layout on the server
  4. apply in place

### 3. Reuse Existing Canonicalization Before Inventing New Logic

- Before writing new mapping code, inspect and reuse:
  - `python/sglang/srt/weight_sync/lora_merge_loader.py`
  - `python/sglang/srt/lora/lora.py`
  - `python/sglang/srt/lora/utils.py`
  - model `load_weights()` mappings in `python/sglang/srt/models/*`
- If a model already defines packed-module or expert mapping for checkpoint load, prefer factoring that into reusable shared metadata/helpers rather than re-encoding it elsewhere.

### 4. Validation Must Have A Strong Oracle

- "Logprobs changed" is not enough.
- First validate exact parameter effects or exact target coverage.
- Then validate end-to-end inference/logprob differences.
- For live LoRA sync, the preferred oracle order is:
  1. exact target/parameter agreement with the existing LoRA path or known-good mapping
  2. replay/versioning semantics
  3. prompt/logprob/output deltas

### 5. Separate Text-Only And Multimodal Validation Honestly

- Do not claim multimodal validation against a text-only base model.
- Follow the intended order:
  1. Qwen3.5-VL class in text-only mode
  2. Qwen3.5-specific linear-attn targets
  3. multimodal request path
- Each phase should explicitly use the right model class and the right oracle.

### 6. Respect Existing Modal/HF Infrastructure

- Assume the `huggingface-cache` Modal volume is the source of truth for cached models unless proven otherwise.
- Prefer the existing `~/flash-fde` patch export + Modal smoke workflow when it shortens iteration time or improves observability.
- Read actual Modal logs before concluding a run failed for the reason it first appeared to fail.

### 7. Temporary Workarounds Must Stay Narrow And Explicit

- If a Modal image bug requires a workaround, isolate it, document it, and do not confuse it with correctness work.
- Do not let environment workarounds become the main deliverable.

## Working Rules For The Repair

- Favor shared live-delta sync abstractions over model-specific payload builders.
- Keep Qwen3.5 text and Qwen3.5-VL support aligned through shared resolver logic wherever possible.
- Preserve the longer-term path to compressed-delta full-weight sync.
- Add tests at the lowest layer that actually covers the bug or abstraction being introduced.
- Use Modal for every execution step that would otherwise require Python.
