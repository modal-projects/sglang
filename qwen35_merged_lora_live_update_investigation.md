# Qwen3.5 Merged LoRA Live Update Investigation

## Summary

This note summarizes the investigation into the production-like merged LoRA live-update failure for `Qwen/Qwen3.5-35B-A3B` on the current `~/sglang` branch.

The short version:

- the merged LoRA update itself succeeds
- `weight_version` and `weight_epoch` advance successfully
- the first decode after the update crashes the scheduler / GPU path
- the failure reproduces under the backend/config mix we care about:
  - `prefill_attention_backend=trtllm_mha`
  - `decode_attention_backend=trtllm_mha`
  - `moe_runner_backend=flashinfer_trtllm`
  - radix cache enabled
  - CUDA graph enabled
  - merged update via `/update_weights_from_tensor`

This now has a targeted failing repro in the repo.

## New Repro Artifacts

- Manual regression:
  [test_qwen35_merged_lora_live_update.py](/Users/jm/sglang/test/manual/test_qwen35_merged_lora_live_update.py)
- Modal runner:
  [modal_validate_qwen35_merged_lora_live_update.py](/Users/jm/sglang/scripts/modal_validate_qwen35_merged_lora_live_update.py)

The Modal runner mounts local `python/sglang` and `test/manual` into the nightly image and uploads adapter assets into the HF cache volume.

## Repro Shape

The regression intentionally follows the live server path instead of the engine-only path.

Flow:

1. Launch HTTP server for `Qwen/Qwen3.5-35B-A3B`.
2. Use the production-like backend settings above.
3. Send a pre-merge OpenAI chat request and confirm it succeeds.
4. Send `POST /update_weights_from_tensor` with merged-LoRA loader:
   - `load_format=sglang.srt.model_loader.lora_merge_loader.merge_lora_tensors_inplace`
   - manifest carries `adapter_config`
   - `atomic_pause_mode=in_place`
   - `flush_cache=False`
5. Confirm `/get_model_info` reflects the new `weight_version` and incremented `weight_epoch`.
6. Send one post-merge OpenAI chat request.
7. Observe crash.

The test currently expects the post-merge chat to succeed, so it fails when the crash happens.

## Key Investigation Notes

### What was already known

There was already evidence that:

- the Tinker download / adapter transport path was fine
- constrained serving configs could survive the merged update
- the unresolved issue was likely in the production backend mix or in missing coverage around it

In particular, a "safe" deployment using reduced serving features could merge and decode successfully, but that was not acceptable as a real fix because disabling radix cache or CUDA graph would materially hurt serving performance.

### Earlier harness dead ends

Two intermediate failures were harness-specific and not the real target bug:

1. Launching the server as a plain subprocess and hitting `/update_weights_from_tensor` caused:
   - `multiprocessing.context.AuthenticationError: digest sent was rejected`
   - this came from `MultiprocessingSerializer.deserialize()` rebuilding shared tensor storage
   - this is a serializer / process-launch mismatch, not the production bug

2. Switching to `HttpServerEngineAdapter` without changing multiprocessing startup caused:
   - CUDA re-init errors in forked subprocesses
   - `Cannot re-initialize CUDA in forked subprocess`
   - this was addressed by forcing multiprocessing start method to `spawn` in the test process

After those were removed, the reproducer reached the real failure mode.

## Latest Concrete Reproduction

The latest successful repro run used:

- image: `lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0`
- GPU: `B200`
- model: `Qwen/Qwen3.5-35B-A3B`
- update path: `/update_weights_from_tensor`
- loader: `merge_lora_tensors_inplace`
- cache behavior:
  - radix cache enabled
  - CUDA graph enabled
  - `flush_cache=False`
  - `atomic_pause_mode=in_place`

Observed sequence:

1. Server started successfully.
2. Pre-merge chat succeeded.
3. `POST /update_weights_from_tensor` returned `200 OK`.
4. `GET /get_model_info` after merge returned `200 OK`.
5. First post-merge chat caused the scheduler/GPU failure.

Key stdout evidence from the run:

- `POST /update_weights_from_tensor HTTP/1.1" 200 OK`
- `GET /get_model_info HTTP/1.1" 200 OK`

Then the next decode failed.

## Failure Signature

The latest run reproduced the same class of failure we were chasing:

- GPU health warning with XID 31 MMU fault
- scheduler process abort
- client-side `RemoteDisconnected`

Representative signal:

```text
[gpu-health] [WARN] ... Xid ... 31 ... MMU Fault ...
```

And the test-side symptom:

```text
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

The captured stderr also shows the scheduler process aborting after the update had already succeeded, during the first post-merge decode path.

## Interpretation

The current evidence points away from transport and toward runtime interaction after the merged update is installed.

Specifically:

- this is not a Tinker download issue
- this is not just an HTTP routing issue
- this is not a failure to ingest or apply the merged LoRA tensors
- this is not a failure to advance `weight_version` / `weight_epoch`

The crash happens after the update is accepted, when the next decode runs under the production-like backend mix.

That strongly suggests the interesting bug surface is one of:

- cache state vs updated weights
- CUDA graph reuse / invalidation after in-place merged update
- `trtllm_mha` decode path after merged update
- `flashinfer_trtllm` MoE path after merged update
- scheduler / worker state interaction after `update_weights_from_tensor`

## Confirmed Root Cause

The confirmed root cause is stale CUDA-graph reuse after the merged LoRA tensor update mutates FlashInfer TRT-LLM MoE weight storage in a graph-unsafe way.

More concretely:

- `merge_lora_tensors_inplace` uses the custom merged-loader path
- for unquantized FlashInfer TRT-LLM BF16 MoE weights, the loader restores canonical shapes, applies deltas, and then re-runs `process_weights_after_loading`
- that postprocess path reshapes and rebinds `.data` for expert weights instead of only copying into the existing storage
- the normal decode path still tries to replay the previously captured decode graph on the next request
- the captured graph therefore reuses stale weight addresses and the first post-update decode can fault with the observed MMU/XID failure

This is why:

- the update itself succeeds
- `weight_version` / `weight_epoch` advance correctly
- the crash happens only on the first decode after the update

The scheduler `in_place` pause behavior also needed tightening so the update runs against a quiescent scheduler-owned batch state, but that was not the primary crash trigger in this repro.

## Implemented Resolution

The branch now applies the following fix:

1. `pause_generation(mode="in_place")` now fully quiesces scheduler-owned overlap / batch pointers before the weight mutation runs, while still preserving the live-update semantics we want (`chunked_req`, running requests, and no forced retract/flush).
2. `UpdateWeightsFromTensorReqInput` now supports an explicit `recapture_cuda_graph` knob.
3. `ModelRunner.update_weights_from_tensor(...)` now supports graph rebuild after tensor updates.
4. custom loaders can declare `sglang_requires_cuda_graph_recapture = True`
5. `merge_lora_tensors_inplace` declares that requirement
6. when that flag is present, the model runner rebuilds decode/device graphs after the update before serving the next request

This fixes the production-like repro without requiring:

- disabling radix cache
- disabling CUDA graph globally
- changing the desired live-update API shape (`atomic_pause_mode="in_place"`, `flush_cache=False`)

## Validation Outcome

The same production-like B200 repro that previously failed now passes:

- pre-merge chat succeeds
- `POST /update_weights_from_tensor` returns `200 OK`
- `GET /get_model_info` reflects the new epoch/version
- first post-merge chat succeeds
- returned metadata reports:
  - `weight_epoch_start=1`
  - `weight_epoch_end=1`
  - `weight_version_start=weight_version_end=qwen35-merged-lora-live-update`
  - `mixed_weight_epochs=false`
  - `resume_from_stale_kv=false`

The Modal run now completes with:

```text
Ran 1 test in 269.874s

OK
```

Additional lightweight validation also passes in Modal:

- scheduler pause unit test
- model-runner weight-update graph-rebuild unit test
- engine API / serialization probes for the `in_place` pause contract

## Perf Implication

The current fix is the safe/correct one, but it is not the minimal-latency shape yet.

Important nuance:

- piecewise CUDA graph is used from `forward_extend`
- the normal decode graph is used from the decode replay path
- this bug manifested on the first post-update decode, so piecewise-only recapture would not have fixed the crash

The current implementation rebuilds both graph systems for safety after graph-unsafe tensor updates.

The likely next optimization is:

1. invalidate both graph runners after the merged update
2. eagerly rebuild only the normal decode graph, or allow the first decode to fall back to eager execution
3. rebuild piecewise graphs lazily/on-demand instead of blocking the update path on them

That is the right direction if the current synchronous graph rebuild shows up as meaningful update latency in the PPO loop.

## Why This Repro Matters

Before this change, existing merged-LoRA validation did not cover the exact path we cared about:

- the earlier Qwen3.5 merged-LoRA validation was engine-level
- it explicitly disabled radix cache
- it did not exercise the live HTTP server update path with the production-like backend mix

The new repro closes that gap.

## How To Run

From `~/tinker-modal` or anywhere with the Modal CLI available:

```bash
uv run modal run /Users/jm/sglang/scripts/modal_validate_qwen35_merged_lora_live_update.py
```

The runner expects these adapter assets to exist in the repo root by default:

- `/Users/jm/sglang/adapter_config.json`
- `/Users/jm/sglang/sampler_weights_init.safetensors`

It can also be driven by env vars such as:

- `QWEN35_LORA_CONFIG`
- `QWEN35_LORA_WEIGHTS`
- `QWEN35_PREFILL_ATTENTION_BACKEND`
- `QWEN35_DECODE_ATTENTION_BACKEND`
- `QWEN35_MOE_RUNNER_BACKEND`
- `QWEN35_LIVE_UPDATE_MEM_FRACTION_STATIC`
- `QWEN35_LIVE_UPDATE_ATOMIC_PAUSE_MODE`

## Remaining Follow-Up

At this point the correctness issue is fixed for the tracked repro.

The next follow-up is not more crash debugging; it is performance refinement:

- measure how much synchronous time the post-update graph rebuild adds in the real PPO loop
- if needed, split eager decode-graph rebuild from lazy piecewise-graph rebuild
- then validate repeated live updates across multiple checkpoints in the deployed Tinker ↔ SGLang loop
