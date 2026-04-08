# Qwen3.5 Merged LoRA Validation Summary

## Scope

This note summarizes the Modal-based validation work for merged-LoRA inference on:

- base model: `Qwen/Qwen3.5-35B-A3B`
- adapter config: `./adapter_config.json`
- adapter weights: `./sampler_weights_init.safetensors`

Primary code:

- manual reference test: [test_qwen35_merged_lora_logprob_diff.py](/Users/jm/sglang/test/manual/test_qwen35_merged_lora_logprob_diff.py)
- main Modal harness: [modal_validate_qwen35_merged_lora_logprob_diff_split.py](/Users/jm/sglang/scripts/modal_validate_qwen35_merged_lora_logprob_diff_split.py)
- FlashInfer TRTLLM tensor probe: [modal_probe_flashinfer_trtllm_moe_layout.py](/Users/jm/sglang/scripts/modal_probe_flashinfer_trtllm_moe_layout.py)
- loader/unit Modal harness: [modal_validate_lora_merge_loader.py](/Users/jm/sglang/scripts/modal_validate_lora_merge_loader.py)

## What The Harness Measures

For the same prompt and token IDs, the harness compares:

1. HF base
2. HF merged
3. SGLang base
4. SGLang merged

It reports:

- `base_* = |HF_base - SG_base|`
- `merged_* = |HF_merged - SG_merged|`
- `delta_* = |(HF_merged - HF_base) - (SG_merged - SG_base)|`

`delta_*` is the most useful signal. It isolates the adapter effect from ordinary HF-vs-SGLang base drift.

It reports those metrics for:

- prompt-side teacher-forced logprobs (`prefill`)
- completion-side teacher-forced logprobs

It also records free-run greedy output to catch token-level divergence.

## Validation History

### 1. The failure is real and starts before free-run decode

The first decisive teacher-forced reruns showed:

- first generated token diverged immediately on all prompts
- teacher-forced completion deltas were catastrophic
- prompt-side prefill logprobs were already badly wrong

Key runs:

- `ap-FV14YlAUPMu2KFYxKOhVis`
  - first decisive teacher-forced completion failure
- `ap-xjzvYWavKiCbfl9ZFePzhf`
  - proved divergence already appears during prefill

Conclusion:

- this is not just sampling drift
- this is not just a prompt/completion slicing bug
- the merged model is assigning the wrong probabilities on the HF continuation itself

### 2. Base-vs-merged-vs-delta split proved the adapter effect was the problem

Run `ap-Y2y9LE5wZLpFzdChbIaHep` added the `base_*` and `delta_*` comparisons.

At the catastrophic stage:

- `overall_base_mean_abs ~= 0.0192`
- `overall_base_prefill_mean_abs ~= 0.0475`
- `overall_delta_mean_abs ~= 5.9202`
- `overall_delta_prefill_mean_abs ~= 4.1715`

Conclusion:

- ordinary Qwen3.5 HF-vs-SGLang drift exists
- but it is far too small to explain the merged failure
- the adapter effect itself was badly wrong

### 3. Pre-fix family ablations localized the catastrophic gap to routed experts

On the default FlashInfer backend, before the routed-expert fix:

| Subset | `overall_delta_mean_abs` | `overall_delta_prefill_mean_abs` |
| --- | ---: | ---: |
| `lm_head` | `0.0170` | `0.0271` |
| `shared_expert` | `0.0219` | `0.0742` |
| `attention` | `0.0392` | `0.0876` |
| `routed_experts` | `5.8416` | `4.1023` |

Conclusion:

- `routed_experts` was overwhelmingly dominant
- non-routed families were real but not catastrophic

### 4. Triton ablations showed the catastrophic routed failure was backend-specific

Under `QWEN35_SGLANG_MOE_RUNNER_BACKEND=triton`:

- `routed_w1`: `0.0237` / `0.0784`
- `routed_w2`: `0.0264` / `0.0516`
- `routed_w3`: `0.0386` / `0.0777`
- full `routed_experts`: `0.0584` / `0.1066`
- full adapter: `0.0800` / `0.1207`

Conclusion:

- routed LoRA math was not inherently catastrophic
- the huge blowup was tied to the default FlashInfer TRTLLM MoE path

### 5. FlashInfer TRTLLM BF16 restore bug was found and fixed

Run `ap-eJ3RYwaKeEp7Qa3cwEfhQy` used a focused tensor probe on the FlashInfer TRTLLM BF16 routed-expert layout.

It showed that the old restore path in [unquant.py](/Users/jm/sglang/python/sglang/srt/layers/quantization/unquant.py) was only reshaping blocked/permuted live tensors instead of truly inverting the FlashInfer transform.

Probe result:

- `w13` current restore vs original: `max_abs=10.5`, `mean_abs=3.5060`
- `w2` current restore vs original: `max_abs=10.4375`, `mean_abs=3.4665`
- exact inverse restore vs original: `0.0 / 0.0` for both

That bug was fixed by making `maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load()` do the actual inverse transform.

Post-fix routed-only validation on default FlashInfer:

- before fix: `routed_experts = 5.8416 / 4.1023`
- after fix: `routed_experts = 0.0552 / 0.0985`

Conclusion:

- the original catastrophic routed-expert corruption was a real backend-specific layout/restore bug
- that bug is fixed

### 6. Full adapter after the routed restore fix

Saved run output in [latest_modal_validate_qwen35_merged_lora_logprob_diff_split.json](/Users/jm/sglang/scripts/latest_modal_validate_qwen35_merged_lora_logprob_diff_split.json) shows the current full-adapter state on the default FlashInfer backend:

- `overall_base_mean_abs = 0.0192`
- `overall_base_prefill_mean_abs = 0.0475`
- `overall_delta_mean_abs = 0.08295`
- `overall_delta_prefill_mean_abs = 0.13636`

Prompt-level behavior:

- prompt 1 often matches exactly
- prompt 2 remains the most consistently unstable
- prompt 3 still tends to diverge early

Conclusion:

- the catastrophic routed failure is gone
- residual merged-model drift remains

### 7. Explicit deterministic inference did not materially change the picture

We made deterministic inference an explicit harness knob:

- [test_qwen35_merged_lora_logprob_diff.py](/Users/jm/sglang/test/manual/test_qwen35_merged_lora_logprob_diff.py)
- [modal_validate_qwen35_merged_lora_logprob_diff_split.py](/Users/jm/sglang/scripts/modal_validate_qwen35_merged_lora_logprob_diff_split.py)
- [modal_validate_qwen35_merged_lora_logprob_diff.py](/Users/jm/sglang/scripts/modal_validate_qwen35_merged_lora_logprob_diff.py)

Then reran the family cuts with deterministic inference explicitly enabled. The resulting metrics were effectively unchanged.

Conclusion:

- the remaining gap is not explained by inference nondeterminism

### 8. Post-fix family ablations on default FlashInfer

With the routed restore bug fixed and deterministic inference enabled:

| Subset | `overall_delta_mean_abs` | `overall_delta_prefill_mean_abs` | Free-run mismatches |
| --- | ---: | ---: | ---: |
| `all` | `0.08295` | `0.13636` | `2` |
| `routed_experts` | `0.05517` | `0.09851` | `3` |
| `attention` | `0.03920` | `0.08760` | `3` |
| `attention,shared_expert,lm_head` | `0.04686` | `0.08283` | `2` |
| `shared_expert` | `0.02191` | `0.07421` | `1` |
| `lm_head` | `0.01701` | `0.02714` | `2` |

Conclusion:

- `routed_experts` is still the largest single contributor
- `attention` is the largest non-routed contributor
- `shared_expert` is smaller but real
- `lm_head` is smallest

### 9. Attention subfamily split

Post-fix, deterministic, default FlashInfer:

| Subset | `overall_delta_mean_abs` | `overall_delta_prefill_mean_abs` | Free-run mismatches |
| --- | ---: | ---: | ---: |
| `linear_attn` | `0.03298` | `0.09596` | `0` |
| `self_attn` | `0.02477` | `0.07197` | `2` |

Conclusion:

- `linear_attn` contributes the larger teacher-forced prefill gap
- `self_attn` contributes more to free-run token instability on this prompt set

A reasonable interpretation is:

- `linear_attn` is nudging logits more broadly during prompt processing
- `self_attn` is more often pushing greedy decode over an argmax boundary

### 10. Partial routed sub-splits exposed and fixed a second FlashInfer-specific bug

After the first routed restore fix, trying `routed_w1`, `routed_w2`, or `routed_w3` alone on default FlashInfer still failed.

Observed failure:

- FlashInfer postprocess asserted on 3D tensors where it expected 2D inputs
- Modal worker later died with exit code `137`

Cause:

- a partial routed hot merge restored only the tensor touched by the loader
- the untouched sibling (`w13_weight` or `w2_weight`) could remain in blocked live layout
- `_finalize_flashinfer_moe_layer_after_merge()` then called `process_weights_after_loading()` on a mixed canonical/blocked pair

That bug was fixed in [lora_merge_loader.py](/Users/jm/sglang/python/sglang/srt/model_loader/lora_merge_loader.py#L496) by restoring both routed tensors back to canonical load shape before rerunning the shared FlashInfer postprocess.

Regression coverage:

- new helpers and partial-routed test in [test_lora_merge_loader.py](/Users/jm/sglang/test/registered/unit/model_loader/test_lora_merge_loader.py#L290)
- new partial-routed regression at [test_lora_merge_loader.py](/Users/jm/sglang/test/registered/unit/model_loader/test_lora_merge_loader.py#L723)
- Modal unit harness `ap-sD7vZqTQ5GDUOxL2CfEDIq`: 6 tests, 0 failures, 0 errors

### 11. Post-fix routed `w1/w2/w3` validation on default FlashInfer

After the partial-routed finalization fix:

| Subset | App | `overall_delta_mean_abs` | `overall_delta_prefill_mean_abs` | Free-run mismatches |
| --- | --- | ---: | ---: | ---: |
| `routed_w1` | `ap-yiENd0sfsJUTKDAZkqbmgV` | `0.02964` | `0.06240` | `2` |
| `routed_w2` | `ap-BDYnBbGAw8ZchrBbJPSBCq` | `0.02420` | `0.04693` | `2` |
| `routed_w3` | `ap-KvfQiP0RtHofRWDZSoJufh` | `0.03615` | `0.07854` | `2` |

Conclusion:

- the partial-routed FlashInfer crash is fixed
- all three routed lanes now complete normally on the production backend
- `w3` looks largest, `w2` smallest, but all three are now in the same mild range

## Current Mental Model

The best current model is:

1. There is real baseline HF-vs-SGLang drift on Qwen3.5.
   - about `0.019` completion mean abs
   - about `0.048` prefill mean abs

2. The merged-model gap is real and begins during teacher-forced prefill.
   - it is not just free-run sampling drift
   - it is not explained by nondeterminism

3. The original catastrophic failure was mostly backend-specific routed-expert corruption in the FlashInfer TRTLLM hot-update path.
   - bad BF16 restore inverse
   - then a second bug in partial-routed finalize ordering

4. Those catastrophic routed bugs are now fixed.
   - routed-only and routed-subsplit runs are now mild, not catastrophic
   - partial routed cuts no longer crash

5. The remaining full-adapter gap is smaller and distributed.
   - `routed_experts` is still the largest single contributor
   - `attention` is the next largest contributor
   - `shared_expert` and `lm_head` are smaller but nonzero

6. The residual full-model gap is likely a combination of several small effects rather than one remaining catastrophic bug.
   - routed experts still matter
   - attention matters independently
   - the full model is somewhat worse than any single family because these deltas interact nonlinearly through later layers
   - greedy decode mismatches can still happen from relatively small logprob shifts

## What Seems Most Likely To Be Causing The Remaining Gap

Most likely causes, in order:

1. Residual routed-expert/backend mismatch that is no longer catastrophic but still present.
   - especially `w3`, then `w1`, then `w2`

2. Attention-path mismatch.
   - `linear_attn` looks strongest on teacher-forced prefill
   - `self_attn` looks stronger on free-run token divergence

3. Smaller additive differences in `shared_expert` and `lm_head`.

What no longer fits the evidence:

- “the whole problem is baseline HF-vs-SGLang drift”
- “the whole problem is sampling nondeterminism”
- “the whole problem is routed-expert corruption”

The evidence now says the remaining gap is mixed, with routed experts still the largest contributor but no longer the only serious one.

## Best Next Steps

If the goal is to keep narrowing the remaining `0.08 / 0.14` full-model gap, the highest-value next steps are:

1. Activation-level teacher-forced tracing on one or two failing prompts.
   - compare HF merged vs SGLang merged hidden states layer by layer
   - start at the first layer that contains `linear_attn` and MoE experts

2. Combination ablations, not just single-family ablations.
   - `routed_experts + attention`
   - `routed_experts + linear_attn`
   - `routed_experts + self_attn`

3. Prompt-focused tracing on the unstable prompts.
   - prompt 2 and prompt 3 are consistently more informative than prompt 1

4. If we want the fastest path to a remaining culprit:
   - prioritize `linear_attn` and routed `w3` interaction first

## Bottom Line

The investigation has already paid off:

- we proved the original failure was real
- we proved it was mostly adapter-effect, not baseline drift
- we localized the catastrophic part to FlashInfer TRTLLM routed hot updates
- we fixed two real backend-specific bugs
- we brought routed-only validation from catastrophic (`~5.84 / 4.10`) to mild (`~0.055 / 0.099`)
- we brought routed `w1/w2/w3` partial updates from crashing to passing

What remains is a smaller, real, distributed merged-model gap. The current best guess is: residual routed-expert differences plus attention-path differences, especially `linear_attn` on prefill and `self_attn` on greedy decode stability.
