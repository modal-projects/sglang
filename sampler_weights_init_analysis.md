# LoRA Target Analysis for `sampler_weights_init.safetensors`

## What was analyzed

- Artifact: `./sampler_weights_init.safetensors`
- Working assumption: this file is a LoRA adapter intended for a Qwen3.5-family hybrid/MoE language model
- Goal: determine which base-model weight modules the adapter targets, so SGLang LoRA merge support can be scoped correctly

## Executive Summary

This adapter targets a Qwen3.5 hybrid/MoE model with:

- 40 decoder layers total
- A repeating hybrid attention pattern:
  - linear attention on layers `0,1,2,4,5,6,...,38`
  - full attention on layers `3,7,11,...,39`
- MoE expert MLP LoRAs on every layer
- shared-expert MLP LoRAs on every layer
- one final LoRA on `unembed_tokens`, which maps to `lm_head` in SGLang

For merge support, the effective SGLang targets are:

- `linear_attn.in_proj_q`
- `linear_attn.in_proj_k`
- `linear_attn.in_proj_v`
- `linear_attn.in_proj_z`
- `linear_attn.out_proj`
- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.shared_expert.gate_proj`
- `mlp.shared_expert.up_proj`
- `mlp.shared_expert.down_proj`
- `mlp.experts.w1`
- `mlp.experts.w2`
- `mlp.experts.w3`
- `unembed_tokens`

Those adapter-facing names resolve to these actual SGLang base params:

- `linear_attn.in_proj_{q,k,v,z}` -> `linear_attn.in_proj_qkvz.weight`
- `linear_attn.out_proj` -> `linear_attn.out_proj.weight`
- `self_attn.{q,k,v}_proj` -> `self_attn.qkv_proj.weight`
- `self_attn.o_proj` -> `self_attn.o_proj.weight`
- `mlp.shared_expert.{gate,up}_proj` -> `mlp.shared_expert.gate_up_proj.weight`
- `mlp.shared_expert.down_proj` -> `mlp.shared_expert.down_proj.weight`
- `mlp.experts.w1` and `mlp.experts.w3` -> `mlp.experts.w13_weight`
- `mlp.experts.w2` -> `mlp.experts.w2_weight`
- `unembed_tokens` -> `lm_head.weight`

## How this was deduced

There was no `adapter_config.json` next to the safetensors file, and the safetensors metadata was empty. So the analysis was done from:

1. The safetensors header itself
2. The tensor key names and shapes
3. The existing Qwen3.5 model definitions in SGLang
4. The existing SGLang LoRA merge-loader name canonicalization and packed-target logic

### 1. Parsed the safetensors header directly

The file header contains all tensor names and shapes even without loading the full tensors. From that header:

- tensor count: `862`
- LoRA tensor kinds: only standard `.lora_A.weight` / `.lora_B.weight`
- no embedded metadata block

That established that this is a normal PEFT-style LoRA adapter, not some custom serialized format.

### 2. Enumerated the target keys and grouped them by layer

The key set showed:

- 40 logical decoder layers: `layers.0` through `layers.39`
- per-layer LoRA targets under either:
  - `linear_attn.*`
  - `self_attn.*`
  - `mlp.experts.*`
  - `mlp.shared_expert.*`
- one non-layer key:
  - `base_model.model.model.unembed_tokens`

By grouping modules per layer, the attention pattern became clear:

- layers `0,1,2` use `linear_attn.*`
- layer `3` uses `self_attn.*`
- layers `4,5,6` use `linear_attn.*`
- layer `7` uses `self_attn.*`
- this repeats through layer `39`

So the attention layout is inferred from the adapter keys themselves as a repeating `3 linear + 1 full attention` pattern.

### 3. Used tensor shapes to validate the interpretation

Representative shapes from the safetensors header:

- `linear_attn.in_proj_q`: `A=(32, 2048)`, `B=(2048, 32)`
- `linear_attn.in_proj_k`: `A=(32, 2048)`, `B=(2048, 32)`
- `linear_attn.in_proj_v`: `A=(32, 2048)`, `B=(4096, 32)`
- `linear_attn.in_proj_z`: `A=(32, 2048)`, `B=(4096, 32)`
- `linear_attn.out_proj`: `A=(32, 4096)`, `B=(2048, 32)`

- `self_attn.q_proj`: `A=(32, 2048)`, `B=(8192, 32)`
- `self_attn.k_proj`: `A=(32, 2048)`, `B=(512, 32)`
- `self_attn.v_proj`: `A=(32, 2048)`, `B=(512, 32)`
- `self_attn.o_proj`: `A=(32, 4096)`, `B=(2048, 32)`

- `mlp.shared_expert.gate_proj`: `A=(32, 2048)`, `B=(512, 32)`
- `mlp.shared_expert.up_proj`: `A=(32, 2048)`, `B=(512, 32)`
- `mlp.shared_expert.down_proj`: `A=(32, 512)`, `B=(2048, 32)`

- `mlp.experts.w1`: `A=(1, 32, 2048)`, `B=(256, 512, 32)`
- `mlp.experts.w2`: `A=(256, 32, 512)`, `B=(1, 2048, 32)`
- `mlp.experts.w3`: `A=(1, 32, 2048)`, `B=(256, 512, 32)`

- `unembed_tokens`: `A=(32, 2048)`, `B=(248320, 32)`

The 3D expert tensors are the main clue that `mlp.experts.*` is not a dense MLP target. They are aggregated-across-experts LoRA tensors and need expert-aware handling in the merge path.

### 4. Cross-checked against SGLang's Qwen3.5 model definitions

The adapter key names line up with the hybrid/MoE implementation in:

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen2_moe.py`

Relevant model facts from the code:

- Qwen3.5 model construction chooses either `self_attn` or `linear_attn` per layer based on `config.layers_block_type`
- linear-attn layers use packed `in_proj_qkvz` and `out_proj`
- full-attn layers use packed `qkv_proj` and `o_proj`
- MoE layers use `Qwen2MoeSparseMoeBlock`
- shared expert uses `shared_expert.gate_up_proj` and `shared_expert.down_proj`
- routed experts use fused expert weights `w13_weight` and `w2_weight`

That is why the adapter-facing names are not always the same as the actual base parameter names that the merge implementation must touch.

### 5. Cross-checked against SGLang's LoRA merge-loader

The existing canonicalization and packed-resolution logic in:

- `python/sglang/srt/model_loader/lora_merge_loader.py`

confirms these name mappings:

- `unembed_tokens` is canonicalized to `lm_head`
- `.self_attn.` is stripped from checkpoint-style names during resolution
- `w1 -> gate_proj`
- `w3 -> up_proj`
- `w2 -> down_proj`
- `q_proj/k_proj/v_proj -> qkv_proj`
- `gate_proj/up_proj -> gate_up_proj`
- `in_proj_q/in_proj_k/in_proj_v/in_proj_z` need to land in packed `in_proj_qkvz`
- expert LoRAs ultimately load into `w13_weight` and `w2_weight`

This is important because the implementor should not treat the adapter key strings as the final in-memory parameter names.

## Concrete module inventory

### Layer-local adapter-facing targets

Linear-attention layers target:

- `linear_attn.in_proj_q`
- `linear_attn.in_proj_k`
- `linear_attn.in_proj_v`
- `linear_attn.in_proj_z`
- `linear_attn.out_proj`
- `mlp.experts.w1`
- `mlp.experts.w2`
- `mlp.experts.w3`
- `mlp.shared_expert.gate_proj`
- `mlp.shared_expert.up_proj`
- `mlp.shared_expert.down_proj`

Full-attention layers target:

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.experts.w1`
- `mlp.experts.w2`
- `mlp.experts.w3`
- `mlp.shared_expert.gate_proj`
- `mlp.shared_expert.up_proj`
- `mlp.shared_expert.down_proj`

Global non-layer target:

- `unembed_tokens`

### Counts

Counts below are by logical LoRA target, not by raw A/B tensor count:

- `linear_attn.in_proj_q`: 30 layers
- `linear_attn.in_proj_k`: 30 layers
- `linear_attn.in_proj_v`: 30 layers
- `linear_attn.in_proj_z`: 30 layers
- `linear_attn.out_proj`: 30 layers

- `self_attn.q_proj`: 10 layers
- `self_attn.k_proj`: 10 layers
- `self_attn.v_proj`: 10 layers
- `self_attn.o_proj`: 10 layers

- `mlp.experts.w1`: 40 layers
- `mlp.experts.w2`: 40 layers
- `mlp.experts.w3`: 40 layers

- `mlp.shared_expert.gate_proj`: 40 layers
- `mlp.shared_expert.up_proj`: 40 layers
- `mlp.shared_expert.down_proj`: 40 layers

- `unembed_tokens`: 1

## Mapping to actual SGLang merge destinations

This is the mapping the implementor should keep in mind.

| Adapter key target | SGLang destination param | Notes |
| --- | --- | --- |
| `linear_attn.in_proj_q` | `linear_attn.in_proj_qkvz.weight` | packed shard `(0,1,2)` family |
| `linear_attn.in_proj_k` | `linear_attn.in_proj_qkvz.weight` | packed shard `(0,1,2)` family |
| `linear_attn.in_proj_v` | `linear_attn.in_proj_qkvz.weight` | packed shard `(0,1,2)` family |
| `linear_attn.in_proj_z` | `linear_attn.in_proj_qkvz.weight` | packed shard `3` |
| `linear_attn.out_proj` | `linear_attn.out_proj.weight` | direct |
| `self_attn.q_proj` | `self_attn.qkv_proj.weight` | packed `q` shard |
| `self_attn.k_proj` | `self_attn.qkv_proj.weight` | packed `k` shard |
| `self_attn.v_proj` | `self_attn.qkv_proj.weight` | packed `v` shard |
| `self_attn.o_proj` | `self_attn.o_proj.weight` | direct |
| `mlp.shared_expert.gate_proj` | `mlp.shared_expert.gate_up_proj.weight` | packed shard `0` |
| `mlp.shared_expert.up_proj` | `mlp.shared_expert.gate_up_proj.weight` | packed shard `1` |
| `mlp.shared_expert.down_proj` | `mlp.shared_expert.down_proj.weight` | direct |
| `mlp.experts.w1` | `mlp.experts.w13_weight` | routed expert shard `w1` |
| `mlp.experts.w3` | `mlp.experts.w13_weight` | routed expert shard `w3` |
| `mlp.experts.w2` | `mlp.experts.w2_weight` | routed expert shard `w2` |
| `unembed_tokens` | `lm_head.weight` | canonicalized by merge loader |

## What is not targeted by this adapter

I did not find LoRA tensors for:

- `embed_tokens`
- `linear_attn.in_proj_b`
- `linear_attn.in_proj_a`
- `linear_attn.conv1d`
- `linear_attn.A_log`
- `linear_attn.dt_bias`
- attention-side `q_norm` / `k_norm`
- MoE router `mlp.gate`
- `mlp.shared_expert_gate`

That absence is based on the safetensors key set, not on speculation.

## Practical implementation notes

- Support for Qwen3.5 LoRA merge cannot be limited to dense attention only. This adapter requires:
  - hybrid attention handling
  - MoE expert handling
  - shared-expert handling
  - LM-head handling

- The expert tensors are aggregated across experts, so merge code must preserve expert indexing semantics rather than flattening them as ordinary dense projections.

- The adapter uses the old-style `w1/w2/w3` naming for experts, but SGLang's actual routed-expert params are `w13_weight` and `w2_weight`. That translation is mandatory.

- The presence of both `linear_attn.*` and `self_attn.*` in different layers is the strongest evidence that this is targeting the Qwen3.5 hybrid architecture rather than plain Qwen3 or plain Qwen3 MoE.

## Confidence and caveats

Confidence is high on the module mapping because it is supported by all of:

- the raw safetensors key names
- the tensor shapes
- the 40-layer hybrid layer pattern
- the Qwen3.5 model class structure in SGLang
- the existing Qwen3.5-aware merge-loader logic already present in SGLang

Caveats:

- No `adapter_config.json` was present, so values like `base_model_name_or_path`, `lora_alpha`, or training-time metadata were not available from sidecar config.
- The exact model variant name was inferred from the module/key structure, not read from adapter metadata.

## Relevant code references

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen2_moe.py`
- `python/sglang/srt/model_loader/lora_merge_loader.py`
- `test/registered/unit/model_loader/test_lora_merge_loader.py`

