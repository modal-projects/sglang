# Inkling Fused Spec-Decode Target-Verify Kernels — Design & Results Journal

Two fused CUDA-JIT kernels for the Inkling (Moonrise) **target-verify** step of
EAGLE speculative decoding, each replacing a multi-kernel chain (and its
intermediate HBM round trips) with a single launch. Companion to the decode
fusion (`inkling_ar_fused_decode.cuh::ArSconvNormKernel`, merged in PR #53/#59) and
the custom all-reduce (`CUSTOM_ALLREDUCE_JOURNAL.md`).

All numbers: B200, TP=4/8, bf16, hidden=6144, head_dim=128, draft_token_num
Q=9. di1/di2.

---

## 1. Motivation

At target-verify (`forward_mode.is_target_verify()`), every request presents
`Q` draft tokens, so the batch is `T = batch * Q` rows. Two per-layer chains
dominate the non-GEMM time, each a string of small launches with intermediate
HBM traffic:

1. **Post-block chain** (attn-side: `wo_ud AR → attn_sconv → mlp_norm`;
   MoE-side: `MoE AR → mlp_sconv → attn_norm`): all-reduce, then the
   verify-mode short-conv, then `save_intermediate_conv_windows`, then the
   fused-add RMSNorm — 4 kernels.
2. **Attention prologue** (post-qkvr): `k_sconv`, `v_sconv`, their two
   `save_intermediate_conv_windows`, the per-head q/k RMSNorm, and the
   KV-cache store — 6 ops.

The decode fusion already collapsed chain (1)'s decode analog; this work
extends it to verify and adds the prologue.

## 2. Kernel 1 — verify AR chain (`inkling_ar_sconv_norm_verify_kernel`)

Fuses `{v5 push all-reduce → causal_conv1d (extend/verify semantics) →
save_intermediate_conv_windows → residual-add + RMSNorm}`. One block per token
row; one 16B vec (8 channels) per thread (VPT tunable, VPT=1 optimal).

- **AR**: identical to the decode fusion — multicast-push this rank's partial
  row into the per-rank v5 staging slot, ONE per-block barrier, then reduce
  the `world` staged shards locally (fp32, rounded to bf16 exactly as the
  standalone AR store would).
- **Cross-token conv**: verify's conv needs up to `W-1` in-sequence neighbor
  rows. Rather than a cross-block dependency, each block **re-reduces** its
  needed neighbor rows straight from the staging shards (`~kNumGPU × 16B`
  extra local L2 reads per tap). Sequence identity is `seq = t / Q`,
  `bos = seq*Q` — the target-verify packed layout. Per-sequence prefix taps
  come from the read-only working conv cache, gated by `has_initial_state &
  (cache_indices != PAD)`.
- **save_windows**: the per-position window after draft token `tq` is raw
  copies of `{cache prefix rows | re-reduced x rows}` — bytes the kernel
  already holds. Written to `intermediate_conv_window` (consumed later by
  `update_conv_state_after_mtp_verify`). The **working cache is NOT updated**
  at verify — matching the unfused path.
- **norm**: `r = residual_in + bf16(y)`; block-reduce `sum(r²)`; write
  `residual_out = bf16(r)` and `hs = bf16(r · rsqrt(mean+eps) · γ)`.

Barrier slots grew 128 → 256 and the v5 staging region 96 → 160 rows to cover
`T = 16×9 = 144`.

## 3. Kernel 2 — attention prologue (`inkling_attn_prologue_kernel`)

Fuses `{k_sconv + v_sconv + both save_windows + q/k per-head RMSNorm + KV-cache
store}` after the qkvr projection. `rel_logits_proj` overlaps on the alt
stream. One block per token; lane roles by vec index — `[0, Dq/8)` q-norm,
then `Dkv/8` k (conv+norm), then `Dkv/8` v (conv). head_dim=128 → a head is 16
contiguous lanes, reduced with width-16 warp shuffles (Dq/8, Dkv/8 are
multiples of 16, so head groups never straddle a warp). Conv taps are read
straight from the strided qkvr rows; prefixes from the read-only k/v conv
caches; windows to the two intermediate buffers.

- **KV store** (user-requested full fusion): full-attention layers write
  `out_cache_loc` into the full KV pool; local/SWA layers write the backend's
  pre-translated `swa_out_cache_loc`
  (`get_attn_backend().forward_metadata.swa_out_cache_loc`) into the SWA
  sub-pool (`get_key_buffer(layer_id)` dispatches by layer). A `loc ≥ 0`
  guard skips the SWA full→swa `-1` sentinels. The attention call then runs
  `save_kv_cache=False`. Bf16 KV pool only.
- **bf16 round before k-norm**: the unfused pipeline writes the conv output to
  HBM as bf16 before the norm reads it back, so the fused kernel rounds `y`
  to bf16 before the k-norm reduction to stay bit-identical.

## 4. Correctness

Both kernels validated **bit-exact against the production kernels** (the real
sglang path, not a torch reimpl): jit `causal_conv1d`, triton
`save_intermediate_conv_windows`, Inkling `RMSNorm` / flashinfer
`fused_add_rmsnorm`.

- Kernel 1 (`validate_ar_fused.py` [F5]/[F6], TP4 + TP8): conv windows +
  residual **bit-exact**; normed hs **bit-identical** (exact-match 1.0000)
  across B ∈ {1,2,4,8,16}, PAD sequences, cache-mask edges, all VPT variants,
  and CUDA-graph capture/replay.
- Kernel 2 (standalone `test_prologue.py`): k/v intermediate windows, v_out,
  q_out, k_out, and both KV-buffer scatters all **exact-match 1.0000**
  (including a PAD sequence). The bf16-round fix took k_out/k_buf from 0.7335
  → 1.0000.

**E2E correctness** (di1 TP4, EAGLE multi-layer, real acceptance — NO
`SIMULATE_ACC_LEN`): gsm8k **0.917 / Invalid 0.000** with both verify fusions
+ custom AR live.

> ⚠️ **`SGLANG_SIMULATE_ACC_LEN` produces garbage output by design** —
> `generate_simulated_accept_index` does `predict.fill_(100)` and force-accepts
> regardless of correctness. Accuracy under it is ~0.000 and means NOTHING;
> only throughput + a real-acceptance run are valid correctness signals. (Cost
> me several server restarts before I recognized it — see the memory note.)

## 5. Kernel benchmarks (vs the production unfused chains)

Kernel 1, µs, min-of-5×20:

| B×Q | unfused chain | fused | speedup |
|---:|---:|---:|---:|
| 1×9 | 54.4 | 15.1 | **3.60×** |
| 4×9 | 54.1 | 17.9 | 3.02× |
| 8×9 | 54.0 | 20.7 | 2.62× |
| 16×9 | 54.4 | 26.1 | 2.08× |

(Kernel 2's standalone timing folds into the e2e prologue; its win is removing
5 launches + 4 intermediate HBM round trips and overlapping rel_logits.)

## 6. E2E throughput (di1 TP4, EAGLE, `SIMULATE_ACC_LEN=5`, ol=128)

Both verify fusions + custom AR vs the unfused-spec baseline:

| bsz | il=4096 | il=8192 | il=16384 |
|---:|---|---|---|
| 1 | 309.7 (+8.1%) | 307.9 (+8.3%) | 302.0 (+7.6%) |
| 4 | 954.3 (+5.4%) | 941.6 (+5.9%) | 935.1 (+6.8%) |
| 16 | 2471.3 (+6.0%) | 2440.5 (+5.3%) | 2386.5 (+5.9%) |

tok/s output; bs=1/il=4096 ITL 3.49 → 3.23 ms. **+5–8% across the sweep.**
Trace-verified in the target-verify decode graph: `inkling_ar_sconv_norm_verify`
(650) + `inkling_attn_prologue` (330) fire; `fused_decode_update` gone,
`save_intermediate` down to ~10 (residual conv/AR counts are the EAGLE
draft-model passes, out of scope for these fusions).

## 7. Integration & gating

- `SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM` (kernel 1, decode+verify) and
  `SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE` (kernel 2, verify only), both
  requiring `SGLANG_OPT_USE_INKLING_CUSTOM_AR`.
- `ar_sconv_norm_fusable()` gates on `is_decode() or is_target_verify()`,
  ≤ staging-region rows, both flags, resources present — a pure per-forward
  function so producer (`reduce=False`) and consumer always agree.
- The prologue fires only for `is_target_verify() && head_dim==128 && kv_conv`
  and keeps the unfused path for every other mode/shape.

## 8. Open items

- Config re-sweep of kernel 1's VPT at verify shapes (T up to 144); VPT=1 held
  at decode shapes.
- Draft-model passes (EAGLE) are not fused — a future target if the draft
  attention/sconv shows up in traces.
- TP8 verify throughput sweep (kernels validated correct at TP8; throughput
  measured at TP4).
