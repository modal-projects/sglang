# Inkling Triton → CUDA-JIT rewrite plan

Goal: reimplement the Triton kernels developed for Moonrise as CUDA-JIT kernels in
the repo's house style (`python/sglang/jit_kernel/`: header-only templated kernel
class in `csrc/<group>/<name>.cuh`, loaded via `load_jit(...)` +
`cuda_wrappers=[(export, "Klass<args>::run")]`, thin `@cache_once` Python wrapper),
then re-run the fine-grained fusion benchmark journal (`opt1`–`opt9`) with a new
`cuda_jit` column verified against the existing independent fp32 reference.

Target arch: **B200 / sm_100** (Blackwell). Toolchain verified on di2:
`tvm_ffi 0.1.9`, a real JIT compile+load+run of `jit_kernel.norm.rmsnorm` succeeds.

## Tier 1 — sconv family (STARTING NOW)

The conv/state kernels written specifically for this model. Each has a ready
correctness+latency harness in `benchmark/tml/fusion/`.

| # | Python entry | Triton kernel | file | bench |
|---|--------------|---------------|------|-------|
| 1 | `causal_conv1d` | `_causal_conv1d_fwd_with_prefix_kernel` | kernels/sconv.py | opt1 |
| 2 | `update_sconv_cache` | `_update_sconv_cache_kernel` | kernels/sconv.py | opt9 |
| 3 | `fused_causal_conv1d_update_decode` | `_fused_causal_conv1d_update_decode_kernel` | kernels/sconv.py | opt7/opt8 |
| 4 | `save_intermediate_conv_windows` | `_save_intermediate_conv_windows_kernel` | kernels/sconv.py | opt6 |
| 5 | `fused_gather_scatter_to_sconv_cache` | `_fused_gather_scatter_to_sconv_cache_kernel` | layers/sconv.py | opt5 |
| 6 | `fused_draft_extend_sconv_cache` | `_fused_draft_extend_sconv_cache_kernel` | layers/sconv.py | opt6 |

Pilot = #1 (`causal_conv1d`, extend): already benchmarked + fixed-config; the opt1
harness has a documented fp32 reference and a CUDA-vs-ref slot to add.

Deliverables per kernel:
- `jit_kernel/csrc/inkling/<name>.cuh` — templated kernel class `<W, USE_SILU, USE_RESIDUAL, IS_DECODE, DType>::run(TensorView...)`.
- `jit_kernel/inkling_sconv.py` — `@cache_once` module getter + drop-in Python fn matching the Triton signature.
- opt harness edit — add `CUDA` impl, a `CUDA-vs-ref` correctness check, and a `cuda_jit` latency column alongside `old_helion`/`new_triton`.
- (integration, later) route `tml/kernels/sconv.py` entrypoints to the CUDA impl behind an env flag (`SGLANG_INKLING_SCONV_BACKEND=cuda|triton`), per env-var conventions.

Semantics to preserve (from opt harness fp32 refs; bf16, fp32 accumulate):
- conv: per packed token t in seq s, bos=cu[s], slot=safe_idx[s]; tap iw shifted=t-(W-1)+iw;
  in-seq history (shifted≥bos) reads x[shifted], else prefix reads cache[slot,pp]*cache_mask[s];
  out=act(Σ tap·weight[:,iw]) (+x[t] if residual). Decode: cache taps iw<W-1, x tap iw=W-1.
- cache update: new state = last W-1 of [old_state(gated) ++ x_seq]; PAD/empty slots untouched.
- W ∈ {3,4,5} in the correctness sweep (real model W=4); template over W.
- D not always a multiple of 256 (e.g. 384) → mask the D tail.
- T=0 (idle) → empty, no launch.

## Tier 2 — remaining Inkling-specific Triton (planned, after Tier 1 sign-off)

`python/sglang/tml/kernels/` and `layers/`, no fusion-journal bench yet (would add
`opt10+` benches with fp32/torch refs before porting):
- `inkling_moe.py` — 11 `@triton.jit` kernels (routed-MoE gather/scatter/gemm-glue, the
  largest chunk). Needs its own correctness refs (vs a torch grouped-MoE reference).
- `gate_topk.py` (6), `sigmoid_gate_topk_renorm.py` (1) — router top-k / sigmoid+renorm.
  Note `jit_kernel/inkling_gate_topk_renorm.py` + `moe_topk_sigmoid`/`moe_fused_gate`
  already exist — check overlap before porting.
- `layers/moe.py` (2) — MoE glue.

## Tier 3 — upstream-derived quant (only if explicitly requested)

`quantize.py` (3), `nvfp4_scale.py` (2). Largely generic nvfp4/fp8 quant patterns
that overlap existing `sgl_kernel` / `jit_kernel/{nvfp4,fp8_quantize,mxfp8}` paths;
port only if we want a single owned implementation.

## Already CUDA-JIT (no work)

opt2 rmsnorm, opt3 qk-norm, opt4 fused_add_rmsnorm — `csrc/elementwise/{rmsnorm,
qknorm,fused_add_rmsnorm}.cuh` exist in-repo (the model uses sgl_kernel/jit for these).

## Verification protocol (per kernel)

1. Compile via `load_jit` on di2 GPU 4 (server on 0–3 untouched).
2. Correctness: CUDA-vs-fp32-ref across the opt harness's full shape sweep (same
   atol/rtol 2e-2), plus CUDA-vs-Triton bit-closeness where applicable.
3. Latency: median-100 CUDA-event vs Triton (and old Helion) in the opt harness.
4. Record in RESULTS.md as a new `cuda_jit` column.
