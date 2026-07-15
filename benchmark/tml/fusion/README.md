# Inkling kernel-fusion micro-benchmarks

Per-optimization old-vs-new latency + correctness for the kernel-fusion work on
`cheng/fuse-1` (vs parent `c262556c2`). See **[RESULTS.md](RESULTS.md)** for the
measured numbers and analysis.

## Files

| File | Optimization |
|------|--------------|
| `_harness.py` | shared: CUDA-event `bench`, `Collector` (correctness + latency + JSON), table printer |
| `_sconv_baseline.py` | **vendored** old `kernels/sconv.py` @ `c262556c2` (Helion `causal_conv1d` + `update_sconv_cache` baselines) |
| `_copy_if_needed.py` | **vendored** `copy_if_needed` (the track fusion removes it from `layers/sconv.py`; opt5/6/8 baseline) |
| `opt1_causal_conv1d.py` | causal_conv1d extend: Helion→Triton — correctness vs fp32 ref + naive latency |
| `opt1b_extend_latency.py` | causal_conv1d **fair** latency — one shape per fresh process (args: `T D B [W]`) |
| `opt2_rmsnorm.py` | RMSNorm wrapper: custom-Triton → `sgl_kernel` + reshape |
| `opt3_qk_norm.py` | QK-norm: 2×(copy+norm) → fused in-place |
| `opt4_fused_add_rmsnorm.py` | decoder residual: add+norm → `fused_add_rmsnorm` |
| `opt5_gather_scatter.py` | sconv-cache gather→scatter fusion |
| `opt6_draft_extend.py` | draft-extend sconv cache update fusion |
| `opt7_decode_update.py` | decode `fused_causal_conv1d_update_decode` (+ unfused cross-check via `_update_sconv_cache_helion_kernel`) |
| `opt8_track_fusion.py` | decode track-copy folded into the decode kernel (needs the track-fused `kernels/sconv.py`) |
| `opt9_update_sconv_cache.py` | `update_sconv_cache` Helion→Triton rewrite (needs the Triton `_update_sconv_cache_kernel`) |

## Run

Run inside a container that mounts this repo at `/sgl-workspace/sglang` on a B200
(needs `helion`, `triton`, `sgl_kernel`). See the "Reproduce" section of
`RESULTS.md`. Each script prints a table plus a delimited JSON block.

These are diagnostic micro-benchmarks (not part of the test suite); the vendored
`_sconv_baseline.py` is a verbatim snapshot of old code for A/B timing only.
