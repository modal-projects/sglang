# K3 on MI355X TP8: comm / fusion gap findings

Target: close the per-step gap to B300 `kimi-k3-fast` (12.85 / 20.61 / 35.46 ms at bs1/4/16 vs MI355 #716 ~15.0 / 23.6 / 41.1 ms).
Gate for server work: microbench win >= 0.3 ms/step with BF16-tolerance accuracy. Nothing below cleared it, so there are no server A/B or GSM8K/AgenticV2 runs.
Microbenches on MI355X (1 or 8 GPUs), HIP graphs, #716 image, arya-dev.

## 1. Large-token fused AR + residual + RMSNorm (+FP8), 65..512 tokens (parked)

Branch `devin/k3-mi355-arfuse2` (d11f3d8a base), opt-in `SGLANG_K3_AR_AGG_LT=1`; stack copy on 17bfcaa: `devin/k3-mi355-arfuse2-stack`.
Design: one launch, up to 80 producer blocks reduce-scatter (2 tokens each) into private IPC regions with per-token flags; a co-resident consumer grid gathers, mixes the residual, applies RMSNorm (optional FP8).

| mode | T | unfused us | fused us |
|---|---:|---:|---:|
| AR | 72 | 18.58 | 18.80 |
| AR | 128 | 22.39 | 23.98 |
| AR | 144 | 26.19 | 28.30 |
| AR | 256 | 34.79 | 42.27 |
| AR | 288 | 39.51 | 47.39 |
| AG | 72 | 15.49 | 16.67 |
| AG | 128 | 17.77 | 20.31 |
| AG | 144 | 20.57 | 24.17 |
| AG | 256 | 26.54 | 35.66 |
| AG | 288 | 29.51 | 39.66 |

Accuracy: prefix outputs bit-exact; normed output <= 1 BF16 ulp (~0.03% of elements); FP8 <= 1 FP8 step (~6e-6 of elements).
It loses because each extra 80-token producer round adds ~8 us of serial cross-GPU waits. The only win is nvb<=4, T<=96 (0.8-1.5 us/call, ~0.1-0.2 ms/step). Noted, not shipped.

Related:
- `SGLANG_K3_AR_FUSION` is B300-only (auto-probe needs SM100/SM103); #716 does not set it.
- `routed_scaling_factor` = 1.0, and the latent norm is already folded into w_up, so there is nothing to fold.
- Standalone attn_res aggregation at bs9-16 (~1.2 ms/step) is latency-bound (4.7-8.4 us/call, flat in T). The current 2x448 launch shape is fastest. Removing it needs fusion into the AR (lost) or a GEMM prologue (not pursued).

## 2. KDA glue (#716 rocprof, rank 287, `k3-mi355-prof`)

At bs1 each KDA layer runs serially on the main stream: qkvg GEMM -> `tiny_n_gemm` (bfa, 5.2 us) -> `tiny_k_gemm` (f_b, 4.75 us) -> `fused_kda_conv_gating_verify` (15.7 us) -> `norm_gemm` (9.4 us) -> `ar_agg`.

- f_b folded into the verify kernel (`f_b_weight=` kwarg, unwired; `bench_kda_verify_fb.py`): bit-exact, but 2.2-2.6x slower (B1 16.4 -> 37.0 us, B16 56.7 -> 136 us). Each V-tile program recomputes the 128x128 dot. `tiny_k_gemm` takes only ~2.5 us in-graph, so the cap is ~0.17 ms/step.
- bfa `tiny_n_gemm`: the in-proj fusion (`SGLANG_ROCM_K3_FUSE_KDA_INPROJ`) is skipped because qkvg is FP8 on #716. It applies with `TARGET_DENSE_FP8=off`; on FP8 it would need beta/f_a quantization (accuracy risk).
- o_norm above M=16: `layer_norm_gated` 4.8 us + o_proj (7.3 us at bs4, 10.2 us at bs16). The fused `norm_gemm` is already slower than a plain GEMM, so extending it would save ~0.07 ms at bs4 and <= 0.33 ms at bs16 (best case).
- MLA glue (3 x 24 launches, ~4.7 us each): only `fused_qk_rmsnorm` is removable (~0.06-0.11 ms).
- In rocprof, small MI355 kernels bottom out at ~4.2-4.8 us (~2.5 us back-to-back in a graph; an empty captured kernel is 2.1 us).

## 3. MLA verify (`k3_mla_verify_auto`, HK regime, fp8 KV), on 138e94f split fix

Branch `devin/k3-mi355-arfuse2-mla`. Benches: `bench_mla_verify_bd.py` (roofline + nsplit sweep), `bench_mla_verify_lsweep.py`, `bench_mla_verify_ts.py` (needs `K3HK_TS=1` debug build).

| bs | auto @90k us | HBM roofline us |
|---|---:|---:|
| 1 | 30.5 | 10.4 |
| 2 | 35.9 | 20.9 |
| 4 | 53.5 | 41.7 |
| 8 | 97.5 | 83.2 |
| 16 | 184 | 166 |

bs8/16 are within 10-15% of roofline. The bs1-4 gap is fixed cost: stage1 is ~14.7 us even at L=2k and streams at ~8 TB/s beyond that; reduce1 ~5 us flat; Gluon regime-select launch ~2.5 us.

Per-CTA phases (s_memrealtime, bs1, L=90k, 235 CTAs, 19.9 us span):

| phase | us |
|---|---:|
| Q staging + fp8 quant | 4.9 |
| first KV tiles | 1.3 |
| main loop (~5.5 tiles) | 10.0 |
| fp8 partial epilogue | 3.0 |

The main loop costs ~1.3-1.8 us per 64-token tile even with 8 CTAs, so it is bound by MFMA latency (1 CTA/CU, 3 compute waves). More splits can't help.

Removable pieces (not implemented):

| piece | est. us/layer | risk |
|---|---:|---|
| producer (`absorb_q_cat_cache`) writes fp8 Q + row scale; drop in-kernel Q quant | 3-4 | moderate: asm Q path + producer dataflow |
| trim fp8 partial epilogue | 1-2 | moderate: hand-scheduled asm |
| fold Gluon regime select into HK (no extra launch in HK regime) | ~2.5 | low-moderate: short-L regime selection |
| fuse/cheapen reduce1 (grid-barrier merge when bs*nsplit <= 256) | 2-3 | moderate: co-residency assumption |

Total ~9 us/layer x 24 layers ~ 0.2 ms/step at bs1, less at bs4: below the 0.3 ms gate.
