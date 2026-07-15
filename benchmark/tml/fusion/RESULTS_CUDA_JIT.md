# Inkling sconv family — CUDA-JIT kernel benchmarks

Kernel-level correctness and latency for the CUDA-JIT ports of Moonrise's sconv-family
kernels (`python/sglang/jit_kernel/csrc/inkling/*.cuh`, wrappers in `jit_kernel/inkling_sconv.py`),
measured against the current Triton kernels and the older baselines via the `optN`
fusion-benchmark harnesses.

- **Hardware**: 1× NVIDIA B200 (sm_100), bf16. `tvm_ffi 0.1.9`, CUDA 13, torch 2.11.
- **Correctness**: CUDA output vs the harness's independent reference — fp32 reference at
  `atol=rtol=2e-2` for the conv kernels (opt1, opt7); **bit-exact** (`atol=0`) for the
  pure copy/select kernels (opt5, opt6, opt9).
- **Latency**: `_harness.bench`, median over 100 CUDA-event-timed launches (25 warmup),
  µs, lower is better. Ratios in the tables are `variant / baseline`.
- **Reproduce**: `CUDA_VISIBLE_DEVICES=<gpu> python3 benchmark/tml/fusion/<optN>.py`.

Columns: `old_*` = pre-Triton baseline (Helion / eager gather); `new_*` = current Triton;
`cuda_jit` = the CUDA-JIT port.

> **Refreshed 2026-07-05** (all `optN` re-run on the current di2 environment). The Triton
> kernels are now ~1.4–2.5× faster than the figures this doc was originally written against
> (newer Triton build/autotune), while the CUDA kernels are unchanged. Net: CUDA still wins
> on every kernel except the conv, but the margin over **current** Triton narrowed to
> **~1.0× (opt1 conv) / 1.1–1.35× (opt5/6/7/9)** — down from the original 1.9–3.8×. The
> CUDA-JIT port's remaining value is these modest wins plus removing the Triton/autotune
> dependency. Per-section summaries note the old vs refreshed numbers.

---

## opt1 — `causal_conv1d` (extend/prefill), silu + residual

Correctness: **CUDA 444/444 vs fp32 ref, 0 fails.** (The 372 fails in the 1333-check
aggregate are the fragile old Helion baseline, which diverges on multi-token shapes; the
Triton and CUDA impls pass all.) Against the current Triton, CUDA is **~1.04–1.08× faster on
the small launch-bound shapes and at parity (1.00×) on the memory-bound shapes** (see note).

> ⚠️ **`old_helion_contig` latency is invalid on every row below and is NOT used as a
> baseline.** The vendored Helion kernel is AOT-autotuned for a single fixed shape
> (`B=1, T=8192`, one sequence — see `_sconv_baseline.py:52`) and that one config is reused
> for all shapes. On these multi-token shapes (all T≥512, B=4) it **fails correctness** (the
> source of the 372 fails) and its runtime is flat at ~55µs regardless of T×D — impossible
> for correct memory-bound work (new_triton correctly scales ~15→214µs over the same 32×
> data range). You cannot compare the speed of a kernel that computes the wrong answer, so
> **ratios are baselined on `new_triton_noncontig`** (the correct production kernel,
> non-contiguous input = what the model sees). The `old_helion*` column is parenthesised to
> mark it invalid.

Harness-generated (`opt1_causal_conv1d.py` on 1× B200, GPU0; ratios computed by `_harness`,
column `cuda_jit_noncontig/new_triton_noncontig`):

```
|   T   |  D   | old_helion* | new_triton_c | new_triton_nc | cuda_jit_nc | cuda / new_triton |
|  512  | 2304 |    (56.3)   |     14.7     |     14.7      |    13.6     |       1.08x       |
|  512  | 4096 |    (55.9)   |     16.0     |     16.0      |    15.2     |       1.06x       |
|  512  | 6144 |    (54.9)   |     16.8     |     16.6      |    16.0     |       1.04x       |
|  512  | 8192 |    (55.8)   |     17.4     |     17.5      |    16.5     |       1.06x       |
|  2048 | 2304 |    (55.6)   |     17.5     |     17.4      |    16.4     |       1.06x       |
|  2048 | 4096 |    (55.5)   |     19.6     |     19.5      |    19.6     |       1.00x       |
|  2048 | 6144 |    (55.7)   |     25.7     |     25.7      |    25.7     |       1.00x       |
|  2048 | 8192 |    (55.4)   |     31.8     |     31.9      |    31.8     |       1.00x       |
|  4096 | 2304 |    (55.2)   |     21.5     |     21.5      |    21.5     |       1.00x       |
|  4096 | 4096 |    (55.7)   |     31.8     |     31.8      |    31.8     |       1.00x       |
|  4096 | 6144 |    (55.3)   |     47.5     |     47.2      |    47.2     |       1.00x       |
|  4096 | 8192 |    (55.3)   |     60.4     |     60.4      |    60.5     |       1.00x       |
|  8192 | 2304 |    (55.7)   |     35.8     |     35.8      |    35.7     |       1.00x       |
|  8192 | 4096 |    (55.5)   |     60.4     |     60.4      |    60.3     |       1.00x       |
|  8192 | 6144 |    (55.4)   |     85.0     |     85.1      |    85.1     |       1.00x       |
|  8192 | 8192 |    (55.5)   |    111.6     |    111.5      |   111.6     |       1.00x       |
| 16384 | 2304 |    (55.7)   |     66.6     |     66.5      |    66.6     |       1.00x       |
| 16384 | 4096 |    (55.2)   |    110.9     |    111.5      |   110.9     |       1.01x       |
| 16384 | 6144 |    (55.1)   |    161.8     |    162.1      |   162.0     |       1.00x       |
| 16384 | 8192 |    (55.7)   |    213.3     |    214.0      |   213.2     |       1.00x       |
```

`cuda / new_triton` = `new_triton_nc / cuda_jit_nc`. Against the **current** Triton kernel,
CUDA is ~1.04–1.08× faster on the small launch-bound shapes (T=512) and within noise (1.00×)
on the memory-bound shapes. `new_triton_c ≈ new_triton_nc` (1.00×) everywhere — the
non-contiguous read is free (gather folded into the kernel). `old_helion*` parenthesised =
invalid (correctness-fail).

> **Supersedes earlier opt1 figures in this file** that showed CUDA ~2–4× at small T. Those
> were measured against a ~2× slower Triton build (small-T `new_triton` ≈29µs then vs ≈15µs
> now); the CUDA kernel itself is unchanged (≈13.6µs). So for opt1 the CUDA-JIT port is now a
> small win (launch-bound shapes) / parity (memory-bound), plus Triton/autotune dependency
> removal — not a large latency win. The opt5/7/9/6 tables below were re-measured against the
> current Triton and show the same pattern: their Triton kernels also got ~1.4–2.5× faster,
> narrowing CUDA's lead from the doc's 1.9–3.8× to ~1.1–1.35× (still faster on every non-conv
> kernel).

---

## opt9 — `update_sconv_cache` (cache shift-update), decode + extend

Correctness: **1152/1152 bit-exact, 0 fails** (includes CUDA-vs-ref).
CUDA vs **current** Triton ≈ **1.28×** (10.1 vs 12.9 µs); vs Helion ≈ **2.75×**. Flat across
shapes. (Supersedes the earlier ~2.07× vs Triton — the Triton kernel is now ~1.6× faster,
12.9 vs 20.7 µs, while CUDA is unchanged at ~10 µs.) Harness-generated (`_harness` ratios
baselined on `old_helion`; both baselines pass correctness so the ratios are valid).

```
|  mode  |  B  |  D   | old_helion | new_triton | cuda_jit | new/old | cuda/old |
| decode |  1  | 2304 |    28.0    |    12.8    |   10.1   |  2.18x  |  2.76x   |
| decode |  4  | 2304 |    28.0    |    12.9    |   10.2   |  2.18x  |  2.75x   |
| decode |  16 | 2304 |    27.6    |    12.8    |   10.2   |  2.16x  |  2.72x   |
| decode |  64 | 2304 |    28.0    |    12.8    |   10.2   |  2.18x  |  2.75x   |
| decode | 256 | 2304 |    28.1    |    12.9    |   10.2   |  2.19x  |  2.75x   |
| decode |  1  | 6144 |    27.8    |    12.9    |   10.2   |  2.15x  |  2.72x   |
| decode |  4  | 6144 |    27.7    |    12.9    |   10.1   |  2.16x  |  2.74x   |
| decode |  16 | 6144 |    28.3    |    12.8    |   10.1   |  2.20x  |  2.80x   |
| decode |  64 | 6144 |    28.1    |    12.9    |   10.1   |  2.19x  |  2.79x   |
| decode | 256 | 6144 |    27.9    |    12.9    |   10.2   |  2.16x  |  2.74x   |
| extend |  1  | 2304 |    27.9    |    12.9    |   10.0   |  2.16x  |  2.77x   |
| extend |  4  | 2304 |    27.6    |    12.9    |   10.1   |  2.15x  |  2.73x   |
| extend |  16 | 2304 |    28.2    |    12.9    |   10.0   |  2.18x  |  2.80x   |
| extend |  64 | 2304 |    28.3    |    12.8    |   10.1   |  2.21x  |  2.79x   |
| extend | 256 | 2304 |    28.2    |    12.9    |   10.2   |  2.19x  |  2.77x   |
| extend |  1  | 6144 |    28.1    |    12.8    |   10.1   |  2.19x  |  2.78x   |
| extend |  4  | 6144 |    28.2    |    12.9    |   10.1   |  2.19x  |  2.80x   |
| extend |  16 | 6144 |    28.2    |    12.9    |   10.1   |  2.19x  |  2.78x   |
| extend |  64 | 6144 |    28.4    |    12.7    |   10.2   |  2.23x  |  2.79x   |
| extend | 256 | 6144 |    28.4    |    13.0    |   10.5   |  2.19x  |  2.71x   |
```

---

## opt7/8 — `fused_causal_conv1d_update_decode` (decode conv + cache update)

Correctness: **2376/2376, 0 fails** (output `y` and updated cache, both vs fp32 ref;
includes CUDA-vs-ref). CUDA vs **current** Triton fused ≈ **1.11×** (16.9 vs 18.8 µs); vs old
fused ≈ 1.6×; vs unfused conv+update ≈ **3.2×**. Baseline for ratios is `unfused_conv+update`.
(Supersedes the earlier ~2.0× vs Triton fused — the fused Triton kernel is now ~1.8× faster,
18.8 vs 34 µs, while CUDA is unchanged at ~17 µs. The big win is still vs the unfused path.)

```
|  T  |  D   | old_fused | new_fused | unfused | cuda_jit | old/unf | new/unf | cuda/unf |
|  1  | 2304 |   27.4    |   18.6    |  54.0   |   16.8   |  1.97x  |  2.90x  |  3.21x   |
|  4  | 2304 |   27.0    |   18.8    |  54.1   |   16.9   |  2.01x  |  2.89x  |  3.21x   |
|  16 | 2304 |   27.3    |   18.7    |  53.4   |   16.9   |  1.96x  |  2.86x  |  3.16x   |
|  64 | 2304 |   26.7    |   19.1    |  54.0   |   16.9   |  2.02x  |  2.82x  |  3.19x   |
| 256 | 2304 |   27.8    |   19.2    |  54.4   |   17.0   |  1.96x  |  2.84x  |  3.20x   |
|  1  | 4096 |   27.1    |   18.8    |  54.5   |   16.7   |  2.01x  |  2.91x  |  3.27x   |
|  4  | 4096 |   28.5    |   18.8    |  54.5   |   17.0   |  1.91x  |  2.90x  |  3.21x   |
|  16 | 4096 |   27.3    |   18.6    |  53.9   |   17.0   |  1.98x  |  2.90x  |  3.18x   |
|  64 | 4096 |   27.5    |   19.0    |  53.6   |   17.1   |  1.95x  |  2.82x  |  3.13x   |
| 256 | 4096 |   27.6    |   19.2    |  54.1   |   17.1   |  1.96x  |  2.82x  |  3.17x   |
|  1  | 6144 |   28.0    |   18.8    |  53.6   |   16.7   |  1.92x  |  2.85x  |  3.21x   |
|  4  | 6144 |   27.6    |   18.8    |  53.7   |   16.8   |  1.95x  |  2.85x  |  3.19x   |
|  16 | 6144 |   27.6    |   18.9    |  53.7   |   16.9   |  1.94x  |  2.84x  |  3.17x   |
|  64 | 6144 |   27.2    |   18.8    |  53.6   |   16.9   |  1.97x  |  2.84x  |  3.17x   |
| 256 | 6144 |   29.1    |   20.6    |  54.7   |   18.8   |  1.88x  |  2.66x  |  2.90x   |
```

---

## opt5 — `fused_gather_scatter_to_sconv_cache` (gather → scatter)

Correctness: **216/216 bit-exact, 0 fails** (includes CUDA-vs-ref; masked-out slots
untouched). CUDA vs **current** Triton ≈ **1.26×** (11.4 vs 14.4 µs); vs old gather+copy
≈ 3.5×. (Supersedes the earlier ~1.9× vs Triton — the Triton kernel is now ~1.35× faster,
14.4 vs 19.5 µs, while CUDA is ~unchanged.)

```
|  B  | W |  D   | old_gather_copy | new_fused | cuda_jit | new/old | cuda/old |
|  8  | 3 | 384  |      39.4       |   14.2    |   11.3   |  2.78x  |  3.48x   |
|  8  | 4 | 2304 |      39.4       |   14.6    |   11.5   |  2.71x  |  3.42x   |
|  8  | 5 | 6144 |      40.5       |   14.5    |   11.4   |  2.79x  |  3.56x   |
|  64 | 4 | 2304 |      39.8       |   14.3    |   11.4   |  2.77x  |  3.48x   |
|  64 | 5 | 6144 |      41.2       |   14.6    |   11.6   |  2.81x  |  3.55x   |
| 256 | 4 | 2304 |      39.7       |   14.4    |   11.3   |  2.75x  |  3.52x   |
| 256 | 4 | 6144 |      39.8       |   14.4    |   11.6   |  2.76x  |  3.45x   |
```

---

## opt6 — `fused_draft_extend_sconv_cache` (speculative draft-extend), with tracking

Correctness: **216/216 bit-exact, 0 fails** (tracking on/off; includes CUDA-vs-ref).
CUDA vs **current** Triton ≈ **1.35×** (11.8 vs 15.9 µs); vs old gather+unfold ≈ **9.8×**.
(Supersedes the earlier ~3.8× vs Triton — the fused Triton kernel is now ~2.5× faster,
15.9 vs 39.9 µs, while CUDA is ~unchanged. The 9.8× is vs the slow eager gather+unfold path.)

```
| B  | T  |  D   | W | old_gather_unfold | new_fused | cuda_jit | new/old | cuda/old |
| 8  | 8  | 2304 | 4 |       116.1       |   15.9    |   11.7   |  7.30x  |   9.91x  |
| 32 | 8  | 2304 | 4 |       114.8       |   16.0    |   11.7   |  7.18x  |   9.83x  |
| 8  | 8  | 6144 | 4 |       115.3       |   15.9    |   11.8   |  7.25x  |   9.79x  |
| 32 | 8  | 6144 | 4 |       114.7       |   16.0    |   11.8   |  7.18x  |   9.74x  |
| 8  | 32 | 2304 | 4 |       114.6       |   15.8    |   11.8   |  7.25x  |   9.68x  |
| 32 | 32 | 6144 | 4 |       114.8       |   15.7    |   11.9   |  7.29x  |   9.62x  |
```
