# Scattered Sconv × Fused All-Reduce: Kernel Design Journal

Branch: `cheng/scattered-sconv-fused-ar` (24 commits on dev `dc11b2b9d9`).
Hardware: 4×GB200-class (SM100), NVLink multicast, TP4, H=6144, W=4, bf16.
All kernel numbers are captured-graph replay µs/site unless noted.

## 1. Goal

Reland scattered sconv (PR #5: output sconvs run on hidden shards, conv cache
sharded `[slots, W-1, H/P]` → **2.28× mamba cache capacity**, 354K vs 243K
tokens at mem 0.85) behind `--enable-scattered-sconv`, and fuse the resulting
`{reduce → sconv(shard) → gather}` chain into the custom all-reduce kernels
(`SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV=1`) so the capacity win costs no
throughput. Target: **beat** the production base chain
`{v3 AR → full-width conv → update_sconv_cache}`, not just the unfused
scattered chain.

## 2. Final architecture

One params struct, three device kernels in
`python/sglang/jit_kernel/csrc/inkling/inkling_ar_scattered_sconv.cuh`, one
dispatch in `python/sglang/tml/kernels/comm.py::ar_scattered_sconv_fused`:

| band | kernel | why |
|---|---|---|
| extend T ≤ 2048 | **chunked column two-shot** (zero-halo mode) | barrier-amortized; smem/stream floors don't pay off |
| extend T ≥ 3072 | **streaming rolling-window** | v3's per-element dataflow + conv in registers; −13…−22% vs base |
| decode / verify T ≤ 204 (+fused add-RMSNorm) | chunked column two-shot + in-kernel norm | accepted 2-sync-round cost (user design decision) |
| verify window save | chunked with `need_scratch=1` | verify externally consumes pre-conv x |

Alternatives kept in-tree, validated, unused by default: **one-shot decode**
(`SsconvNormDecodeKernel`, 14.2µs @T=1 vs 28.6 two-shot — re-enable = one
dispatch line if decode latency ever matters) and **banded-scattered**
(window-shard push + contiguous bands).

### The streaming kernel (the endgame design)

Each thread walks L tokens down one cvec column holding the last W−1 reduced
vectors in registers:

```
warm-up: re-reduce W-1 halo rows (or cache taps at sequence starts)
per token: v = multimem.ld_reduce(x[t, col])
           y = conv(regs, v) [+SiLU][+residual]
           multimem.st(out[t, col], y)
           shift regs
```

No staging, no `__syncthreads`, no phases — full memory-level parallelism,
identical to the plain v3 AR's dataflow with the conv riding in registers.
Halo overhead = (W−1)/L remote re-reads; `stream_walk` exposes L (the derived
min-48 default underfills the grid at mid T — L=16–32 wins there). Phase 3
(rank-shard cache update + track) re-reduces its B×(W−1) rows from the
pristine multicast input after the exit barrier.

## 3. The journey (each step measured, @T=4096 unless noted)

1. **Column two-shot v1** (stage A ld_reduce→scratch, sync, stage B
   conv→broadcast, phase 3 update/track, entry+exit barriers): in-graph
   **204µs** vs base ~170 — and at decode 31.3µs vs base's one-shot 11.7
   → e2e −19% decode, −6…−12% prefill.
2. **Enablement bugs dominated first** (see §4): with them fixed, prefill
   reached parity but the kernel itself still only tied base.
3. **Launch config** ≠ micro-tuned config: the v3-AR-tuned `(nb,bs)` starved
   the chunk pipeline in-graph (T=8192: 171 chunks / 96 CTAs → 629µs vs ~310
   at full residency). Graph-replay tuning became mandatory (§5).
4. **smem weight stage + vectorized taps** (stage B did 32 scalar 2-byte
   weight loads per item): 16K 625→604. Gated T≥8192 (the per-block 12KB copy
   loses where there's little barrier spin to hide under). `alignas(16)` on
   the smem array — bf16 arrays are 2B-aligned and `uint2` reinterpret faults.
5. **Phase 3 moved after the exit barrier**: peers' consumers no longer wait
   on our rank-local cache update; +4% decode e2e.
6. **smem tile** (stage the (halo+tile) in 35KB smem, kill the global
   x_scratch round-trip and its 4× tap-read amplification ≈ 60MB/site):
   16K 604→562–577 — first shape beating base. Gated T≥8192 (forced finer
   cvec split → ragged waves below). 48KB static smem limit forced the
   weight-stage resize (6144 elems).
7. **Streaming rolling-window**: the remaining leak was the stage-A→sync→B
   split serializing what v3 streams per element. Ideal movement cost at 16K
   ≈ 525µs; tile version measured 562–577. Streaming: **512** first try,
   **488.5** walk-tuned. Then the walk-length sweep took 4096 too (155.3
   with L=24 vs 171.8 chunked).

## 4. False trails and diagnostic lessons (chronological)

- **Zombie servers**: `pkill -f launch_server` does NOT kill
  `sglang::scheduler_TP*` children. A killed gate server kept answering
  :30000 → an entire "base @0.80" benchmark column was actually the fused
  config; a later co-tenant race produced identical mem-fraction red
  herrings. Kill schedulers explicitly; wait for GPUs <2GB.
- **"dev is 20–40% faster"** — same tree at the right mem fraction matched
  dev exactly. The gap was the squeezed-KV environment, not code.
- **"grouped-GEMM is slower in this branch"** — flashinfer autotune picks
  different tactics per *node* (kernel-name strings differ di1 vs di2);
  same-node base-vs-fused bmm ≈ equal.
- **Decode fusion silently disabled**: capture batches always carry a
  persistent (all-False) `mamba_track_mask` buffer — any `is not None` gate
  bakes the unfused fallback into every decode graph. Tracking must be
  data-dependent in-kernel.
- **Extend fusion silently disabled**: the FULL prefill CG runner wraps
  captures in the same `set_tc_piecewise_forward_context` as BCG; the gate's
  BCG exclusion killed fusion in all full prefill graphs. Fixed via
  `TcPiecewiseForwardContext.full_graph` keyed on `_is_full_backend`
  (NOT `layer_model is None` — both backends set `layer_model`).
- **OUT region sizing**: 8192×6144 while `max_prefill_tokens=16384` → all
  multi-request prefills failed eligibility → unfused. 512MiB symm buffer +
  16384×6144 region.
- **Eager micro ≠ in-graph**: three separate times (204-vs-169, the 8192
  config cliff, the base-anchor mismatch). Every config decision must be
  graph-replay measured; NCCL barriers abort inside capture loops — align
  ranks with a **gloo** group.
- **Strided-column multimem was NOT the villain** (initial hypothesis): the
  banded variant (v3-contiguous slices + window push) tied, not beat, the
  column kernel; per-row 3KB strips coalesce fine. The real leaks were
  enablement, launch config, and local-memory round-trips.
- **The theory check that unlocked the endgame**: base chain spends
  ~100MB/site on intermediate HBM round-trips at 4096 that fusion should
  save; when the kernel merely tied base, that arithmetic said "implementation
  leak" — scratch traffic (60MB) and stage serialization — not "design limit".
- **Extend norm-tail fusion is a measured pessimization** (2026-07-12): the
  add+RMSNorm tail was added to the streaming kernel (validated bit-exact,
  chunked+stream, T=301..8192) and the extend call sites were wired to pass
  `norm` — then graph-replay (`--tune-norm-ext`, TP4) showed the unfused
  chain {fused no-norm + sgl_kernel fused_add_rmsnorm} wins at EVERY extend
  shape: 46.8-vs-48.9µs @512 up to 607.9-vs-680.6 @16384 (best swept fused
  config, 1-18% worse). Structural, not tuning: the tail moves the same
  ~4·T·H bytes as the standalone kernel but under the AR grid's barrier
  co-residency cap (~148 blocks) and conv-shaped block sizes, while fusion
  only saves one ~3µs launch (and multimem-stored OUT rows are not L2-hot,
  so there's no reuse win either). Decode/verify keep the fused tail (tiny
  T, launch-dominated, L2-hot — where it measurably wins). The stream-kernel
  tail + `--tune-norm-ext` + norm-ext validation cases stay in-tree; the
  call sites stay unfused at extend.

## 5. Tuning methodology (harness: `benchmark/tml/allreduce/validate_ar_scattered_sconv.py`)

- Correctness: bit-exact vs the unfused full-width reference (channelwise
  conv ⇒ full-width == concat of shard convs). 65+ cases: multi-seq
  boundaries inside tiles, fresh/PAD slots, decode shapes, track (gather and
  from-cache), both barrier scopes, interop with plain v3 ARs, CUDA-graph
  capture+replay. `RESULT: PASS` required before any perf number counts.
- `--tune`: per-shape 56-config × 2-barrier graph-replay sweep, extend
  T=128…16384 (chunked-prefill closes the domain), decode T=1…204 @16-token
  steps. gloo barriers around capture.
- `--tune-bs`: B∈{1…256} at fixed T — flat (±0.6%); T is the only dispatch
  dimension.
- `--sweep-stream`: stream vs chunked, 12 configs × 5 walk lengths per shape.
- `--sweep-base`: the base chain in the same harness (v3 best-of-5 configs +
  conv + update) — measured-vs-measured, no anchors/interpolation.

## 6. Final numbers

### Kernel (graph-replay, same harness, production `need_scratch=False`)

| T | ours (dispatched) | base | Δ |
|---|---|---|---|
| 128 | 26.8 c | 29.2 | −8% |
| 256 | 30.8 c | 33.8 | −9% |
| 512 | 41.7 c | 42.2 | −1% |
| 1024 | 57.5 c | 60.4 | −5% |
| 2048 | 94.2 c | 97.7 | −4% |
| 3072 | 127.0 s | 140.0 | −9% |
| 4096 | 155.3 s | 178.3 | −13% |
| 6144 | 210.8 s | 252.9 | −17% |
| 8192 | 266.6 s | 331.6 | −20% |
| 10240 | 316.8 s | 404.9 | −22% |
| 12288 | 375.6 s | 479.3 | −22% |
| 14336 | 429.5 s | 549.1 | −22% |
| 16384 | 488.5 s | 623.5 | −22% |

(c = chunked, s = streaming.) vs the unfused scattered chain: 1.5–2.0×.
Decode band: 28.6–36µs (two-shot+norm) vs base one-shot 11.7 — accepted.

### e2e (di1, TP4, mem 0.85, full-CG prefill, vs base)

| case | in-tps Δ | out-tps |
|---|---|---|
| bs1-4096 | +4.0% | 101.1 |
| bs1-8192 | +6.3% | 100.8 |
| bs4-4096/8192 | +3.0/+3.5% | 358.9/355.0 |
| bs16-4096/8192 | +3.6/+2.2% | 1046.2/1025.2 |
| bs64-4096 | 44.4K (base cannot run) | 2728.8 |

gsm8k throughout: P64 0.925–0.930, P1 0.850–0.900 band, 0% invalid.

## 7. Remaining catalogued work

- Wave-quantized / atomic-ticket chunk scheduling for the chunked kernel
  (review P1a) — mostly moot now that streaming owns T≥3072.
- One-shot decode dispatch if the −11…−21% decode trade is ever revisited
  (kernel in-tree, 14.2µs @T=1, needs window staging in the OUT region).
- Host-validation hardening (review P5: `block_size % 32` silently corrupts
  the norm reduction; stride/dtype checks).
- Streaming kernel: `griddepcontrol` (PDL) port; L auto-tuning per SM count.
- The `--tune` decode-band numbers predate the phase-3 reorder; a re-sweep
  may shave 1–2µs of config drift.
- **Done**: cleanup pass (stale docstrings P11, journal-style µs comments,
  the redundant `scattered_ar_sconv_fusable` capacity check, and the
  never-consumed `weight_full` one-shot-decode weight retention — that
  path takes its full-width weight as a caller-supplied argument instead,
  exercised only by the validation harness).

## 8. File map

- `python/sglang/jit_kernel/csrc/inkling/inkling_ar_scattered_sconv.cuh` — all
  kernels (chunked column, streaming, banded, one-shot decode) + hosts.
- `python/sglang/jit_kernel/inkling_ar_scattered_sconv.py` — JIT wrappers.
- `python/sglang/tml/kernels/comm.py` — gates (`scattered_ar_sconv_fusable`),
  dispatch + tuned tables, symm-region layout (`ssconv_out`).
- `python/sglang/tml/layers/sconv.py` — metadata providers
  (`extend_fused_ar_inputs`, decode metadata).
- `python/sglang/srt/models/inkling.py` — the three fusion call sites.
- `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`,
  `.../tc_piecewise_cuda_graph/context_manager.py` — `full_graph` marker.
- `python/sglang/srt/distributed/device_communicators/torch_symm_mem.py` —
  512MiB buffer under scattered.
- `benchmark/tml/allreduce/validate_ar_scattered_sconv.py` — validation +
  bench + tune harness.

## 9. Full-width (non-scattered) extend mode (2026-07-12)

The same column kernels now serve the NON-scattered path at extend
(`ar_fullwidth_sconv_fused`, gated by `fullwidth_ar_sconv_fusable`:
`SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV` without `--enable-scattered-sconv`),
replacing {one-shot AR + replicated full-width causal_conv1d +
update_sconv_cache}. Two kernel deltas, no new kernel:

- `cache_col0` (= rank*Hc): conv taps read this rank's column slice of the
  REPLICATED [slots, W-1, H] cache (the layout already puts W-1 before the
  channel dim, so the slice is just a column offset + the existing strides).
  The weight is this rank's contiguous [Hc, W] row slice of the full [H, W].
- `full_update`: phase 3 spans ALL H columns on every rank (window rows
  re-ld_reduced full-width from the multicast input; B*(W-1) rows,
  negligible), keeping the replicated cache coherent for the full-width
  decode/verify consumers. Same for full-width track rows.

The conv itself stays column-sharded: 1/P of the FLOPs of the replicated
full-width conv, and the pre-conv x round trip disappears. Verify
(`need_scratch`) is not supported full-width -- decode/verify keep the v5
`ar_sconv_norm_fused` (unchanged).

Measured (graph-replay `--tune-fw`, GB200): fused loses below T=3072
(+5..+14% -- the fixed two-shot machinery outweighs the savings) and wins
above with the STREAMING kernel at every shape:

    T       TP4 base   TP4 fused   TP8 base   TP8 fused
    3072      140.5      127.9       138.2      125.8
    4096      178.1      157.2       173.9      153.8
    6144      252.7      211.4       246.1      204.8
    8192      331.5      267.3       319.5      258.5
    12288     479.4      375.7       462.8      364.6
    16384     624.8      488.3       605.0      465.5   (us; -8..-26%)

`_INKLING_AR_FW_MIN_TOKENS = 3072` is part of the producer/consumer
contract; both TP4 and TP8 dispatch tables are measured (not extrapolated).
Correctness: `fw-*` cases (chunked+stream, decode-shaped/fresh/pad/multi-seq/
track, full replicated-cache equality on EVERY rank) pass at TP4 and TP8.
The 512 MiB symm buffer bump now also applies under the fused-AR-sconv env
flag without scattered (the OUT region needs the same headroom).
