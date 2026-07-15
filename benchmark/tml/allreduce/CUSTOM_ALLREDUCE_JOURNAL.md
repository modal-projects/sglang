# Inkling Custom All-Reduce — Design & Autotuning Journal

A self-contained, autotuned all-reduce for the Inkling (Moonrise) tensor-parallel
reductions, living at `python/sglang/jit_kernel/csrc/inkling/inkling_all_reduce.cuh` +
`python/sglang/jit_kernel/inkling_all_reduce.py`. It matches or beats torch
`multimem_all_reduce_` across the full shape range on B200 / TP4 / bf16, is
CUDA-graph safe, and exposes an epilogue seam so RMSNorm / short-conv can later
fuse into the reduce.

All numbers below are B200, TP=4, bf16, hidden=6144, di2 GPUs 4-7. min-of-N to
filter contention noise from the box's co-located job.

---

## 1. Motivation

A profiler trace of the running Inkling server (TP=4) showed all-reduce is a
first-order cost, not a rounding error:

| phase | AR share of GPU-kernel time | kernel |
|---|---|---|
| decode (bsz=1) | ~15% | torch symm-mem `multimem_all_reduce_` (one-shot) |
| prefill il=4096 | ~47% | `multimem_all_reduce_` |
| prefill il=8192 | ~58% | **NCCL `AllReduce_RING_LL`** |
| prefill il=16384 | ~22% | NCCL ring |

Two findings drove the work:
1. AR is a big fraction of both decode and prefill time.
2. **Large prefill silently falls off the multimem fast path onto NCCL ring.**
   The torch symm-mem eligibility cap is 64 MiB (ws4); `[N, 6144]` bf16 exceeds
   it at N≈5300 tokens (il=8192 → 101 MiB, il=16384 → 201 MiB), so those reduce
   via NCCL ring LL — the latency protocol, not bandwidth-optimal. That fallback
   is the single biggest, easiest target.

## 2. Substrate: torch symmetric memory (not the legacy custom-AR IPC path)

Inkling already enables `--enable-torch-symm-mem`; `TorchSymmMemCommunicator` owns a
rendezvous'd symmetric `comm.buffer` and the producer (wo_ud / MoE-combine GEMM)
writes its shard straight into it via `get_ar_buffer` (the existing zero-copy /
copy-elimination fusion). We build the custom kernels **on that same buffer**:

- `hdl = torch_symm_mem.rendezvous(comm.buffer, group_name)` (idempotent) exposes
  `hdl.buffer_ptrs_dev` (device array of N peer buffer bases), `hdl.multicast_ptr`
  (NVLink multicast VA), `hdl.rank`, `hdl.world_size`.
- This **preserves the `get_ar_buffer` zero-copy fusion for free** — the kernel
  reduces over peer *views of the same buffer* the producer already wrote into.
- We do **not** reuse the heavyweight `CustomAllReduceBase` IPC path (which stages
  input via `cudaMemcpyAsync`, defeating copy-elimination) nor its per-block
  `PullController`. Barrier state (a dedicated symm `uint32[world_size]` flags
  buffer + a device-local `uint32[8]` state buffer) is ours.

## 3. Kernel family

All operate on the symm buffer in units of 16 B (`bf16x8` = `AlignedVector<bf16x2,4>`).

| kernel | algorithm | in/out | barriers | best for |
|---|---|---|---|---|
| **v1** | two-shot explicit, torch `hdl.barrier()` on each side | in-place | 2 (host) | reference only (3 launches) |
| **v2** | two-shot explicit: read N peer slices, fp32 accum, broadcast | in-place | 2 (in-kernel) | token=64 |
| **v3** | two-shot multimem: `multimem.ld_reduce` scatter + `multimem.st` broadcast | in-place | 2 (in-kernel) | **≥96 tokens (up to 2.14×)** |
| **v4** | full one-shot: every rank `ld_reduce`s the *entire* range → local out | out-of-place | **1** (in-kernel) | **tokens 1-2 (1.36×)** |

**Two-shot is race-safe in place.** Rank r owns a disjoint, warp-aligned vec slice
`S_r`; it reads every peer's `S_r`, sums, and broadcasts back to every peer's
`S_r`. Only rank r ever writes `S_r` (in any buffer), and each per-element load
completes before its store — no write-write conflict, no read-after-partial-write.

**Full one-shot (v4) trades reads for a barrier.** Each rank `ld_reduce`s the whole
range (N× the read traffic) and writes the full result to a *separate local*
output — so there is no broadcast and the exit barrier can be dropped (the result
is complete on-rank). This only pays off for tiny, latency-bound messages; the N×
read makes it lose by token=4. Its input-reuse safety is the caller's (double
buffer; the next AR's entry barrier proves all ranks finished reading the old one)
— validated by a 2000-iter rotation stress.

The reduced `Storage` (before the broadcast/local store) is the **EPILOGUE SEAM**
where RMSNorm / short-conv / bias will fuse (mirrors `tp_qknorm.cuh`). v4 is the
natural fusion base: each rank holds the full row, so a norm-over-hidden reduction
fuses cleanly (two-shot only holds a scatter slice mid-reduce).

## 4. The barrier — where all the performance was

The cross-GPU barrier went through three revisions; the last was the key unlock.

1. **Per-block `threadfence_system` (first correct version).** Every block did a
   full system-scope memory flush + a cross-GPU signal/wait. Correct, but the
   cost scales with `num_blocks`: token=64 (48 blocks) hit 40 µs, and it lost to
   multimem everywhere it mattered.
2. **Grid-level, single-leader.** All blocks arrive at a self-resetting device
   counter (`atomicInc(arrival, gridDim.x-1)` wraps to 0 each barrier); the
   last arriver is the leader and is the *only* block that does the cross-GPU
   sync — so that O(1) cost is independent of `num_blocks`. Followers spin on a
   device-scope release counter. **Requires all blocks co-resident** — a manual
   grid barrier deadlocks otherwise (token=4096 with 768 blocks hung), so the
   grid is capped at `max_resident_blocks = get_blocks_per_sm(kernel) × SM count`.
   A `gridDim==1` fast path skips the arrival/release bookkeeping entirely.
3. **Release/acquire instead of `threadfence_system` (the win).** The leader's two
   `threadfence_system` flushes (~4 per AR) were ~6 µs of the token=1 latency.
   NVLink symmetric memory is coherent, so the data doesn't need a full flush —
   only the *timing* handshake does. Switching the flag store/wait to
   `st.release.sys` / `ld.acquire.sys` (device-scope acquire for the local
   release counter) recovered all ~6 µs: **token=1 went 19.5 → 13.3 µs**, and with
   the `gridDim==1` fast path, 13.0.

**Graph safety.** The cross-GPU epoch is a device-resident monotonic counter
(`xepoch`), incremented per barrier; under CUDA-graph replay it keeps advancing,
so flags never go stale (spin is `flag < epoch`). Entry/exit use separate
arrival/release slots so the two barriers in one kernel don't collide. Validated
by 50 graph replays + 2000-iter back-to-back stress with zero wedge/corruption.

## 5. Config autotuning

Two launch knobs are tunable (`num_blocks`, `block_size`); a fine-grained sweep
(28 shapes × per-kernel `num_blocks` × `block_size ∈ {256,512,1024}`, min-of-2)
found:

- **`num_blocks`**: the multimem kernel (v3) is bandwidth-efficient and saturates
  NVLink with far fewer blocks than the `max_resident` (~148) auto value; ~32-96
  blocks is optimal, worth up to **1.22×** (1024 tokens: 50.6 → 41.5 µs). Extra
  blocks just add grid-barrier atomic contention. v2 (explicit) wants many blocks.
- **`block_size`**: matters at the mid boundary — bs=256 @ 64-128, bs=512 @ 256,
  **bs=1024 @ ≥512**.
- Both optima sit on a **broad plateau** (nb 32-96, bs 512/1024 within ~2%), so
  the baked table is robust to noise.

## 6. Auto-select wrapper

`select_ar_config(num_tokens, world_size)` returns `(kernel, num_blocks,
block_size)` from the autotuned table (round-up to the nearest tested shape).
Boundaries below are the current sweep — on the barrier as it stands (exit-publish
`__threadfence_system` + `acq_rel` arrival / `red.release` handoff) with the
cached occupancy query. Crossover boundaries (TP4):

| token band | winner | vs multimem |
|---|---|---|
| 1-2 | **v4** | 1.08-1.30× |
| 3-192 | **multimem** | 1.00× (hardware floor) |
| 256-16384 | **v3 (tuned)** | 1.05× → **2.12×** |

Crossover boundaries (TP8, B200 — full sweep on a rebooted/healthy node):

| token band | winner | note |
|---|---|---|
| 1-2 | **v4** | 10.6 / 12.1 µs (1.31× / 1.08×) |
| 3-768 | **multimem** | small/medium hardware floor (13-36 µs) |
| 1024-16384 | **v3 (tuned)** | dominates; e.g. 16384 = 435 µs vs mm 530 (1.22×) |

`v2` never wins (neither TP4 nor TP8 post-retune), and the v3-over-multimem margin
is thinner at TP8 (~1.2× at 16384 vs up to 2.1× at TP4) with a higher crossover
(1024 vs 256) — more ranks halve v3's per-GPU slice, so it needs more rows to
amortize the barrier past mm's floor. v4 still owns the latency-bound decode band
(bsz≤2) that matters most, at both world sizes.

The kernel family + auto-select dispatch are validated by
`validate_inkling_all_reduce.py` (`ALL_OK`): token-grid correctness, uint32 epoch
wraparound, v4 A/B rotation, CUDA-graph capture+replay, and a tuned-config bench,
at TP4 and TP8.

## 7. Results & reproducibility

Best kernel per shape vs torch multimem (µs), and run-to-run stability across two
independent full sweeps:

| tokens | multimem | best (kernel) | speedup | run-to-run Δ |
|---:|---:|---:|---:|---:|
| 1 | 12.6 | 9.7 (v4) | 1.36× | (v4 ±6%) |
| 64 | 17.0 | 16.5 (v2) | 1.03× | <1% |
| 256 | 27.5 | 21.1 (v3) | 1.30× | <2% |
| 1024 | 66.8 | 41.7 (v3) | 1.60× | <0.1% |
| 4096 | 228 | 125 (v3) | 1.83× | <0.1% |
| 16384 | 963 | 452 (v3) | 2.14× | <0.05% |

**Reproducibility**: two full 28-shape sweeps agreed on **100% of kernel choices**
(0 disagreements) with latencies within **±1.7% (mostly <1%)**. The earlier
apparent "v3 regression" was purely the box's co-located-job contention — v3
reproduces to <0.2% at large shapes across independent runs.

## 8. Correctness summary

- **Two-shot in-place**: disjoint per-rank slices → no write-write conflict; the
  entry/exit barriers fence producer→AR and AR→consumer. The exit barrier has each
  CTA `__threadfence_system()` before signalling arrival, so the single leader's
  cross-GPU release publishes *all* blocks' stores, not just the leader's (see §11).
- **v4 single-barrier**: result is local + complete; input reuse safe under 2-buffer
  rotation (next entry barrier orders it) — 2000-iter rotation stress passed.
- **Graph-safe**: device epoch advances under replay; 50 replays × correct.
- **Numerics**: fp32 accumulation in v2; hardware `multimem.add` (bf16) in v3/v4,
  matching torch multimem. All variants reduce to the exact expected sum
  (`bad=0`) at every tested size.

## 9. Open items

- **TP=8 tuning** — DONE. `select_ar_config` now dispatches per `world_size`;
  `_AR_TUNED_TP8` holds the B200/TP8 sweep (see §6). Untuned sizes fall back to TP4.
- **Fusion (deferred)** — fold RMSNorm / short-conv into the epilogue seam (v4 as
  the base) to remove a separate launch + HBM round-trip per layer.
- **Autotune generality** — the config table is B200/TP4-specific; make it
  per-(device, world_size) via a small first-use autotune + cache.

## 10. Model integration (`SGLANG_OPT_USE_INKLING_CUSTOM_AR`) & e2e results

The auto-select wrapper is wired into `tml/kernels/comm.py`'s `symm_mem_all_reduce`
behind the opt-in `SGLANG_OPT_USE_INKLING_CUSTOM_AR` flag:

- **Enlarged buffer**: the torch symm-mem communicator buffer is bumped to 256 MiB
  at init (flag-gated, in `torch_symm_mem.py`) so large prefill ARs
  (`[16384,6144]` = 192 MiB) fit and dispatch v3 instead of falling off the 64 MiB
  cap to NCCL. It must be allocated at init (a normal, non-inference tensor) —
  a lazily-allocated buffer becomes an "inference tensor" and the producer GEMM's
  `out=` fails under BCG capture.
- **Dispatch**: `select_ar_config(num_tokens)` → v3 (medium/large, in place), v2
  (token=64), multimem ("mm" bucket), or **v4** for num_tokens<=2.
- **v4 decode path**: out-of-place with the exit barrier dropped, so it uses two
  rotating input regions + one output region carved from the buffer tail; a region
  is reused only two ARs later, separated by the intervening AR's entry barrier —
  capture-safe (the A/B alternation bakes into the decode graph). The resource
  build is guarded by `is_current_stream_capturing()` and populated on the first
  eager call before capture.

**Correctness**: gsm8k **0.900–0.909** (BCG-on, TP=4) with v3 (prefill) + v4
(decode) live; no hang, no InferenceMode/OOM errors, clean BCG capture.

**e2e (bsz=1, out=128), custom-AR ON vs OFF:**

| il | latency Δ | TTFT Δ (v3 prefill) | output tput Δ (v4 decode) |
|---:|---:|---:|---:|
| 4096 | **−4.5%** | **−11.9%** | **+3.6%** |
| 8192 | −3.5% | −4.0% | +3.6% |
| 16384 | −2.9% | −0.8% | +3.6% |

Trace-verified (bs=1/il=4096, TP0): prefill EXTEND = `inkling_multimem_one_shot_fused`
(v3, 132 ARs); decode DECODE = `inkling_multimem_full_oneshot` (v4, 660 ARs, down from
660 multimem — decode AR time 7.84 → 5.80 ms, −26%). The TTFT win is largest at
small il (AR is a bigger fraction of a short prefill); the +3.6% output-throughput
win is il-independent (decode-side v4).

## 11. Review fixes (Codex on PR #53): exit-barrier publish + guards + retune

Three findings, all fixed and re-validated:

- **P1 — exit-barrier system visibility (correctness).** The barrier's cross-GPU
  release is done by the *single leader block* (the O(1) win in §4). A
  `st.release.sys` only orders the **leader thread's own** prior writes; the other
  CTAs signalled arrival with a relaxed atomic, so on a multi-block launch (the
  tuned v2/v3 configs) a peer could leave the *exit* barrier and read a slice a
  non-leader CTA wrote but never system-published. Held up empirically (NVLink
  timing masked it — the relaxed data stores land before the flag round-trip), but
  fragile. **Fix:** each CTA runs `__threadfence_system()` before signalling *exit*
  arrival (`publish_writes=true`), so all blocks' stores are system-visible before
  the leader's release; the arrival counter's device ordering then guarantees every
  fence completed before the release. Entry barriers keep `publish_writes=false`
  (the data they gate on was written by a prior kernel, already uniformly visible),
  and v4 (no exit barrier) is untouched — the decode hot path pays nothing.
- **P1 — TP=6 power-of-two guard (crash).** `InklingAllReduceTrait` static_asserts
  `std::has_single_bit(kNumGPU)`, but TP=6 is torch-multimem-eligible and would hit
  that assertion at JIT compile. **Fix:** `_get_inkling_ar_resources` returns `None` for
  non-power-of-two worlds → they stay on the plain-multimem fallback.
- **P2 — non-vector size fallback.** `validate()` requires `num_items % 8 == 0`
  (16 B vector) while torch symm-mem only enforces 4 B. **Fix:** gate the custom
  dispatch on `n % 8 == 0`, else fall back to torch multimem. (Never bites Inkling —
  `hidden=6144` is a multiple of 8 — but the utility is general.)

**Retune.** The exit fence shifts the mm↔v3 crossover, so both tables were
re-swept. (§12 later made the fence per-CTA-cheaper and changed the arrival/
release atomics, which prompted a final re-sweep — the committed tables reflect
that latest barrier, not this interim one; see §12 for the current crossovers.)
Large-shape v3 wins hold and v4 (1–2 rows) is unchanged throughout.

**Re-validation (post-fence, di2 B200):** standalone TP=4 — v2/v3/v4 correct at
1…16384 rows, 2000-iter v4 rotation, 1000-iter v2/v3 stress (incl. multi-block
4096), 50 CUDA-graph replays, all `bad=0` / `ok=True`. e2e **gsm8k 0.901**
(Invalid 0.000), in the pre-fix 0.900–0.909 band.

## 12. Second review pass: barrier hardening + cheaper exit publish

A second correctness review (Claude, Jul 2026) found and fixed four issues in
the barrier, and made the exit publish ~32x cheaper. Validation script now
committed as `benchmark/tml/allreduce/validate_inkling_all_reduce.py`.

- **P1 (hang/corruption after days of uptime) — uint32 counter wraparound.**
  The spins were `got < e` / `got <= s_prev`. After 2^32 barriers (~4-7 days of
  continuous decode at 132 ARs/step): (a) `xepoch` wraps to 0 and a handful of
  cross-GPU barriers pass WITHOUT waiting (`got(≈2^32-1) < e(0)` is false) — a
  silent cross-rank data race; (b) the grid release counter wraps and followers
  spin on `got(0) <= s_prev(2^32-1)` FOREVER — a hard deadlock on any
  multi-block launch. Fixed with wrap-safe signed-distance compares
  (`(int32_t)(got - e) < 0`); barrier skew is bounded (≤ a couple of epochs),
  far below the 2^31 window. **Empirically confirmed**: preloading
  xepoch/flags/releases to 2^32-8 (phase [2] of the validation script)
  deadlocks the pre-fix kernel (all 4 GPUs spin at 100% until killed) and
  passes with the fix.
- **P2 (formal memory-model gaps, timing-masked on B200).** (a) The leader's
  grid-release bump was a *relaxed* `atomicAdd`, so the followers' paired
  `ld.acquire.gpu` formally synchronized with nothing — the leader's acquired
  peer state (and its plain `xepoch` store, read by the *next* barrier's leader,
  possibly a different block) was unordered. Now `red.release.gpu`. (b) The
  arrival `atomicInc` was relaxed, so the leader formally never *acquired* the
  non-leader CTAs' publish fences before releasing to peers. Now
  `atom.acq_rel.gpu.inc`. Both are single-instruction changes on a per-CTA/
  per-grid (not per-thread) path — no measurable cost.
- **P2 (exit publish cost) — one fence per CTA instead of per THREAD.** The
  Codex-fix `__threadfence_system()` ran in every thread (256-1024 fences per
  CTA). The `__syncthreads` already orders all CTA stores before thread 0, and
  `fence.sys + arrive` is a release pattern, so thread 0 alone fencing before
  its arrival publishes the whole CTA (the standard TRT-LLM block-barrier
  shape). The solo (gridDim=1) path needs no fence at all (its `st.release.sys`
  signals are release ops). A/B (di2 B200 TP4, same box back-to-back, us):
  v3@64 23.1→19.8 (-14%), v3@192 25.8→23.5 (-9%, now BEATS mm's 24.4),
  v3@256 27.8→25.7 (-7.5%), v3@1024 47.3→45.1, v3@4096 129.4→127.9,
  v3@16384 ~unchanged, v4@1 ~unchanged. Pure win; the acq_rel/release atomic
  strengthening costs nothing measurable. **Re-swept on this barrier** (full
  per-config sweep, TP4 + TP8): v3's decisive win starts at **256 rows (TP4)**
  and **1024 (TP8, down from 1536** — the cheaper fence + cached occupancy pull
  the TP8 crossover in); the 3-192 (TP4) / 3-768 (TP8) bands stay on mm (its
  hardware floor — the sub-2% ties there aren't worth a custom dispatch); the
  large-shape wins hold (TP4 up to 2.12×, TP8 up to 1.22×) and v2 never wins at
  either world size. Re-validated e2e: **gsm8k 0.904** (Invalid 0.000, TP4, BCG
  on, custom-AR on) with the current barrier + msgspec `comm.py` + these tables.
- **P3 (latent) — full one-shot auto grid under-provisioned.** `nb_override=0`
  sized the grid from the per-rank two-shot slice (`work_num_blocks`) but v4
  grid-strides the ENTIRE range → kNumGPU x too few blocks. Never bitten (the
  tuned table always passes nb=1 for v4's 1-2-token band); fixed via
  `full_range_num_blocks`.
- **Hardening.** `validate()` now checks `buf.numel() >= num_items` (and out);
  multimem kernels `static_assert` bf16 (their PTX hardcodes `.bf16x2`);
  occupancy/SM-count queries cached per (kernel, block_size, device) — they
  cost a few us on every eager launch; setup does a device-side
  `hflags.barrier()` after `flags.zero_()` so a fast peer's first epoch store
  can't race a slow rank's pending zero (the protocol self-heals from that
  clobber, but don't rely on it); `_InklingArResources` is a typed msgspec.Struct.
- **Documented invariant (unfixed by design):** v4's A/B double-buffer safety
  requires the v4 AR sequence to alternate globally — guaranteed today by the
  even per-forward v4 count (attn+MLP per layer), but an odd count (e.g. a
  layer reduce-scattering at decode) would let a graph replay reuse a region at
  distance 1. Documented at `_INKLING_AR_V4_REGION`; audit when changing decode-AR
  layering.

## 13. v5 push one-shot + per-block barrier: taking the small/medium band from mm

The 3-192 (TP4) band had stayed on torch multimem — a two-shot with two
barriers, like v3, so v3 could only ever tie it. Two new pieces (Claude, Jul
2026) take most of that band, plus the 256-1024 mid band:

**v5 — one-shot PUSH (multicast store + local reduce, ONE barrier).** Each rank
`multimem.st`'s its full input into its per-rank slot of a symmetric staging
area (the NVSwitch replicates it to every GPU; n egress, (N-1)·n ingress), one
barrier waits for all pushes to land, then each rank reduces the N staged
shards LOCALLY (fp32 accum) into a local output. No entry barrier (a rank
pushes only its own producer's data, stream-ordered locally), no exit barrier
(output is local) — the two-shot's second round trip is gone. Staging is A/B
double-buffered exactly like v4's input (same SAFETY INVARIANT). Unlike v4's
full-range `ld_reduce` (whose switch-side reduce does N redundant full-range
sums — why v4 died at 4 rows), replication is cheap for the switch, so v5
scales until the (N-1)·n ingress catches the two-shot's n/N-per-rank traffic.
Like v4, each rank holds the FULL row at the epilogue seam — the RMSNorm-fusion
base now covers the whole latency band, not just 1-2 rows.

**Per-block barrier (`block_system_barrier`).** The single-leader grid barrier
funnels every block through arrival/release atomics + one leader's handshake —
that funnel is what kept multi-block launches above mm's floor. The per-block
flavor has block b handshake ONLY with block b on each peer (per-(writer,block)
flag slots after the leader slots; per-block device-local epochs; wrap-safe
compares): one NVLink round trip per block, all blocks in parallel, no funnel.
Correct for v5 because the push and reduce loops use the same grid-stride
mapping (block b consumes exactly what the peers' block b pushed); correct for
v3 ("v3b") because any peer block's entry signal proves that peer's producer
kernel completed, and kernel end is a grid-wide join, so per-block exit waits
compose into "all peers' broadcasts done". The release-store signal makes the
explicit publish fence unnecessary.

**TP4 results (B200, min-of-5x20, hidden=6144), best-config vs torch mm:**

| rows | mm | v5-pb | speedup | | rows | mm | best | winner |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 12.7 | 8.8 | **1.44×** | | 128 | 20.2 | 21.9 | mm keeps |
| 4 | 13.2 | 10.3 | **1.27×** | | 192 | 24.4 | 24.5 | mm keeps (tie) |
| 8 | 15.5 | 11.4 | **1.37×** | | 256 | 27.8 | 26.2 | **v3b** |
| 16 | 15.4 | 12.3 | **1.25×** | | 384 | 34.2 | 29.2 | **v3b** 1.17× |
| 32 | 15.7 | 13.9 | **1.13×** | | 512 | 40.7 | 32.1 | **v3b** 1.27× |
| 64 | 17.7 | 17.1 | **1.03×** | | 1024 | 67.6 | 45.6 | **v3b** 1.48× |
| 96 | 20.1 | 19.5 | **1.03×** | | ≥1536 | | | v3 (unchanged) |

v5 also supersedes v4 at 1-2 rows (8.8 vs 10.3 µs — 1.44× vs mm, the best
token=1 number yet); v4 stays selectable (and in the TP8 table). mm keeps only
128-192, where v5's ingress and v3's second barrier both just miss (within 8%).

**Dispatch/integration.** Table kernels now: v5 (1-96), mm (128-192), v3b
(256-1024), v3 (≥1536). `comm.py` carves two v5 staging buffers
(world × 96·6144 elems) + an out region from the buffer tail; the v5 branch
reads `input` directly (LOCAL read — no stage-in copy even off the ar_buffer
path). Barrier resources grew: flags = world×(1+64) u32 slots, state = 8+64
words (per-block epochs advance under graph replay like xepoch; the wrap
preload in the validation script covers them).

**Validation** (`validate_inkling_all_reduce.py` phases [6],[7] + extended [2],[5];
`validate_comm_integration.py`): v5 leader+pb correct across 3-1024 incl. odd
shapes, 200-iter rotation interleaving barrier flavors, v3b correct at 3-4096
incl. auto-nb capping, CUDA-graph capture/replay with mixed flavors, uint32
wrap crossing with per-block epochs preloaded, full comm.py dispatch mix — all
pass at TP4; TP8 kernel correctness validated (table entries for TP8 pending
the TP8 sweep, see TODO in `_AR_TUNED_TP8`).

**Complete five-way matrix** (`bench_ar_variants.py`, B200 TP4, us, min-of-5x20,
best config per variant; run with a neighbor capture workload on the other 4
GPUs -- absolutes ~0.5-1us above the idle-box runs, orderings identical):

| rows | mm | v3 | v3b | v4 | v5 | best | vs mm |
|---:|---:|---:|---:|---:|---:|:--|---:|
| 1 | 13.6 | 21.7 | 18.4 | 10.3 | **9.4** | v5 | 1.45x |
| 2 | 13.8 | 21.7 | 18.4 | 12.4 | **10.5** | v5 | 1.32x |
| 4 | 14.2 | 20.4 | 19.3 | 13.9 | **11.8** | v5 | 1.20x |
| 8 | 15.8 | 21.1 | 20.4 | 14.2 | **13.2** | v5 | 1.19x |
| 16 | 16.9 | 21.4 | 20.6 | 16.0 | **15.2** | v5 | 1.11x |
| 32 | 18.8 | 20.9 | 22.0 | 17.3 | **15.7** | v5 | 1.20x |
| 64 | 19.4 | 22.1 | 22.4 | 19.3 | **17.5** | v5 | 1.11x |
| 96 | 21.6 | 22.6 | 23.4 | 21.3 | **20.2** | v5 | 1.07x |
| 128 | 23.3 | 24.1 | 23.7 | 23.5 | **22.5** | v5 | 1.04x |
| 192 | 26.7 | 26.0 | **25.6** | 27.9 | 27.6 | v3b | 1.04x |
| 256 | 29.2 | 27.1 | **27.0** | 32.0 | 31.8 | v3b | 1.08x |
| 384 | 35.7 | 30.6 | **30.3** | 40.7 | 41.0 | v3b | 1.18x |
| 512 | 42.1 | 33.8 | **33.3** | 50.5 | 50.3 | v3b | 1.26x |
| 768 | 55.8 | 40.6 | **40.5** | 69.2 | 70.0 | v3b | 1.38x |
| 1024 | 68.9 | 47.0 | **46.7** | 87.2 | 89.7 | v3b | 1.48x |
| 1536 | 96.2 | 60.4 | **60.3** | - | - | v3b | 1.60x |
| 2048 | 123.2 | **73.9** | 74.2 | - | - | v3 | 1.67x |
| 4096 | 230.3 | **129.2** | 129.7 | - | - | v3 | 1.78x |
| 16384 | **964.0** | 457.3 | 457.7 | - | - | v3 | 2.11x |

Reading: v5 strictly dominates v4 at every size (hence v4 dropped from the TP4
table); v5 owns 1-128; v3b beats v3 exactly where the barrier funnel matters
(small shapes + the 192-1536 mid band) and ties beyond; mm's former 128-192
island is a coin-flip zone (winner alternates run-to-run within ~5%) -- the
baked table keeps mm there per the idle-box data. The tuned-table picks sit on
the winner column everywhere else.

**Open.** TP8 sweep for v5/v3b bands; fold the RMSNorm epilogue into v5's local
reduce (full row on-rank across the whole 1-96 band now); 128-192 may fall to a
hierarchical push (half-multicast, half ld_reduce) if it ever matters.
