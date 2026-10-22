---
title: "DFlash core-change minimization"
description: "Controlled ablations of optional runtime changes, their evidence, and the resulting core diff."
---

# DFlash core-change minimization

Started **2026-10-22** at the user's request. The objective is to reduce the
core diff by removing changes that do not help, while preserving bounded draft
KV and complete recurrent-checkpoint recovery. Test minimization is deferred.
This work supersedes the paused-optimization status in the earlier handoff.

Completed **2026-10-22**: **50 net lines removed in two runtime files** after
21 individual A/B decisions and combined validation. Final paired throughput
changes are **+0.053% at C1** and **+0.030% at C32**, within observed variation.
The final source passed 48 tests, 132 subtests, and forced-host recovery. Tests
and the checkpoint index are unchanged; no commit or push was made.

The starting tree includes the staging-memory accounting cleanup. HEAD is
`6f168a98d8438f3cb8bfd7ecfbfd9c2efa15b5e2`; the checkpoint index remains
`63ceecea536f74cd797eea515199a196126181df`. The complete runtime diff covers
44 files under `python/sglang/srt` and `python/sglang/kernels`, including new
feature modules. The initial diff and source/test hashes are saved under
`/tmp/dflash-core-min-20261022/`.

## Measurement protocol

- Compare the retained implementation with one coherent removal at a time.
  Each server imports a frozen full `python/` snapshot. Record the exact changed
  files, source hashes, commands, effective configuration, and result hashes.
- Start with an A/A control using identical source. For A/B, use TP2 GPU pairs
  4/5 and 6/7 on the same NUMA node, then launch fresh servers with assignments
  reversed. Server affinity is CPUs 48–95; the serial client uses CPUs 0–7.
- Both arms enable bounded DFlash and HiCache with identical model, draft,
  precision, graph settings, write policy, and pool capacities. Pin Full KV to
  **1,048,576 tokens**, above the workload's total tokens, to prevent different
  free-memory profiles or staging reserves from changing capacity. Keep 32
  running requests, 256 recurrent slots, and the existing raw-tail budget.
- Measure total input-plus-output tokens/s at C1 and C32. Use the established
  seeded random-ID workload: 2,048 input/512 output tokens, 32 prompts at C1
  and 128 at C32, cache flush and four warmup requests per run. Exclude a full
  prime for each server/concurrency before three measured repeats. Alternate
  arm order and workload order. No concurrent benchmarks, server startups,
  tests, profiling, or heavy analysis during measured runs.
- Validate completed-request and token counts, retain acceptance and latency
  metrics, and record GPU clocks, temperature, and power around each run.
  Run long-decode, prefix-reuse, and 32-branch correctness checks before timing.
- Report both assignment-specific medians and paired ratios. A paired-run
  bootstrap interval measures repeat noise; it does **not** establish launch
  stability. Repeat fresh launches for borderline or contradictory results.
  Treat differences around 0.5% cautiously, in light of A/A noise and crossover
  consistency. Do not equate lack of significance with proven equivalence.
- Recovery-only changes need recovery/transfer measurements; unchanged cold
  throughput does not show that a restore optimization is useless. Validate
  forced host recovery and the retained combination before completing cleanup.

Positive B/A means removing the change improved throughput. Benchmark variants
live outside the working tree until the evidence supports a decision. The
checkpoint index is not modified.

Each individual result measures the removal with the other changes retained.
The percentages are not additive: multiple changes can eliminate different
waits on the same path. Any accepted combination needs its own comparison.

## Optional changes reviewed

These began as hypotheses, with final decisions based on isolated comparisons
and combined validation. Coupled lifecycle edits form one coherent change;
independent upload/clear operations remain separate.

| ID | Change | Main files | State |
| --- | --- | --- | --- |
| C01 | One asynchronous request-index upload shared by both mappings | `memory_pool.py` | Keep: removal costs 2.02% at C32 |
| C02 | Asynchronous draft context/prefix length uploads | `dflash_worker_v2.py` | Keep: removal costs 4.38% at C32 |
| C03 | Asynchronous prefill tracking-mask/sequence uploads | `schedule_batch.py` | Keep: C32 assignment-median removal losses 2.14–2.20% |
| C04 | Keep prefill tracking indices on device | `schedule_batch.py` | Keep: removal costs 2.90% at C32 |
| C05 | Tensor `minimum` for convolution index bounds | `hybrid_linear_attn_backend.py` | Keep: removal costs 2.35% at C32 |
| C06 | `index_fill_` for recurrent slot clearing | `unified_memory_pool.py` | Keep: removal costs 0.66% at C32 |
| C07 | Build FlashInfer checkpoint plans from host mirrors | `gdn_flashinfer.py` | Keep: removal costs 2.53% at C32 with unaligned inputs |
| C08 | CPU ownership bits for recurrent tracking slots | `schedule_batch.py`, `memory_pool.py`, result processing, session cleanup | Keep: removal costs 1.27% at C32 |
| C09 | `fill_`/`copy_` when replacing a tracking slot | `memory_pool.py` | Keep conservatively: all four C32 assignment medians favor it |
| C10 | Staged recurrent D2H copies | `pool_host/mamba.py` and its sizing | Keep: removal costs 0.81% at C32 |
| C11 | Staged strided target-KV D2H copies | `pool_host/mha.py`, shared transfer kernels and sizing | Keep conservatively: all four C32 assignment medians favor staging |
| C12 | Pipelined HiCache readiness consensus | `unified_radix_cache.py` | Keep: removal costs 6.00% at C1; C32 inconclusive |
| C13 | Raw tails share HiCache's D2H stream | `kv_cache_builder.py` | Keep conservatively: small consistent C1 advantage; C32 inconclusive |
| C14 | Retain draft bindings when Full pages are freed | `allocator/dflash_swa.py` | Keep: removal costs 0.61% at C32 and adds release logic |
| C15 | Record canonical remaps on the host instead of comparing device tables | `unified_radix_cache.py`, `dflash_cache.py`, `dflash_feature_tail.py` | Keep: removal costs 3.72% at C32 |
| C16 | Direct strided recurrent transfers instead of gather/scatter fallback | Mamba transfer kernels/wrapper and `pool_host/mamba.py` | Keep: removal costs 5.10% at C32 |
| C17 | Enqueue target prefill before independent draft-window preparation | `dflash_worker_v2.py` | Keep for control-flow clarity; throughput inconclusive |
| C18 | Skip tail-selection gathers for short prefill extensions | `dflash_worker_v2.py` | Retain: the two-line guard avoids identity gathers; removal adds copies without a meaningful simplification |
| C19 | Remove repeated donation prechecks; use the helper's existing failure result | Mamba tree component | Applied; final combined and forced-host validation passed |
| C20 | Combined window gathering/rebinding versus the existing two-kernel path | `dflash_window.py` | Restored after the larger cleanup repeatedly lost 0.21% at C1 |
| C21 | Page-only fast specialization | `dflash_window.py` | Removed only the specialization/dispatch unreachable with C20 retained; final validation passed |

The remaining changes implement feature contracts: opt-in validation, separate
Full/SWA physical views and translations, fixed pool sizing and admission,
prefix-valid writes, raw capture/result ownership, exact-boundary donation,
independent recurrent/tail residency, transfer fences, and Python TreeCore
support. They require a contract-preserving simplification, not an ablation
that drops recovery data, weakens admission, or introduces target replay.
The integration audit below covers those contracts and their scope.

## Results and decisions

C19's duplicate donation checks have been removed; it has the same Python AST
as its tested candidate. C18's short-tail guard is retained. The kernel cleanup
now removes only the unused page-only
specialization and dispatch. The fused serving kernel and its shared helper
match the baseline AST exactly.

The first, larger cleanup removed both kernels but repeatedly lost about
0.21% at C1 in combined validation. That result changed the decision: C20 is
restored. C18 is also retained to avoid extra identity gathers for a two-line
reduction. The final cleanup removes **50 net lines in two runtime files**.
All 2,762 initially hashed test files and the checkpoint index remain
unchanged. Final combined throughput and forced-host recovery validation passed.
Full logs, snapshots,
manifests, and the reusable driver are under `/tmp/dflash-core-min-20261022/`.
[Compact measurement evidence](draft-swa-core-minimization.json) preserves
completed cases, source hashes, commands, per-run metrics, and GPU assignments.

The initial identical-source control (`aa-r0`) completed three measured repeats
per arm/concurrency after primes. Paired B/A ratios were **+0.023% at C1** and
**−0.271% at C32**. The per-arm median ratios were +0.011% and −0.169%.
Both servers passed the five reference output hashes and 32 branches.

Even this identical-source comparison produces a repeat-only interval excluding
zero. That demonstrates why the interval alone cannot justify a code decision:
it omits fixed GPU/launch effects. This control used one GPU assignment; actual
removal decisions require the crossover and, when needed, fresh-launch repeats.

Before C07, the driver was changed to reuse the benchmark client's imports,
avoiding repeated process startup outside the timed interval. Every invocation
still calls the unchanged serving CLI, rebuilds its arguments and event loop,
resets the dataset seed, flushes the cache, and performs the same warmups. The
client is pinned to the same CPUs and requests remain serial between servers.
Results record the driver/client hashes and this change in client lifetime.

An additional identical-source control (`aa-v3-unaligned-r0`) used the
**2,087-input/512-output** workload needed for C07. Three paired repeats measured
**+0.069% at C1** and **−0.246% at C32**, with median ratios +0.050% and −0.146%.
Both servers passed the reference serving checks. This is again a one-assignment
calibration, not an A/B removal decision. The measured before/after clock
samples for both controls and the first three ablations were consistently
1,980 MHz SM and 3,201 MHz HBM; these are boundary samples, not continuous power
or clock traces. Compare variants within a case, not absolute throughput across
different input lengths or client-driver versions.

The first C08 launch stopped before measurement because baseline HTTP port
33050 was occupied when the server finished initializing (`EADDRINUSE`). No
timed runs or serving-check results came from that attempt. Both servers were
stopped, and the same sources and settings were retried on ports below the
host's ephemeral range. The original failure logs remain in
`c08-ownership-r0/`; the replacement case is `c08-ownership-retry-r0`.
This infrastructure failure is not evidence about the ownership change.

### C01: retain the shared request-index upload

Removing this upload loses **2.016% at C32** across six paired repeats. The
per-assignment median losses are 2.137% and 1.568%; the result persists when GPU
assignments reverse. C1 is effectively unchanged (+0.056% overall; assignment
medians +0.146% and −0.076%). This is larger than the A/A difference and is not a
candidate for removal.

The alternative passed 13 unit tests (and two subtests), and all four servers
passed the five reference output hashes and 32-branch serving check. All timed
runs completed the expected requests and token counts. Source B differed only in
`memory_pool.py`, replacing the shared asynchronous row-index upload with two
Python-list indexing operations. Evidence: `c01-reqmap-r0/results.json` under
the artifact root above. The repeat-only C32 interval is [−2.321%, −1.742%]; the
crossover, rather than that interval alone, supports the decision.

### C02: retain asynchronous draft-length uploads

Replacing the two asynchronous uploads with the original device `torch.tensor`
constructors loses **4.382% at C32** and **0.620% at C1** across six paired
repeats. The C32 assignment-median losses are 4.370% and 4.217%; the C1 losses
are 0.601% and 0.710%. Keep the change: the larger concurrent-serving effect is
consistent across assignments and substantially exceeds the A/A difference.

All four servers passed the five reference hashes and 32 branches. Timed runs
completed every expected request/token, and acceptance lengths match between
arms. Only `dflash_worker_v2.py` differed. Evidence:
`c02-draft-upload-r0/results.json`. The repeat-only C32 interval is
[−4.837%, −4.038%]. The pageable constructors wait for the previously enqueued
target prefill; preserving the asynchronous uploads avoids that wait.

### C03: retain asynchronous prefill tracking uploads

Replacing the tracking-mask and sequence-length uploads with pageable
`torch.tensor` constructors loses **2.827% at C32** across six paired repeats.
The assignment-median losses are **2.200% and 2.143%**; C1 is effectively
unchanged (−0.099%). Keep the asynchronous uploads.

Acceptance varied in two baseline runs: A reported 5.16091 in those runs and
4.98932 in the other four, while B reported 4.98932 throughout. The full paired
result includes those runs and should not be described as a pure measurement of
copy overhead. As a descriptive sensitivity check, the four pairs with matching
reported acceptance give −2.132%. This post-hoc subset does not replace the
primary result or the assignment medians; no raw observations were discarded.

All four servers passed the five reference hashes and 32 branches, and measured
request/token counts matched. Only `schedule_batch.py` differed. Evidence:
`c03-track-upload-r0/results.json`. The repeat-only C32 interval is
[−3.678%, −2.091%], including the acceptance variation.

### C04: retain device tracking indices

Reading each tracking index back with `.item()` and rebuilding a device tensor
loses **2.900% at C32** across six paired repeats. Assignment-median losses are
2.924% and 2.781%; C1 is unchanged (−0.001% overall). Keep the device-view and
stacking path.

All four servers passed the five reference hashes and 32 branches, and measured
request/token counts matched. Only `schedule_batch.py` differed. Evidence:
`c04-track-indices-r0/results.json`. The repeat-only C32 interval is
[−3.127%, −2.670%]; both GPU assignments favor keeping the change.

### C05: retain the tensor convolution bound

Replacing tensor `minimum` with the original `clamp` expression loses
**2.354% at C32** across six paired repeats. Assignment-median losses are
2.440% and 1.971%; C1 is effectively unchanged (−0.074% overall). Keep this
small change: its concurrent-serving benefit persists after the GPU swap.

All four servers passed the five reference output hashes and 32 branches,
and every measured request/token count matched. Only
`hybrid_linear_attn_backend.py` differed. Evidence:
`c05-conv-bound-r0/results.json`. The repeat-only C32 interval is
[−2.664%, −2.016%]. The ordinary 2,048-token workload exercises this bound
through convolution tracking even when its SSM checkpoint plan is empty.

### C06: retain recurrent-slot `index_fill_`

Returning to scalar indexed assignment loses **0.657% at C32** over six paired
repeats, with assignment-median losses of 0.801% and 0.562%. C1 changes by
−0.096%. This is a modest benefit, but both assignments favor retaining the
small `index_fill_` change; the removal is not a useful simplification.

All four servers passed the five reference hashes and 32 branches, and measured
request/token counts matched. Only `unified_memory_pool.py` differed. Evidence:
`c06-slot-clear-r0/results.json`. The repeat-only C32 interval is
[−0.780%, −0.529%]; it does not establish performance on other workloads.

### C07: retain host metadata for unaligned checkpoint planning

With **2,087 input/512 output tokens**, removing the host-metadata path loses
**2.532% at C32** across six paired repeats. Assignment-median losses are
2.761% and 2.119%; C1 is unchanged (−0.007% overall). Keep the change. The
unaligned workload exercises the checkpoint-planning code that the usual
2,048-token input can skip.

The alternative passed 20 unit tests (41 subtests). All four servers passed the
five reference output hashes and 32 branches, and all timed request/token counts
matched. Only `gdn_flashinfer.py` differed. Evidence:
`c07-checkpoint-plan-r0/results.json`. The repeat-only C32 interval is
[−2.767%, −2.270%]; the same-sign crossover supports the decision.

### C08: retain CPU tracking-slot ownership

Removing the ownership bits and restoring device scalar reads/filtering loses
**1.266% at C32** and **0.227% at C1** across six paired repeats. C32
assignment-median losses are 1.325% and 1.014%; C1 losses are 0.242% and
0.239%. Keep the ownership bookkeeping: the concurrent-serving benefit survives
the GPU swap and exceeds the identical-source difference.

The coherent removal touches allocation/replacement, scheduling, result
processing, and session cleanup in four files; other upload and slot-clear
optimizations remain enabled. All four servers passed the five reference
hashes and 32 branches. Measured request/token counts and acceptance lengths
matched between arms. Evidence: `c08-ownership-retry-r0/results.json`. The
repeat-only C32 interval is [−1.479%, −1.008%]. The initial startup failure
described above contributed no measurements; both frozen source digests match
between that attempt and this retry.

### C09: retain slot replacement conservatively after fresh launches

Replacing `fill_`/`copy_` with indexed assignment measures **−0.054% at C1**
and **+0.020% at C32**. The C32 assignment medians are −0.628% and −0.038%.
Five paired C32 observations favor the retained implementation, but one slow
baseline run reverses the aggregate result. That run remains in the primary
measurement. Its TTFT is 391.6 ms, versus 328.5–347.4 ms for the other baseline
C32 runs; acceptance is identical throughout. The recorded clock samples are
also unchanged. These observations do not identify the cause of the slow run.

The fresh crossover measures **+0.018% at C1** and **−0.441% at C32**.
Its C32 assignment medians are −0.780% and −0.144%; C1 reverses direction
(−0.145%, +0.226%). Across both comparisons, the paired removal results are
**−0.018% at C1** and **−0.211% at C32**, retaining the initial slow observation.
Keep the small slot-write change conservatively: all four C32 assignment
medians favor it, although the effect is modest and C1 remains inconclusive.

All eight servers passed the five reference hashes and 32 branches, with
matching measured request/token counts and acceptance lengths. Only
`memory_pool.py` differed. Evidence: `c09-slot-replace-r0/results.json` and
`c09-slot-replace-r1/results.json`. The repeat-only C32 intervals are
[−0.546%, +0.849%] and [−0.793%, −0.074%]; they do not establish launch
stability or a benefit on other workloads.

### C10: retain recurrent copy-engine staging

Disabling recurrent staging loses **0.810% at C32** and **0.308% at C1**
across six paired repeats. Assignment-median losses are 1.157% and 0.450%
at C32, and 0.351% and 0.261% at C1. Keep the staging path and its matching
61.40625 MiB per-rank reservation for this configuration. Both assignments
favor retaining it; this is not a useful removal.

The timing variant disables staging allocation in `pool_host/mamba.py` and
uses its existing direct-transfer fallback. The fixed Full-KV cap and budgeting
formula are held constant to isolate transfer behavior. All four servers
passed the five reference hashes and 32 branches, and all measured request/token
counts matched. C1 acceptance matches; C32 A ranges from 4.98639 to 4.98932,
while B reports 4.98932 throughout. Evidence: `c10-mamba-staging-r0/results.json`.
The repeat-only C32 interval is [−1.118%, −0.459%].

### C11: retain target-KV staging conservatively after fresh launches

Disabling staging for strided target-KV rows measures **−0.297% at C32**
and **−0.035% at C1**. The C32 assignment medians are −0.657% and −0.010%;
C1 reverses direction (−0.252%, +0.200%). The repeat-only C32 interval is
[−0.668%, +0.106%].

The fresh crossover measures **−0.029% at C1** and **−0.022% at C32**.
Its assignment medians are −0.060%/−0.045% at C1 and −0.567%/−0.063%
at C32. Across both crossovers, the paired removal results are **−0.032%
at C1** and **−0.159% at C32**. Keep the staging path and its matching
40 MiB per-rank reserve conservatively: all four C32 assignment medians favor
staging. The size of the benefit is uncertain, and these results do not
establish a substantial serving gain.

One fresh B run reports acceptance 5.16091, versus 4.98932 for every other
measured C32 run in that comparison. It remains in the primary result; the
fresh repeat-only interval [−0.555%, +0.693%] includes that variation.

Only `pool_host/mha.py` differs in the timing variant; recurrent staging stays
enabled. The fixed Full-KV cap and budgeting formula remain constant. All eight
servers passed the five reference hashes and 32 branches, with matching
measured request/token counts. C1 acceptance matches throughout. Evidence:
`c11-kv-staging-r0/results.json` and `c11-kv-staging-r1/results.json`.

### C12: retain pipelined HiCache readiness for C1

Returning to synchronous readiness consensus loses **6.000% at C1** across
six paired repeats. The assignment-median losses are 6.256% and 5.472%, so the
low-concurrency benefit survives the GPU swap. Keep the optimization.

C32 does **not** establish a clear benefit: the overall removal result is
−0.381%, but the assignment medians are −0.691% and +0.086%. The repeat-only
interval [−0.691%, −0.065%] does not resolve that launch/assignment variation.
Do not describe this as a confirmed C32 gain.

The alternative passed 544 unit tests (59 skipped, 39 subtests), followed by
all four servers' five reference hashes and 32 branches. Every timed request and
token count matched. C1 acceptance was identical; measured C32 acceptance ranged
from 4.98932–4.98977 for A and 4.98639–4.98932 for B. Only
`unified_radix_cache.py` differed. Evidence: `c12-readiness-r0/results.json`.

### C13: retain stream sharing conservatively after fresh launches

Giving raw tails their own D2H stream instead of sharing HiCache's stream
measures **−0.021% at C1** and **−0.096% at C32** over six paired repeats.
The C32 assignment-median results reverse sign (−0.222%, +0.223%); the C1
median differences are −0.061% and −0.072%. These small differences do not
establish a useful benefit from stream sharing in the current implementation.

The fresh confirmation (`c13-d2h-stream-r1`) measures **−0.102% at C1** and
**−0.211% at C32**. Its C1 assignment medians are −0.060% and −0.147%; C32
again reverses sign (−0.342%, +0.227%). Across both comparisons, the combined
paired removal results are **−0.061% at C1** and **−0.153% at C32**.

Keep this small builder change conservatively. All four C1 assignment medians
favor sharing, although the magnitude is small; C32 remains inconclusive.
This does not establish a substantial throughput gain. Removing seven lines
would not justify giving up the small, directionally consistent C1 result.

All eight servers passed the five reference output hashes and 32 branches,
with the expected measured request/token counts. The confirmation also passed
five CUDA ownership/ordering tests and three subtests. Only
`kv_cache_builder.py` differed. Evidence: `c13-d2h-stream-r0/results.json` and
`c13-d2h-stream-r1/results.json`. The repeat-only C32 intervals are
[−0.352%, +0.168%] and [−0.556%, +0.136%], respectively; they do not resolve
the launch/GPU effects.

### C14: retain spare bindings across Full-page frees

Restoring draft-binding release alongside Full-page release loses **0.613%
at C32**, with assignment-median losses of 0.792% and 0.253%. C1 improves
by 0.077% (assignment medians +0.112%, +0.053%). Keep the current behavior:
it avoids the concurrent-serving loss and uses less release-path logic.
The small C1 result does not justify reinstating GPU filtering and an additional
allocator release/history operation.

Both variants use the same fixed draft reservation and preserve content-validity
repair. All four servers passed the five reference hashes and 32 branches;
measured request/token counts and acceptance lengths matched. Only
`allocator/dflash_swa.py` differed. Evidence: `c14-spare-bindings-r0/results.json`.
The repeat-only C32 interval is [−0.852%, −0.373%]. This establishes the
tradeoff on the measured workload, not a universal benefit at every concurrency.

### C15: retain host-recorded canonical remaps

Returning to device-table snapshots and comparisons loses **3.722% at C32**
and **0.222% at C1** across six paired repeats. C32 assignment-median losses
are 3.703% and 3.425%; C1 losses are 0.367% and 0.077%. Keep the host-recorded
range and its radix hook. The concurrent-serving benefit is substantial and
survives the GPU swap.

The alternative coherently restores device comparison in `dflash_cache.py`
while removing the host range and radix notification. Spare-binding retention
remains enabled. Before launching the candidate, its stale checkpoint-era
binding-lifecycle docstring was corrected; executable AST was unchanged.
All four servers passed the five reference hashes and 32 branches, with
matching measured request/token counts and acceptance lengths. Evidence:
`c15-canonical-remap-r0/results.json` and `c15-variant-review.json`. The
repeat-only C32 interval is [−3.953%, −3.497%].

### C16: retain direct strided recurrent transfers

Forcing the existing gather/scatter fallback loses **5.101% at C32** and
**0.356% at C1** across six paired repeats. Assignment-median losses are
5.361% and 4.814% at C32, and 0.292% and 0.485% at C1. Keep direct strided
transfers. The ordinary-serving D2H regression alone rejects the removal;
these timings do not quantify an independent H2D speedup.

The alternative changes `_can_transfer_strided_slots` to decline the direct
path, preserving the existing fallback and all other staging choices. The
fallback also allocates extra gather/scatter buffers, as described in the
integration audit. A fixed-capacity timing trial would not validate the memory
budget for adopting that fallback generally.

All four servers passed the five reference hashes and 32 branches, and measured
request/token counts matched. Evidence: `c16-strided-state-r0/results.json`.
The repeat-only C32 interval is [−5.270%, −4.928%]. Final forced-host recovery
will validate correctness of the retained implementation separately.

### C17: retain the simpler existing ordering

Preparing draft state before enqueuing target prefill measures **−0.023% at C1**
and **−0.048% at C32**. Assignment medians reverse direction at both
concurrencies: −0.222%/+0.173% at C1 and −0.351%/+0.450% at C32.
The repeat-only C32 interval is [−0.312%, +0.247%]. This comparison does not
establish a useful throughput benefit for either ordering.

Keep the current sequence for control-flow clarity. Preparation stays after
the existing idle-batch return; moving it earlier needs an additional idle
guard to preserve collective execution. The alternative removes two net lines
mainly by dropping the Full/SWA independence comment, without removing state,
an interface, or a kernel. That is not a useful reduction in complexity or core
diff scope. This decision does not claim a measured speedup or equivalence.

All four servers passed the five reference hashes and 32 branches, and measured
request/token counts matched. Only `dflash_worker_v2.py` differed. Evidence:
`c17-target-order-r0/results.json`, using the idle-safe
`without-c17-r0-v2` snapshot. The earlier invalid prototype was not measured.

### C18: retain the short-tail guard after reviewing the combined scope

Always selecting prefill-tail rows measures **+0.092% at C1** and **+0.112%
at C32**. Assignment medians are +0.065%/+0.114% at C1 and
−0.018%/+0.219% at C32. The repeat-only C32 interval is [−0.092%, +0.313%].
The first crossover does not establish a useful serving benefit from the
short-extension guard. Removing it also makes short extensions perform the
existing selection and copy, so both a fresh comparison and a memory-bound
review were needed.

The fresh crossover measures **+0.144% at C1** and **−0.009% at C32**.
Its assignment medians are +0.069%/+0.205% at C1 and −0.530%/+0.288%
at C32. Across both crossovers, the paired removal results are **+0.118%
at C1** and **+0.052% at C32**. All four C1 assignment medians favor removal,
while C32 remains inconclusive. That initially led to removing the guard.
The fresh repeat-only C32 interval is [−0.356%, +0.351%].

All eight servers passed the five reference hashes and 32 branches, with
matching measured request/token counts and acceptance lengths. Only
`dflash_worker_v2.py` differed. Evidence: `c18-tail-selection-r0/results.json`
and `c18-tail-selection-r1/results.json`. The temporary edited file matched the
measured candidate byte for byte, and its feature-tail suite passed 21 tests
and 71 subtests before combined validation.

Retain the guard in the final cleanup. The 52-line combined revision still
measured a small C1 cost, and removing this two-line guard adds identity gathers
and allocations without removing an ownership rule, interface, or substantial
control-flow complexity. This is a conservative scope decision, not proof that
C18 caused the combined cost. The worker is restored byte for byte to the
starting source. The standalone measurements remain in the record.

The existing reserve covers the selection output: it has
`sum(min(extend_len_i, W))` rows, bounded by both total new prefill rows and
`N × W`. The sizing formula already reserves
`max(2 × W, min(prefill_rows, N × W))` projection rows. Here that is 4,096
rows, or **96 MiB per rank** of raw-feature payload. Always selecting short
tails changes allocation frequency, not that maximum or the number of
simultaneous projection operations. No reserve increase is required for C18.
This is a source-derived bound, not a measured total CUDA memory peak; see
`c18-memory-review.json`.

### C19: delegate donation eligibility to the existing helper

The caller now retains its boundary and missing-state guards, then handles
`donate_checkpoint` returning `None`. The helper already checks for a matching
checkpoint and reserves host capacity before removing request ownership.
Repeating those checks in the caller and asserting success adds no contract
that the helper lacks.

The isolated comparison measures **−0.022% at C1** and **−0.157% at C32**.
C32 assignment medians reverse sign (−0.371%, +0.184%); C1 medians are
−0.025% and −0.060%. These results do not establish a throughput improvement
or a material regression on this workload. The repeat-only C32 interval is
[−0.403%, +0.115%]. The change is retained for its clearer ownership of
eligibility checks; the final combined and forced-host checks also passed.

The candidate passed 21 unit tests and 10 subtests, including full-capacity
decline and pending donations. All four serving instances passed the five
reference hashes and 32 branches; every measured request/token count matched.
Only `unified_cache/components/mamba.py` differed. Evidence:
`c19-donation-checks-r0/results.json` and `c19-unit.json`.

### C20: restore the combined window kernel after combined validation

Using the existing gather-then-rebind path instead of the combined kernel
measures **+0.102% at C1** and **−0.055% at C32** across six paired repeats.
C1 assignment medians are +0.061% and +0.067%; C32 medians reverse sign
(−0.254%, +0.214%). This first crossover does not establish a useful serving
benefit from the combined kernel.

The fresh confirmation (`c20-window-fusion-r1`) measures **−0.038% at C1**
and **−0.060% at C32**. Its assignment medians are −0.088%/+0.017% at C1
and −0.354%/+0.214% at C32. Combining both crossovers gives **+0.032% at C1**
and **−0.057% at C32**. These results do not show a useful serving benefit
from retaining the combined kernel on this workload.

These isolated results initially led to removing its private kernel and
dispatch in favor of the existing gather-then-rebind calls, a reduction of 83
net lines. C21 below was then measured in that two-kernel context, where the
page-only specialization was live.

The complete 171-line cleanup subsequently lost **0.198% and 0.215% at C1**
in two fresh crossovers, with all four assignment medians negative. The
individual comparisons therefore did not establish that removing both kernels
was free in combination. Restore the fused kernel and its shared rebinding
helper to the exact baseline AST. This conservative decision preserves the
existing serving execution path; it does not attribute the full combined cost
to C20 alone. The final combination was validated separately below.

The alternative passed six CUDA tests and 51 subtests. All eight servers passed
the five reference hashes and 32 branches, with the expected measured
request/token counts. Only `dflash_window.py` differed. Evidence:
`c20-window-fusion-r0/results.json` and `c20-window-fusion-r1/results.json`.
The repeat-only C32 intervals are [−0.295%, +0.183%] and
[−0.337%, +0.226%]. Actual-source and final combined validation are recorded
separately; deleting unused definitions is not itself a serving benchmark.
The actual edited kernel subsequently passed the same six CUDA tests and 51
subtests, plus Ruff format/lint checks, before the C21 benchmark began.

### C21: remove only the page specialization unreachable from serving

With C20's two-kernel implementation as A, routing every page update through
the general plan measures **−0.101% at C1** and **+0.069% at C32**.
C1 assignment medians are −0.099% and −0.105%; C32 medians are −0.314% and
+0.348%. Reported acceptance lengths match between arms throughout. C32 does
not establish a benefit for either implementation; the small C1 cost was
directionally consistent in that first comparison.

The fresh confirmation measures **+0.086% at C1** and **−0.369% at C32**.
Its C1 assignment medians are −0.054% and +0.184%; C32 medians are −0.623%
and +0.363%. Combining both crossovers gives **−0.008% at C1** and
**−0.150% at C32**. The C1 effect did not repeat consistently, and C32
continues to reverse direction with GPU assignments.

One fresh C32 A run reported acceptance 5.16091, versus 4.98932 for all other
measured runs in that confirmation. Keep it in the primary result. The five
pairs with matching reported acceptance give +0.076% as a descriptive
sensitivity check, not a replacement result. The fresh repeat-only C32
interval is [−1.342%, +0.333%], including that variation.

The initial choice removed the private page specialization and its shared
rebinding helper, another 81 net lines in the C20-removed context. The combined
C1 regression described below led to restoring the fused kernel and that
helper. The final C21 scope is narrower: remove only the page-only kernel and
its fast dispatch, **45 net lines**. With the fused serving path retained,
that dispatch is unreachable from its sole production caller. Direct calls
still use the general implementation and retain their numerical contract.

The alternative passed six CUDA tests and 51 subtests. All eight serving
instances passed the five reference hashes and 32 branches, and measured
request/token counts matched. Evidence: `c21-page-specialization-r0/results.json`
and `c21-page-specialization-r1/results.json`. The first repeat-only C32
interval is [−0.275%, +0.418%]. These cases use
`without-c20-r0` as A and `without-c20-c21-r0` as B; it measures C21 alone in
the selected C20 context, not the combined effect of removing both kernels.
The actual edited module also passed six CUDA tests and 51 subtests, plus Ruff
format/lint checks, after the unused definitions were deleted.

## Combined cleanup validation

### Rejected larger cleanup

The actual cleanup is frozen as `final-cleanup-r0`, with 5,846 source files and
digest `c02c22169dbc65ce1500fb95645dfa7648767452b9e2c77bc704a6b87a85ddaa`.
It differs from the starting snapshot only in the three intended runtime files.
Before server startup, the host-tail pool, window-kernel, and feature-tail suites
passed **48 tests and 132 subtests**. Ruff lint and format checks also passed.

The first complete crossover (`final-combined-r0`) measures:

| Concurrency | Paired cleanup versus baseline | Assignment 0 median ratio | Assignment 1 median ratio |
| --- | ---: | ---: | ---: |
| 1 | −0.198% | −0.043% | −0.396% |
| 32 | −0.083% | −0.158% | −0.028% |

All four servers passed the five reference output hashes and 32 branches.
Measured request/token counts and acceptance lengths match between arms. The
repeat-only intervals are [−0.370%, −0.037%] at C1 and [−0.217%, +0.061%]
at C32. The C1 cost varied by assignment, so a fresh crossover was run.
The first result is retained in full; it is not replaced by the confirmation.

Forced-host validation (`final-forced-host-r0`) passed on both baseline and
cleanup with a 131,072-token Full cap and 20 cache-pressure requests. Both
restored **4,096 host-cached tokens and zero device-cached tokens**, then
extended the **37-token suffix**. All five reference output hashes and all
32 branches passed. All servers stopped.

The fresh confirmation (`final-combined-r1`) measured **−0.215% at C1** and
**+0.271% at C32**. Its assignment medians were −0.342%/−0.116% at C1 and
−0.300%/+0.375% at C32. Across both comparisons, the paired results are
**−0.207% at C1** and **+0.094% at C32**. C1's small cost repeats across all
four assignment medians; C32 remains inconclusive. C1 acceptance matches
throughout; fresh C32 B acceptance ranges from 4.98639 to 4.98932, versus
4.98932 for A. No observations are dropped.

The larger cleanup was rejected on this combined evidence. C20 and its shared
helper are restored, leaving the C21 deletion unreachable from serving and
retaining the C18/C19 simplifications.

### Revised cleanup

`final-cleanup-r1` contains the actual **52-line** reduction. Its 5,846-file
source digest is
`527db73a9ef34fd27fd5991e5125f55b0228ed5e6f339da39955426d1c448cb7`.
`final-fusion-variant-review.json` records the unchanged serving-function ASTs
and the page-dispatch reachability proof. The revised comparison
(`final-revised-r0`) measured **−0.110% at C1** and **+0.307% at C32**.
Its assignment medians were −0.060%/−0.088% at C1 and −0.208%/+0.365% at C32;
acceptance and request/token counts matched. C32 remains inconclusive. All
48 tests, 132 subtests, and four reference serving checks passed. Its
forced-host run (`final-revised-forced-host-r0`) also passed both servers'
five reference hashes, 32 branches, and the 4,096-host/zero-device prefix
with a 37-token suffix.

C18 was then restored for the scope reason above, leaving only C19 and the
unreachable C21 specialization removal. The final **50-line** cleanup is
`final-cleanup-r2`, source digest
`f38037e989141a130e1b49b6e4f0c790274fbfdafc26a000d15e13f9aece8c46`.
Its completed comparison (`final-pruned-r0`) measured:

| Concurrency | Paired cleanup versus baseline | Assignment 0 median ratio | Assignment 1 median ratio |
| --- | ---: | ---: | ---: |
| 1 | +0.053% | −0.098% | +0.107% |
| 32 | +0.030% | −0.193% | +0.245% |

The repeat-only intervals are [−0.051%, +0.154%] at C1 and
[−0.215%, +0.258%] at C32. Acceptance lengths and measured request/token
counts match throughout. The final result is within observed run/assignment
variation, with no repeatable regression established on this workload; it is
not a general performance-equivalence or flag-on/off parity claim.

The actual final source passed **48 tests and 132 subtests** before timing.
All four timing servers passed the five reference hashes and 32 branches.
The separate forced-host case (`final-pruned-forced-host-r0`) passed both
servers with the 131,072-token cap and 20 pressure requests: **4,096 host-cached
tokens, zero device-cached tokens, and a 37-token new suffix**, with all five
hashes and 32 branches passing. All owned validation servers stopped.

Final verification confirms that the 5,846-file working Python tree matches
the measured snapshot, the 2,762 test files and their file set are unchanged,
and HEAD plus the 75-file checkpoint index are preserved. Ruff lint/format and
diff whitespace checks pass. The complete runtime diff is now **3,155 additions
and 201 deletions across 44 files**, down from 3,205/201. Evidence is in
`final-verification.json` and the compact JSON companion. Cache allocations,
admission caps, and staging/raw-feature reserves are unchanged.

## Core integration audit

The initial runtime diff was **3,205 additions and 201 deletions in 44 files**.
New feature modules account for much of it. Moving code into another file is not
by itself a simplification; review the contracts and runtime costs as well.

- Pool provisioning uses existing Full/SWA/Mamba allocators and views. The
  separate draft view and physical-write marker prevent a draft write from
  translating through the target's Full map. The additional validation checks
  the loaded draft's actual layer IDs, window, dtype, and head geometry against
  the target-created allocation; those checks are not replaced by a config
  comment or one successful model launch.
- Recurrent admission must count slots even when their incremental Full-token
  charge is zero. The scheduler's unit change from token equivalents to slots,
  and the independent allocator cap, implement that requirement. Reverting them
  would over-admit requests under fixed reservations.
- Matching, donation, and release keep a raw-tail handle paired with the exact
  checkpoint boundary. The tail stays owned while either a device or host
  recurrent copy exists. Cleanup of a rejected donation is separate from
  eviction of an accepted tree entry.
- Serving retains the fused window kernel for small geometries and the existing
  general helpers for the large-page-count fallback. The page-only wrapper
  uses those helpers for direct calls as well; its count/epoch/recycling
  protocol and physical-ownership contract remain intact.
- Both staging paths survived the ablations, so their allocation and shared
  sizing remain unchanged: 61.40625 MiB of recurrent scratch and 40 MiB of
  target-KV scratch per rank. The controlled timing variants kept the fixed KV
  cap and existing reserve to isolate transfer behavior. No unused staging
  allocation or budget term is left by the accepted cleanup.

C07 requires unaligned prefill lengths: aligned 2,048-token prompts can leave
`track_ssm_h_src` empty and skip its changed checkpoint-planning code. C16 also
would require host-restore measurements before a removal based on flat cold
timings, because ordinary cold requests do not cover the H2D benefit of direct
strided recurrent transfers. Its measured D2H regression already rejects that
removal. Its fallback also adds
temporary allocations: D2H gathers each layer and then stacks those gathered
tensors, while H2D allocates a packed destination before scattering. A flat
timing result would not justify removal without checking that additional peak
memory against the serving budget. The fixed-capacity timing trial alone does
not validate automatic memory sizing for that alternative.

The C19 candidate delegates missing-checkpoint and full-host-pool outcomes to
`DFlashFeatureState.donate_checkpoint`, which already returns `None` before
removing the request's checkpoint in either case. Boundary and missing-state
checks stay in the component. The valid serving lifecycle keeps the host pool
open: its production destroy path runs after the scheduler event loop exits,
with the asynchronous result worker stopped. This does not weaken the host
pool's public error on operations after destruction. The existing 21 host-tail
unit tests (10 subtests) pass on the isolated candidate, including full-capacity
decline and pending-donation ownership. The removal is applied after isolated
A/B validation; the final combined and forced-host checks also passed.

The initial kernel review found a page-only specialization that was unused by
serving **before C20's removal**. The original `dflash_update_window` called
`dflash_update_window_pages` only when
`next_power_of_2(max(count, held.numel())) > 4096`. The latter received
`1 + count` pages, so its own block size could not be smaller; its
`block <= 4096` fast branch was unreachable from that caller. The only other
repository caller was the direct numerical test. The general fallback was
still required for supported large geometries.

C20's initial removal made that page-only specialization live for small serving
geometries. C21 was measured in that changed context. Combined validation then
restored C20, making the original reachability proof applicable again. Only
the unused page specialization and dispatch are removed; the shared helper
stays because the fused kernel calls it. The direct numerical contract and
the revised combination are checked separately.
