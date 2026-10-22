# DFlash bounded-cache throughput optimization

**Implementation changed after these measurements.** The
[2026-10-22 deployment split](draft-swa-deployment-shapes.md) adds materialized
checkpoint KV, separates optional tap retention, and supports GPU-only cached
windows. The measurements below describe the preceding raw-tail/HiCache path.
They remain historical evidence and do not establish performance for the new
four-mode deployment matrix.

The [current performance matrix](draft-swa-performance-matrix.md) measures those
four modes against flag off with identical pool capacities and Python TreeCore.
It includes cold and warm-prefix workloads, two fresh launches per mode, and
observed GPU allocation. The earlier tables below are unchanged.

Measured on 2026-10-21 and 2026-10-22. This report supersedes the implementation
and performance status in [the initial fused-gather report](draft-swa-throughput-analysis.md).
The earlier report remains useful as the initial diagnosis.

**Last pre-split flag-on/off measurement, before staging accounting: parity was not
established, but the gap was smaller.**
The [parity pass](#parity-pass-2026-10-22) measured the retained code at about
**−0.5% total throughput at concurrency 1** and **−0.6% to −1.3% at concurrency 32**
against flag off on the same NUMA node, down from −1.0% and −3.2% on the same
harness at the start of that pass. Sections after the parity pass record the
2026-10-21 work. Their figures come from a different harness and code state and
are not directly comparable with the parity-pass tables.

A separate [review reproduction](#independent-review-reproduction-2026-10-22)
confirmed **−0.49% at C1 and −0.56% at C32** on fresh servers. Those reproduction
servers are stopped. The
[handoff](draft-swa-optimization-handoff.md) describes the current working tree,
retained changes, and the subsequent
[staging-memory accounting cleanup](#hicache-staging-memory-accounting-2026-10-22).
The throughput tables precede that cleanup; no new flag-on/flag-off comparison
has been run with the corrected cache budget. The user subsequently resumed
work to [minimize core changes through individual A/B ablations](draft-swa-core-minimization.md).
Those comparisons include the corrected budget and measure the cost of each
removal; they do not replace the earlier parity comparison.

The core minimization pass is complete. It removes **50 net lines in two runtime
files**: duplicate donation checks and a page-only specialization unreachable
from serving. The larger 171-line attempt repeatedly cost about 0.21% at C1,
so the fused serving kernel and shared helper were restored; the short-tail
guard was also retained. The final cleanup measured **+0.053% at C1** and
**+0.030% at C32** against the pre-minimization bounded-cache implementation,
within observed variation. Its 48 tests, 132 subtests, and forced-host recovery
passed. Tests and the checkpoint index are unchanged. The
[minimization report](draft-swa-core-minimization.md#combined-cleanup-validation)
records the decisions, rejected combinations, source hashes, and validation.

## Parity pass, 2026-10-22

This pass re-derived the gap from Nsight traces instead of continuing from the
previous ranking. Three findings changed earlier conclusions:

- Host synchronizations compound. Several prefill-path and release-path waits
  blocked the scheduler behind queued GPU work, and removing any one of them was
  invisible while another remained. Two changes rejected on 2026-10-21 (keeping
  draft bindings across Full-page frees, and copy-engine recurrent backups)
  became large wins once the other waits were gone.
- A pageable device-to-host readback (`.item()`, boolean indexing, `.cpu()`)
  waits behind every in-flight pinned copy on the device, including HiCache
  backups on other streams: about 900 µs instead of 10–40 µs when a backup is
  queued. Pinned asynchronous copies are not affected. This makes the same
  readback much more expensive with flag on.
- HiCache transfer kernels that store directly to host memory occupy SMs for
  the whole PCIe transfer. Overlapping model graphs slowed by 7–8% while a
  kernel-plus-copy backup ran, versus about 1.6% for copy-engine traffic alone.

### Measurement method

Each server ran a frozen source snapshot, pinned to its GPU pair's NUMA node.
Benchmarks ran serially; the first run per server and concurrency was excluded,
and later repeats alternated order. Flag on and flag off were compared only
within one NUMA node, because GPUs 4–7 ran 0.3–0.6% faster than GPUs 0–3.

The configuration is the bounded-cache evaluation setup: Qwen3.6-35B-A3B in
BF16 at TP2 on H200, the local DFlash baseline draft with 16 draft tokens,
unified memory, page size 64, FA3, `extra_buffer_lazy` recurrent caching,
breakable prefill graphs, 32 running requests, chunked prefill 4,096, and
`--mem-fraction-static 0.8`. Flag on adds `--enable-dflash-swa-cache` with
HiCache write-through at ratio 2 and the Python TreeCore; flag off uses the Rust
TreeCore without HiCache. This is not the
[Qwen3.6 cookbook](cookbook/autoregressive/Qwen/Qwen3.6.mdx) deployment, which
uses FP8 at TP1 with EAGLE MTP on H200. The pass adds no flags and changes no
defaults, so the cookbook page is unchanged.

The workload is `sglang.benchmark.serving` with random token IDs, 2,048 input
and 512 output tokens, range ratio 1, seed 20261021, `--flush-cache`, and four
warmup requests: 32 prompts at concurrency 1 and 128 at concurrency 32. The
metric is total input-plus-output tokens per second.

Flag-on servers running identical code varied between launches: three
measurements of the retained code at C32 gave −0.57%, −1.04%, and −1.30%
against the same flag-off server. Flag-off servers agreed within 0.1% on a
node. Treat single flag-on measurements of under about 0.5% as unresolved.

### Results

Same-node flag-on versus flag-off total throughput, medians of three repeats
(two for the starting point):

| Code | On / off GPUs | C1 | C32 |
| --- | --- | ---: | ---: |
| Start of pass | 2,3 / 0,1 | −1.01% | −3.21% |
| Start, flag off with Python TreeCore | 6,7 / 4,5 | −0.99% | −3.26% |
| + canonical-remap repair on the host | 4,5 / 6,7 | −0.76% | −3.20% |
| + prefill-path synchronizations removed | 0,1 / 2,3 | −0.72% | −2.89% |
| + copy-engine HiCache backups | 0,1 / 2,3 | −0.52% | −1.92% |
| + release without a count readback (retained code) | 4,5 / 6,7 | −0.50% | −0.57% |
| Retained code, same server later | 4,5 / 6,7 | −0.52% | −1.04% |
| Retained code, fresh server | 4,5 / 6,7 | — | −1.30% |

The prefill-path change is common code. It raised flag-off throughput by about
2.2% at C32 as well, so it barely moved the ratio. No change altered acceptance
length: every C1 run measured 5.151, and C32 runs measured 4.989 with flag on and
4.990 with flag off, except two runs of each that differed by under 0.1% as
batch timing varied. The
[JSON companion](draft-swa-parity-pass.json) records every run, server, GPU
pair, and source-tree digest.

### Retained changes

**Rebuild canonical-page rows from host-recorded ranges.** A radix insert that
moves a request's rows onto canonical pages now records the row range on the
request's feature state
([`unified_radix_cache.py`](../python/sglang/srt/mem_cache/unified_radix_cache.py),
[`dflash_feature_tail.py`](../python/sglang/srt/speculative/dflash_feature_tail.py)).
The next window preparation rebuilds only those rows from the live tail
([`dflash_cache.py`](../python/sglang/srt/speculative/dflash_cache.py)). This
replaces a per-prefill copy of the request's table rows and a device comparison
whose `.any()` waited for the prefill to finish.

**Remove prefill-path host synchronizations (common code).** Tracking-slot
indices stay on the device instead of being read back with `.item()`
([`schedule_batch.py`](../python/sglang/srt/managers/schedule_batch.py)). Host
lists upload through pinned memory instead of pageable `torch.tensor(...,
device=...)` copies in the request-slot mapping, recurrent tracking metadata,
and DFlash prefill lengths
([`memory_pool.py`](../python/sglang/srt/mem_cache/memory_pool.py),
[`dflash_worker_v2.py`](../python/sglang/srt/speculative/dflash_worker_v2.py)).
The FlashInfer GDN checkpoint plan reads its existing host mirrors instead of
copying device tensors back
([`gdn_flashinfer.py`](../python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py)).
Index clamping and slot clearing avoid scalar uploads
([`hybrid_linear_attn_backend.py`](../python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py),
[`unified_memory_pool.py`](../python/sglang/srt/mem_cache/unified_memory_pool.py)).

**Stage HiCache backups through the copy engine.** Page-first backups of
recurrent state now gather on the device into a two-slot staging buffer and then
use the existing batched copy-engine transfer
([`pool_host/mamba.py`](../python/sglang/srt/mem_cache/pool_host/mamba.py)). The
staged KV write-back now accepts strided source rows, so unified KV views use
it too ([`relayout.cuh`](../python/sglang/kernels/jit/csrc/kvcacheio/relayout.cuh),
[`staged_write_back.cuh`](../python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh),
[`hicache.py`](../python/sglang/kernels/ops/kvcache/hicache.py),
[`pool_host/mha.py`](../python/sglang/srt/mem_cache/pool_host/mha.py)). Before
this change, the direct transfer kernels ran for about 420 ms per rank per C32
run alongside model graphs. Loads and unstaged layouts keep the direct kernels.
This closed about 0.9 point at C32 and matched decode time per output token at C1.

**Release target pages without reading draft bindings back.** Releasing Full
pages used to filter which pages still held a draft SWA binding. That count
readback ran on the schedule stream, which the overlap scheduler fences behind
the in-flight forward. When that forward was a prefill, the scheduler stalled
for most of it and the GPU then idled for 10–18 ms while the CPU caught up.
Draft bindings now outlive the freed Full pages as spares inside the fixed draft
reservation
([`allocator/dflash_swa.py`](../python/sglang/srt/mem_cache/allocator/dflash_swa.py)).
This is safe because a binding never implies materialized content: every path
that brings a page into a window writes its rows, restores them from a raw tail,
or rebuilds them through the canonical-remap repair. Window updates recycle
spare bindings as before, and bindings are released when the cache is flushed.
The C32 gap fell from −1.93% to between −0.57% and −1.30%.

### Measured and not retained

| Change | Result |
| --- | --- |
| Pinned count readback in the release filter | Still waited for the fenced forward; no measurable change. Replaced by the binding change above. |
| One batched `_foreach_copy_` for decode feature appends and prefill chunk clones, recording each chunk's stream once | Halved `consume` CPU time at C32 (about 0.77 to 0.40 ms per step) but showed no measurable serving gain (C32 −0.73% to −1.21% across three comparisons). Reverted to keep the diff small, matching the 2026-10-21 multi-tensor copy result. |
| Skipping the per-iteration HiCache readiness consensus when no acknowledgment is outstanding (diagnostic) | Cut that step from about 124 to 25 µs without improving C1 time to first token. Not rank-safe if acknowledgment queues ever diverge. Not retained. |

### Remaining gap

At C32 the remaining cost is concentrated after prefills. The TP prefill
launch is CPU-bound for about 40 ms per batch, so the GPU waits for the CPU
afterwards. Flag on adds about 2–3 ms of CPU work to each of those windows
under Nsight, which leaves about 1.2 ms more GPU idle per prefill. The work is
HiCache write submission (about 0.6 ms), raw-tail donation (about 0.4 ms),
decode feature consumption (about 0.4 ms), checkpoint bookkeeping (about 0.4 ms),
and the readiness consensus (about 0.1 ms per iteration). Flag on also runs the
same prefill-launch code 1–2% slower, spread evenly across functions with
identical call counts. The cause is unknown; garbage collection was ruled out.

At C1 the remaining gap is almost entirely time to first token, about 2.4–2.9 ms.
It consists of the same per-request CPU items on the first-token path, plus
cross-rank launch skew in the CPU-bound TP2 prefill. Decode time per output
token is within 0.1% of flag off.

### Validation

Feature, allocator, recurrent-cache, GDN policy, and HiCache unit suites passed
(693 tests), as did 214 CUDA kernel tests for the staged write-back and Mamba
transfers. On the exact retained source, the TP2 serving reproducer
[`test_dflash_swa_cache.py`](../test/manual/spec/test_dflash_swa_cache.py) passed
on the default server and on the forced-host server with 131,072 target tokens.
Every output matched the SHA-256 recorded in
[`draft-swa-serving-validation.json`](draft-swa-serving-validation.json),
including the restore served entirely from host (4,096 host and zero device
tokens).

### Independent review reproduction, 2026-10-22

Fresh servers ran a full frozen copy of the current `python/` directory:
5,857 files, source digest
`f866ad8c07a81a98cd7b8486e14427a02f449efc7f9f66a742f61658c0452f86`.
Enabled used GPUs 4/5 and disabled used 6/7, both pinned to CPUs 48–95.
The configuration and workload above were unchanged. Each server/concurrency
had an excluded prime followed by three serial repeats in alternating order.
Tests, startup, and profiling did not overlap measured runs.

| Concurrency | Enabled total tokens/s | Disabled total tokens/s | Enabled versus disabled |
| --- | ---: | ---: | ---: |
| 1 | 3,882.87 | 3,902.01 | −0.49% |
| 32 | 20,911.85 | 21,029.17 | −0.56% |

These are medians. All measured request and input/output token counts matched.
C1 median TTFT was 53.156 versus 50.629 ms (+2.527 ms); mean time per output token
was 1.1850 versus 1.1836 ms (+0.12%). Acceptance length was 5.1512 for both C1
variants and 4.9893/4.9902 at C32. This same-node reproduction supports the
latest gap estimate; it does not repeat the GPU-assignment crossover or establish
parity across launches or other workloads.

The independent correctness run passed **680 unit tests**, **214 CUDA transfer
tests**, and **five GPU host-tail tests**; 59 unit cases were skipped. Default
and fixed-capacity TP2 serving checks covered warm reuse, 2,400-token decode,
and 32 branches. The fixed-capacity run used `--max-total-tokens 131072` and
20 pressure requests and restored 4,096 host-cached tokens with zero
device-cached tokens. All five recorded output hashes in each serving run
matched `draft-swa-serving-validation.json`. This validation overlaps earlier
rounds; do not add their counts into a unique-test total.

Compact evidence is in `independent_review_reproduction` in
[draft-swa-parity-pass.json](draft-swa-parity-pass.json). Full launch commands,
logs, benchmark JSONL files, serving outputs, the frozen source, and the driver
are under `/tmp/dflash-review-repro-20261022/`. `results.json` records completion;
`final-verification.json` confirms source/test hashes, benchmark artifact hashes,
HEAD, index, and Git status were unchanged during reproduction. All owned
servers were stopped. The subsequent documentation reconciliation changes only
this report, the handoff, and the compact evidence JSON.

### HiCache staging memory accounting, 2026-10-22

The staged backups allocate persistent GPU scratch outside the unified cache
arena. Each transfer gathers scattered rows into that scratch, then the copy
engine moves the packed payload to pinned host memory. Larger backups reuse
the buffers in chunks; their capacity does not scale with the number of active
requests or cached checkpoints.

For the measured Qwen3.6-35B-A3B configuration at TP2, the tensor payload is:

| Staging buffer | Geometry per rank | GPU bytes | MiB |
| --- | --- | ---: | ---: |
| Recurrent temporal state | 2 slots × 30 layers × 16 value heads × 128 × 128 × FP32 | 62,914,560 | 60.00000 |
| Recurrent convolution state | 2 slots × 30 layers × 4,096 channels × 3 history rows × BF16 | 1,474,560 | 1.40625 |
| Target K/V | 64 pages × 64 tokens × 10 layers × K/V pair × 1 KV head × 256 × BF16 | 41,943,040 | 40.00000 |
| Total | Reused scratch for one TP rank | **106,332,160** | **101.40625** |

These sizes follow `_STAGED_BACKUP_SLOTS = 2` in
[`pool_host/mamba.py`](../python/sglang/srt/mem_cache/pool_host/mamba.py),
`_WRITE_BACK_STAGING_PAGE_CHUNK = 64` in
[`pool_host/base.py`](../python/sglang/srt/mem_cache/pool_host/base.py), and the
model's state/KV geometry. The two recurrent slots are the transfer chunk
capacity, independent of a request's ping-pong checkpoint candidates. The
figures exclude allocator rounding and small metadata tensors. Raw tails and
materialized draft KV keep their separate budgets below.

The review found that the allocation order left this scratch unpriced:

1. `KVCacheConfigurator._profile_available_bytes` profiles free memory and
   subtracts runtime headroom and the raw-feature reserve before sizing pools.
2. The scheduler finishes `init_model_worker`, then `build_kv_cache` constructs
   the HiCache host pools. Their constructors allocate the staging buffers.
3. The newly enabled strided MHA path and the recurrent path therefore consume
   additional GPU memory after this configuration's cache budget was set.

There is an existing `prepare_hicache_staging` hook before post-capture resizing,
but it prepares MHA buffers only. More importantly, `post_capture_kv_sizing_planned`
currently excludes unified Mamba and unified DFlash configurations, so that
hook does not run in this reproduction. Extending only that hook would not
resolve accounting for this setup. See
[`kv_cache_builder.py`](../python/sglang/srt/mem_cache/kv_cache_builder.py),
[`overrides.py`](../python/sglang/srt/arg_groups/overrides.py), and
[`scheduler.py`](../python/sglang/srt/managers/scheduler.py).

The cleanup now derives the eligible staging shapes through helpers shared with
the host-pool allocators. `KVCacheConfigurator._dflash_hicache_staging_bytes`
uses the target's existing Full/Mamba sub-pool specifications, including local
layer counts and storage dtypes. `_profile_available_bytes` subtracts the
result alongside the raw-feature reserve before sizing the bounded DFlash pools.
Kernel eligibility is resolved before profiling free memory.

This reservation is specific to the bounded DFlash path, which already requires
pre-capture sizing. Other sizing paths retain their existing behavior; MHA
post-capture preparation still provides resident staging buffers to the host
pool without an additional reservation. No new staging owner, runtime flag,
transfer capacity, or kernel was added. Disabled HiCache, other layouts, and
unavailable transfer kernels receive no staging charge. Empty recurrent state
and unsupported row widths keep their existing fallbacks.

The initial solve reserves the maximum 64-page KV transfer chunk because host
capacity is determined later. A host pool with fewer pages allocates less, so
the reservation is a conservative upper bound in that case. For the measured
geometry it is exactly **106,332,160 bytes per rank**, equivalent to 10,384
Full-KV tokens at 10,240 bytes/token before page and allocator rounding. A lower
explicit token cap may absorb the charge without changing capacity. The raw GPU
reserve, materialized draft budget, and host-tail capacity are separate.

This corrects the bounded path's memory budget. No OOM was observed, and it is
not an established cause of the remaining throughput gap. Historical throughput
results above used the earlier budget and must not be relabeled as measurements
of this cleanup.

Validation passed **143 unit tests and 214 CUDA transfer tests**. The new initial
budget regression fails against the frozen pre-cleanup source by exactly
106,332,160 bytes and passes with the fix. Tests compare the actual staging
allocation shapes with the reservation, exercise empty/unaligned state and
unavailable-kernel fallbacks, preserve the flag-off budget, and check reuse of
preallocated MHA buffers. Existing post-capture and capped-pool sizing tests
also passed.

Fresh TP2 serving checks passed for automatic sizing and a 131,072-token cap,
including 2,400-token decode, prefix reuse, and 32 branches. All five output
hashes in each run matched the existing serving reference. Forced recovery
restored 4,096 tokens from host with zero device-cached tokens, then extended
the 37-token suffix. The automatic-size launch had **4,462,464 Full-KV tokens**,
versus 4,472,832 in the earlier reproduction; these are separate free-memory
profiles. The capped launch retained exactly **131,072 tokens** and the same
10,506,895,360-byte unified arena as before.

Evidence is in `staging_memory_accounting_cleanup` in
[draft-swa-parity-pass.json](draft-swa-parity-pass.json), with logs, source copies,
and the driver under `/tmp/dflash-staging-cleanup-20261022/`. A harness port-bind
collision after the successful default run required retrying the capped launch
on a fresh port; it was not a server or serving-test failure. All owned servers
were stopped, the four edited source/test files stayed fixed during validation,
and HEAD plus the checkpoint index were preserved. The minimization pass kept
the fix in three existing runtime modules and one existing test module.

## Status on 2026-10-21

Historical sections below describe their dated code states and controls. In
particular, asynchronous request-map uploads, retained spare draft bindings,
and staged recurrent backups were later adopted together in the parity pass.

**Parity is not achieved yet.** The completed flag-on/off GPU-swap comparison of
retained code measured **1.03% lower total throughput at concurrency 1** and
**3.46% lower at concurrency 32**, combining ratios from the same GPU pairs.
The original large regression has been reduced, but the remaining gap should
not be described as noise or attributed entirely to raw-feature bandwidth.
Optimization was paused for user review after this comparison; the later
parity pass and core minimization supersede that status.
Earlier comparisons remain separate below.

Optimization resumed after the intermediate checkpoint `4f895d40ed`. Its
accounting fix passed serving checks and two further fixed-cap comparisons:
concurrency 1 measured 3,837 enabled versus 3,885 off (−1.25%), and concurrency
32 measured 19,407 versus 20,094 (−3.42%). These later runs used GPUs 0/1 enabled
and 6/7 disabled, with local CPU affinity; the earlier three-repeat table below
remains a separate series. The interrupted automatic-sizing run was excluded;
the new comparison below restarted and primed fresh servers.

The feature still preserves checkpoint recovery, keeps materialized draft KV,
and uses the existing unified allocator and HiCache transfer engine. It remains
opt-in through `--enable-dflash-swa-cache`.

## Tracking-slot bookkeeping optimization

Lazy recurrent tracking previously read GPU slot indices to decide whether a
position was allocated and used GPU boolean filtering during cleanup. Clearing
a slot with a Python scalar also caused a blocking host-to-device copy. The
retained change maintains allocation bits with the CPU allocation/release
decisions, uses them for planning and cleanup, and clears device slots with
`fill_`. The existing device mappings and scheduler-to-forward stream dependency
remain authoritative for GPU execution. Allocation, replacement, donation,
release, retraction, and session cleanup preserve the ownership lifecycle.

The runtime change spans four existing files and adds no flags, dependencies,
transfer streams, or kernel APIs. This common recurrent-bookkeeping improvement
also applies with the DFlash flag disabled, so the comparison includes a fresh
disabled server with the same change.

Four configurations used the same automatic sizing and serving workload: original
enabled on GPUs 0/1, original disabled on 2/3, updated enabled on 4/5, and updated
disabled on 6/7, all with local CPU affinity. Every server completed separate
C1/C32 primes. Two serial measured repeats reversed configuration order. There
were no profilers, diagnostic pool counters, tests, or model startups during
those measurements.

| Concurrency | Original on, total tokens/s | Updated on | On improvement | Updated off | Updated on versus off |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,830 | 3,856 | +0.69% | 3,885 | −0.76% |
| 32 | 19,524 | 19,637 | +0.58% | 20,207 | −2.82% |

The original disabled medians were 3,871 and 20,194; its updated path improves
by 0.36% and 0.07%, respectively. All request and token counts were correct.
This is a small improvement, with different GPU pairs and only two repeats;
it does not establish parity. The retained runtime matches the measured
prototype's AST except for a slot-value parameter type annotation. The JSON
preserves source hashes, individual results, acceptance lengths, and commands.

Validation passed **137 CPU tests with 258 subtests**, **13 accelerator tests
with 2 subtests**, and TP2 warm reuse, long decode, and 32 branches. A separate
131,072-token-pool run forced eviction and restored 4,096 prefix tokens from
host while computing only the 37-token suffix. New tests fail on the previous
code at the actual scalar-read, boolean-indexing, and scalar-upload operations.
They also cover pending allocations before confirmation, allocation failure,
empty tracking positions, retained donations, and cleanup. No timing threshold
is used as a correctness assertion.

A fresh C32 Nsight capture confirms zero CUDA stream synchronizations in the
three instrumented slot-planning/result/cleanup methods on both ranks. Those
methods still execute in the trace. Waiting shifts partly to later operations:
rank-0 draft-page release waits grow from about 12 to 81 ms, while the large
prefill residency check still waits about 446 ms. These are diagnostic traces
on different GPU pairs, not an additional throughput comparison. The follow-up
below distinguishes these CPU waits from inactivity on the model's CUDA stream.

### Follow-up on the remaining mapping checks

A temporary CPU prefix-remap version skipped the prefill GPU mapping check when
insertion could not have changed the request's Full IDs. New leaves adopt the
request's IDs; only an overlapping prefix can replace them. Exact GPU mapping
comparison and draft-content repair remained in place for possible overlaps.
This repeats an earlier experiment after the tracking-slot optimization, using
current source and new measurements.

The candidate ran on GPUs 0/1, retained enabled code on 4/5, and current disabled
code on 6/7, all with local CPU affinity and automatic sizing. Separate C1/C32
primes were excluded. Two serial repeats reversed configuration order; no
profiling, tests, or model startup overlapped the measurements.

| Concurrency | Prefix guard, total tokens/s | Retained on | Flag off | Guard versus retained |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3,846 | 3,857 | 3,887 | −0.29% |
| 32 | 19,523 | 19,596 | 20,200 | −0.37% |

The guard is **not retained**. Both repeats at each concurrency favored the
existing code. Different GPU pairs and two repeats limit attribution, but this
comparison provides no serving benefit to justify the additional version state.
Every run completed its expected request and input/output token counts.

Before timing, temporary CPU checks exercised real tree insertion with fresh,
duplicate, repeated, and partially shared prefixes, plus draft-content repair
with page sizes 1 and 4. Unchanged prefixes performed no host-visible mapping
decision. TP2 warm reuse, long decode, and 32 branches passed; this automatic-size
run did not force host eviction and is not an additional host-recovery test.

The existing Nsight traces were then reanalyzed using activity on the CUDA stream
that executes the model graphs. This distinguishes model inactivity from a CPU
wait while the GPU is still computing. For retained code at C32, rank 0's
17,216 ms forward span contains 1,692 ms of model-stream inactivity, versus
1,249 ms with no GPU stream active. Selected CPU ranges contain:

| CPU range | CPU duration, ms | Model stream inactive, ms | All GPU streams inactive, ms |
| --- | ---: | ---: | ---: |
| Draft-page release | 146.8 | 13.7 | 5.0 |
| Prefill residency, batch size 3 | 482.4 | 29.2 | 29.2 |
| Raw-tail consumption | 506.2 | 68.3 | 61.6 |
| Checkpoint donation preparation | 127.2 | 82.0 | 12.2 |
| Batch preparation for prefill | 213.5 | 164.4 | 73.9 |

Rank 1 shows the same distinction: about 15.2 ms of model inactivity during
draft-page release and 27.3 ms during the batch-size-3 residency check. Before
the tracking-slot optimization, rank 0 already had 13.2 and 29.2 ms, respectively,
in those ranges. The increase in cleanup's CPU synchronization time therefore
does not establish a comparable increase in model inactivity.

These nested ranges are not additive, and inactivity during a CPU range does
not prove that the range caused it. Whole-graph tracing also counts internal
graph waits as activity. No sum of the table is a predicted throughput gain.
The prepared retain-and-rebind page-ownership experiment was not rerun: the
81 ms cleanup wait alone is insufficient justification for changing ownership.
The next investigation should follow the dependencies in batch preparation and
checkpoint donation, and the host submission work in raw-tail consumption.
The earlier direct-capture prototype still needs its queued-allocation memory
bound and stream/lifetime proof before it could be adopted.

Artifacts are `prefix-version-results.json`, `prefix-version-property.json`,
`prefix-version-serving.json`, and both automatic-size
`*.1.compute-stream.json` trace analyses under `/tmp/dflash-parity/`.
The accompanying report JSON preserves measurements, source hashes, and the
selected trace ranges. The pushed checkpoint and its staged index are unchanged.

### Direct capture with a bounded destination reserve

The direct-capture prototype was retested after deriving a conservative bound
for destinations allocated ahead of CPU commit metadata. With the current fixed
DFlash block reservation, the allocated end is at most two blocks plus page
rounding beyond the CPU committed end. Rounding that span to independent raw
chunks requires at most
`ceil_align(2 * block_size + page_size - 1 + chunk_size - 1, chunk_size)`
destination rows per request. Overlap reserves two queued capture sets and one
active uncommitted set, including captures whose request has closed. Their
storage can overlap, but the reserve does not assume it does.

For the measured configuration, this raises raw GPU headroom from **9.743 to
10.586 GiB per rank**, an extra **0.844 GiB**. Automatic sizing reduces Full KV
capacity from 4,472,832 to 4,384,384 tokens. This is a prototype reservation;
production keeps its existing memory budget.

Exact GPU checks passed 17 cases, including 480 pairs of queued overlapping
captures, immutable checkpoints, canceled/replaced owners, partial chunks,
rejected rows, noncontiguous sources, and FP32/BF16 features. A separate
20,706-case arithmetic check covered destination alignment. TP2 warm prefix
reuse, long decode, and 32 branches passed. This retest did not force host
eviction; it is not new evidence for the prototype's host-recovery path.

Six separate C1/C32 primes were excluded. Two serial repeats reversed variant
order, with no profiling or pool counters. All expected request and token counts
passed. The candidate used GPUs 0/1; retained on and off used 4/5 and 6/7.

| Concurrency | Direct capture, total tokens/s | Retained on | Off | Candidate versus retained | Retained versus off |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,838 | 3,856 | 3,887 | −0.48% | −0.79% |
| 32 | 19,558 | 19,596 | 20,191 | −0.19% | −2.95% |

Both repeats favored the retained implementation at each concurrency. Different
GPU pairs and two repeats limit causal attribution, but there is no measured
benefit to justify extra code and memory. The candidate was rejected and its
server stopped. Production runtime and tests are unchanged by this experiment.
Artifacts are `direct-reserve-results.json`, `direct-capture-reserve-bound.json`,
`direct-capture-reserve-correctness.json`, and `direct-reserve-serving.json` in
`/tmp/dflash-parity/`.

### Follow the prefill dependencies

Fresh matched C32 Nsight captures instrumented scheduler admission, request/state
allocation, forward-input preparation, and result processing on retained code.
Both execute 765 measured forward ranges. Rank 0's model-stream inactivity is
1,512 ms enabled versus 1,175 ms off, across forward spans of 17,230 and 16,554 ms.
These instrumented durations are not throughput measurements.

| CPU range | Model stream inactive on, ms | Model stream inactive off, ms |
| --- | ---: | ---: |
| Prefill result processing | 157.1 | 61.3 |
| Prefill allocation | 115.4 | 70.9 |
| Request/state slot allocation | 66.2 | 35.9 |
| Recurrent slot allocation, all phases | 59.7 | 59.4 |
| Resolve CPU sequence lengths | 1.6 | 1.7 |

These nested ranges are not additive. Sequence-length resolution and total
recurrent slot allocation do not explain the additional idle time. The
instrumented HiCache transfer-wait method does not execute in this workload.

Small-copy analysis also subtracted observed preceding stream work and matched
CUDA event dependencies from enqueue-to-start delay. For result D2H copies,
rank 0 has 7.9 ms of remaining delay enabled versus 6.3 ms off; neither has an
individual remaining delay above 0.011 ms. Most of their apparent queue time
belongs to unfinished model work. A new mapped-host result-copy kernel is not
justified by that evidence. Scheduling-stream scalar readbacks still show about
83 ms of remaining delay, mainly in draft-page release, whose model inactivity
is only about 15 ms. Removing that entire CPU wait would not imply an 83 ms
serving gain.

The reconstruction resolves 4,919 event waits per enabled rank and leaves 256
unresolved, so the remaining delay is an upper bound relative to known
dependencies, not proof of copy-engine contention. Off resolves all 4,519 waits.
Profiles and analyses are `*dependency_profile_whole_local.1.*`,
`small-copy-queues.json`, and `small-copy-readiness.json` under
`/tmp/dflash-parity/`. The next investigation follows prefill allocation and
donation dependencies rather than introducing a result-copy kernel.

The allocation trace identifies a smaller concrete issue: each of 83
`HybridReqToTokenPool.alloc` calls performs two synchronous H2D uploads when
Python-list request indices publish the recurrent and tracking-slot maps.
Rank 0 spends 29.6 ms in those 166 stream synchronizations enabled versus
2.64 ms off; rank 1 shows 27.6 versus 2.59 ms. A temporary prototype uploads
the index list once with the existing `async_h2d` helper and shares the device
indices between both maps. It adds no kernel, cache state, or memory reservation.
CUDA synchronization-debug mode fails the old code at the implicit index upload.
The prototype passed ten CPU/CUDA cases covering fresh allocation, continuation,
allocation failure, and row recycling. TP2 warm reuse, long decode, and 32
branches also passed. This experiment did not add a forced host-recovery check.

The throughput comparison applied the change to both enabled and disabled
servers. It used two reversed serial repeats per concurrency, then exchanged
the candidate and control GPU pairs and repeated the measurements. All 32
measured runs completed the expected request and token counts; separate primes
were excluded. The source manifests matched across the GPU swap. No profiling,
tests, or server startup overlapped the measurements.

| Concurrency | Candidate change, flag on | Candidate change, flag off | Retained on versus retained off |
| --- | ---: | ---: | ---: |
| 1 | −0.091% | −0.089% | −0.96% |
| 32 | −0.009% | +0.019% | −3.11% |

Candidate changes are geometric means of candidate/control ratios measured on
the same GPU pair across the two assignments. This reduces stable hardware
differences but does not eliminate temporal variation. Before the swap, the
candidate appeared about 0.44% slower with the flag on; after the swap it was
0.27% faster at C1 and 0.43% faster at C32. The combined result provides no useful
gain, so the prototype was rejected at that time. Its two servers were stopped.
The asynchronous upload is now retained as part of the combined prefill
synchronization changes in the 2026-10-22 parity pass; the isolated historical
result does not measure that combined implementation.

Evidence is in `async-reqmap-results.json`, `async-reqmap-cross-results.json`,
and `async-reqmap-crossover-summary.json` under `/tmp/dflash-parity/`.

### Queue raw backups until request cleanup finishes

A temporary prototype queued complete raw-tail backups until the existing
end-of-batch backup flush. It passed correctness checks but was rejected after
measurement. The retained implementation starts raw D2H during donation,
before subsequent request cleanup finishes. The experiment submitted immutable
sources together before the ordinary HiCache backup flush, to test whether
later submission reduced interference with cleanup and transfer overhead.

Queued and submitted sources share the existing 64-lease limit at 32 requests.
Saturation flushes queued sources before waiting for capacity. Producer events
protect delayed captures, and the existing host-slot fence protects earlier
readers. Cancellation drops an unsubmitted source without losing that fence.
A queued complete tail remains matchable: restore submits its backup before
ordering the H2D copy. The prototype adds no GPU or host reservation and preserves
every live checkpoint payload. The automatically sized Full capacity remains
4,472,832 tokens.

Validation passed five new CPU cases, 42 existing CPU cases with 81 subtests,
three new CUDA cases, and five existing CUDA cases. Separate negative controls
remove only the producer fence or only the prior-reader fence; both fail with
incorrect restored values. The checks cover on-demand restore, queued/in-flight
capacity, cancellation, slot reuse, partial submission failure, and delayed
producers. TP2 warm reuse, 2,400-token decode reuse, and 32 branches passed.
With 131,072 target tokens and 20 pressure requests, recovery reported 4,096
host-cached tokens, zero device-cached tokens, and the 37-token new prompt suffix.

The automatic-sizing comparison completed 24 measured runs across two GPU
assignments, with separate excluded primes and two reversed serial repeats
per assignment. The enabled candidate and retained control exchanged GPUs 0/1
and 4/5; the disabled control stayed on GPUs 2/3. Source manifests matched
across the swap. Tests and the fixed-capacity server finished before timing.

| Concurrency | Candidate change on GPUs 0/1 | Candidate change on GPUs 4/5 | Geometric mean change |
| --- | ---: | ---: | ---: |
| 1 | −0.001% | +0.041% | +0.020% |
| 32 | −0.269% | −0.352% | −0.310% |

The first assignment alone suggested gains of 0.26% at C1 and 0.05% at C32.
The same-GPU comparison shows no useful C1 improvement and a C32 regression
on both pairs. The queue adds ownership machinery without a measured benefit,
so it is not retained. Its servers were stopped. Source, validation,
negative-control, serving, and throughput artifacts use the `queued-raw-`
prefix under `/tmp/dflash-parity/`.

### Check the remaining synchronization costs

A broader pass over the saved matched dependency profiles attributes blocking
CUDA runtime calls by their innermost CPU range. Rank 0 spends 5,656 ms enabled
and 6,810 ms disabled waiting for decode-result completion. Prefill waits also
move between the common forward code and the enabled residency checks. These
are waits for unfinished work, not independent costs that can be summed as
potential gains. The other enabled stream waits concentrate in the already
examined draft-page release, request-map uploads, and batch preparation.

The analysis identifies no new large upload bottleneck that justifies another
small index-conversion change. It is recorded in `dependency-synchronization.json`.
The completed comparison below exchanges flag-on and flag-off GPU assignments
using only retained code. This checks the overall gap separately from the
candidate/control comparisons above, which exchanged two enabled implementations.

### Retained flag-on/off GPU-swap comparison

The retained enabled and disabled configurations exchanged GPUs 2/3 and 4/5.
The first assignment reuses their qualified control measurements from the
queued-backup crossover; the candidate measurements are excluded. The second
assignment uses fresh servers with source manifests identical to those controls.
Neither configuration includes the rejected mapping-upload or queued-backup
prototype. Both include the retained CPU tracking-slot optimization.

Each assignment has two reversed serial repeats per flag and concurrency,
with separate priming runs excluded. All 16 selected measurements completed
the expected requests and input/output token counts. Model startup, profiling,
tests, and diagnostic pool counters were excluded from timing. Automatic sizing,
TP2, 32 maximum active requests, and 256 configured recurrent slots were retained;
flag on uses HiCache write-through, while flag off does not use HiCache.

| Concurrency | GPU pair | Flag on, total tokens/s | Flag off, total tokens/s | On versus off |
| --- | --- | ---: | ---: | ---: |
| 1 | 2/3 | 3,843 | 3,882 | −1.00% |
| 1 | 4/5 | 3,850 | 3,891 | −1.06% |
| 32 | 2/3 | 19,463 | 20,198 | −3.64% |
| 32 | 4/5 | 19,604 | 20,269 | −3.28% |

The geometric mean of the same-GPU ratios is **−1.03% at C1** and **−3.46% at
C32**. Swapping assignments removes a fixed GPU-pair advantage from this
comparison; two repetitions per assignment do not eliminate temporal variation
or establish a confidence interval. Both GPU pairs show a remaining gap at
both concurrency levels. This is total input-plus-output throughput for the
documented workload, not a claim about every workload or concurrency.

The comparison completed, and optimization was then paused at the user's
request. No prototype was adopted from this comparison. Artifacts are
`retained-flag-crossover-results.json`, `retained-flag-crossover-pipeline.log`,
the referenced `queued-raw-cross-results.json`, and their per-run JSONL files
under `/tmp/dflash-parity/`. The accompanying report JSON preserves compact
measurements and artifact hashes.

## Automatic-sizing comparison

The same workload and serving settings described below were tested after
removing `--max-total-tokens 131072`. Both configurations retained 32 maximum
requests, 256 configured recurrent-cache slots, and static memory fraction 0.8.
This changes cache capacity; it is not an additional production optimization.

Fresh servers ran enabled/write-through on GPUs 4/5, disabled on 2/3, and
enabled/write-back on 0/1, each with local CPU affinity. All startup and serving
checks completed before timing. Separate C32 and C1 priming runs are excluded;
three serial repeats at each concurrency rotated configuration order and used
the same four benchmark warmups. No profiler, model startup, or tests ran
during the measured comparisons.

| Concurrency | Flag off, total tokens/s | On, write-through | On versus off | On, write-back | Write-back versus off |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,871 | 3,848 | −0.58% | 3,838 | −0.84% |
| 32 | 20,138 | 19,563 | −2.86% | 19,546 | −2.94% |

Write-back did not improve either median and is not adopted as a new default.
Every measured run completed its expected input and output token counts. C1
acceptance length was 5.1512 throughout. At C32 it was 4.9893 for both enabled
policies and 4.9878–4.9901 for the disabled runs; these runs do not establish
identical verification schedules.

The enabled startup Full-token budget was 4,472,832, versus the disabled
profiled budget of 2,538,432. The latter is not a hard capacity limit: the
existing allocator can lend unused recurrent-state space to Full KV. Enabled
admission still reserves its draft and recurrent budgets independently. Raw GPU
and host reservations are unchanged from the memory section below.

Both enabled policies passed warm reuse, long decode, and 32 concurrent
branches. These automatic-sizing checks used zero pressure requests and did
not force host-only restoration; the separate fixed-cap validation covers that.
The JSON's `automatic_sizing_comparison` entry contains individual results,
launch commands, hashes, and source-equivalence checks against this checkpoint.

### Host-tail capacity experiment

A separate experiment doubled only raw-host capacity from 256 to 512 windows,
including the corresponding host-budget accounting. GPU recurrent capacity
stayed at 256; raw GPU and materialized draft reservations were unchanged.
The extra host allocation is **12 GiB per rank**, taking it from 12 to 24 GiB.
These changes exist only in temporary source snapshots.

Two serial repeats, reversing configuration order, used the same automatic-sizing
workload and four benchmark warmups. Separate C1 and C32 primes are excluded.
The enabled 256-window control ran on GPUs 0/1, the 512-window candidate on 6/7,
and flag off on 2/3, all with local CPU affinity. A read-only `/server_info`
addition exposed pool counts. The harness saved an additional count query after
each enabled benchmark; the benchmark's existing final metrics query also sees
these fields and is included in its reported duration. Both enabled variants
have this instrumentation; flag off does not.

| Concurrency | Flag off, total tokens/s | On, 256 host windows | On, 512 host windows | 512 versus 256 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3,871 | 3,835 | 3,835 | +0.02% |
| 32 | 20,187 | 19,546 | 19,457 | −0.45% |

At the end of both C32 repeats, the control had all 256 raw-host slots occupied,
189 GPU recurrent checkpoints cached, and 67 recurrent slots free. The candidate
retained 315 host tails and 222 GPU recurrent checkpoints, with 34 recurrent slots
free. Host tails can outnumber GPU checkpoints because recurrent states can be
host-resident. At C1, both retained only 64 tails and 64 GPU checkpoints.

The larger pool retains more checkpoints but does not establish a throughput
benefit on this predominantly cold-prefix workload; it is not adopted. End-of-run
free slots do not rule out transient recurrent pressure. The two enabled variants
also had slightly different C32 acceptance lengths, 4.9893 versus 4.9901, and ran
on different GPU pairs. All token counts were correct. The candidate passed warm
reuse, long decode, and 32 branches; this check did not force host-only recovery.
The JSON's `raw_host_capacity_experiment` entry preserves the individual runs,
occupancy, source hashes, and validation artifacts.

### Recurrent-capacity and write-policy experiment

A subsequent comparison held raw-host capacity at 256 windows and increased
GPU recurrent capacity from 256 to 512 slots. Holding host capacity fixed
requires a temporary source patch; this is not an existing CLI configuration.
The extra 256 recurrent slots reserve **7.676 GiB per rank**, reducing the
enabled Full-token cap from 4,472,832 to 3,667,968. Raw GPU and materialized
draft reservations remain unchanged.

Two serial repeats reversed configuration order after separate C1/C32 primes.
The original enabled control used GPUs 0/1, the 512-slot write-through variant
4/5, the 512-slot write-back variant 6/7, and flag off 2/3. All used local CPU
affinity. The read-only counter instrumentation and timing caveat from the
host-tail experiment apply here too.

| Concurrency | Flag off, total tokens/s | On, 256 slots/write-through | On, 512 slots/write-through | On, 512 slots/write-back | 512/write-back versus off |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,870 | 3,835 | 3,834 | 3,846 | −0.60% |
| 32 | 20,153 | 19,539 | 19,594 | 19,847 | −1.52% |

Increasing recurrent capacity with write-through improves the C32 median by
0.28% over the enabled control. Combining that capacity with write-back improves
it by 1.57%, but does not reach parity. Neither change is adopted as a new
default. Different GPU pairs and small differences in acceptance schedules
limit causal attribution from these timings alone.

At C32 completion, both 512-slot variants retained 256 GPU recurrent checkpoints
and 256 raw-host tails, with 256 recurrent slots free. The control retained 189
GPU checkpoints and 256 tails, with 67 recurrent slots free. All measured token
counts were correct. Both candidates passed warm reuse, long decode, and 32
branches; these checks did not force host-only recovery. Individual measurements
and commands are preserved under `recurrent_capacity_experiment` in the JSON.

### Inference-stream idle time

Reanalysis of the existing fixed-cap Nsight captures separates the stream
running model graphs from background transfers. At C32 on TP rank 0, model-stream
idle time is 1,734 ms enabled versus 1,359 ms disabled, across forward spans of
17,208 and 16,665 ms. Measuring the union of all GPU activity instead gives
1,208 versus 1,344 ms: background copies conceal some inference idle time.
Graph durations include internal waits, so these totals are not independent,
additive estimates of removable overhead.

Several large enabled gaps contain repeated CUDA stream synchronizations during
request-result processing. A fresh automatic-sizing C32 capture with temporary
markers attributes 205 ms of rank-0 stream waits to lazy checkpoint bookkeeping
and 70 ms to recurrent-slot release. Of the former, approximately 122 ms follows
host-to-device scalar writes and 84 ms follows device-to-host scalar reads.
Clearing a slot with Python `-1` uses a blocking scalar copy; the device-side
`fill_` operation avoids that copy.

The retained tracking-slot change above addresses these reads and writes. Its
ownership-only predecessor passed tests and serving checks, but its priming
series was stopped before measured repeats to include the scalar-write fix.
Only the combined implementation has measured comparison results. Profile
durations overlap other work and do not directly predict an unprofiled
throughput gain.

## Fixed-cap throughput comparison

Each server used two H200 GPUs, the local Qwen3.6-35B-A3B target snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, and `/tmp/autospec/baseline` as the draft.
Both configurations used BF16, FA3, unified memory, CUDA graphs, 32 maximum active
requests, `--max-total-tokens 131072`, `--max-mamba-cache-size 256`, page size 64,
draft block size 16, and 4,096-token chunked prefill. The draft window is 2,048 tokens.

Flag off used **HiCache off and Rust TreeCore**. Flag on used **write-through
HiCache, ratio 2, and Python TreeCore**. This is the requested end-to-end comparison;
the cost of enabling the required HiCache path is included in the enabled result.
An attempted flag-off/HiCache-on control cannot start with this model's current
unified fused-draft configuration and is excluded.

The benchmark served fixed-length random token-ID prompts: 2,048 input tokens
and 512 generated tokens, seed 20261021, greedy decoding, ignoring EOS. Each run
flushed the cache, performed four warmup requests, then measured 32 requests at
concurrency 1 or 128 requests at concurrency 32. Warmups use the first benchmark
prompt with a shorter output; this is predominantly a cold-prefix workload, not
a claim of zero prefix reuse. The latest series used a fresh enabled server
and a warmed disabled control, with three repeats at each concurrency. The
enabled configuration ran first in the first repeat; the disabled configuration
ran first in the next two. Runs were serial, with no active profiler and no
concurrent model startup or tests. Initial priming measurements are excluded.

| Concurrency | Flag off, median total tokens/s | Flag on, median total tokens/s | On versus off |
| --- | ---: | ---: | ---: |
| 1 | 3,889 | 3,850 | −0.99% |
| 32 | 20,147 | 19,497 | −3.23% |

Across three runs, C1 ranges were 3,886–3,891 off and 3,846–3,851 on. C32 ranges
were 20,120–20,164 off and 19,497–19,501 on. These are **total input plus output
tokens/s**. Output-only throughput is one fifth for this workload. Every run
completed all requests with the requested token counts.

The enabled server used GPUs 4/5 and the disabled server GPUs 6/7, both with
local CPU affinity from startup. An earlier paired comparison on GPUs 0/1
and 2/3 established the stream-serialization improvement described below.
Acceptance length was 5.1512 for C1 and 4.9906 for C32 in the final series.

The [measurement JSON](draft-swa-throughput-optimization.json) contains individual
runs, exact launch commands, source hashes, output hashes, correctness results,
and paths to the full local artifacts. These measurements include the admission
correction and shared backup stream described below; earlier results remain
archived in the JSON. These measurements precede the automatic-sizing accounting
fix described below; they are not a new measurement of that fix. The disabled
process predates the final flag-gated stream
binding; its executed flag-off path is unchanged, and the previous builder source
is archived alongside its other source hashes.

**Matching sizing arguments do not make these allocators equivalent in capacity.**
The enabled path enforces separate Full/SWA/recurrent admission caps. Flag off's
existing unified allocator can lend unused recurrent-state space to Full KV.
Startup logs report an 11,194,597,376-byte arena off and a 10,506,895,360-byte
arena on; these exclude separately allocated raw features. Thus this comparison
includes different cache-pressure behavior, not just the marginal cost of raw
capture and offload. The enabled Full cap is 131,072 tokens. The off allocator
can exceed that count as the other end of its arena permits; its unused Full
capacity must not be presented as a second fixed cap.

The 128-request C32 workload can retain more than 131,072 unique prefix tokens
across completed requests. Its reported gap is a valid result for these serving
configurations, but does not isolate equal-capacity allocator overhead. This
benchmark also does not measure the benefit of avoiding historical target replay
after a prefix hit or draft update.

### NUMA-local repeat

The host has two NUMA nodes: GPUs 0–3 are local to CPUs 0–47, and GPUs 4–7 to
CPUs 48–95. Automatic binding was unavailable because `numactl` was absent.
A second comparison launched both configurations with local CPU affinity from
startup, using `taskset`. `/proc/<pid>/numa_maps` confirmed that both enabled TP
ranks' raw, target, and recurrent host pools were on their local NUMA node.
The workload and other serving arguments were unchanged. An initial C32 run
primed each server, followed by three serial repeats at each concurrency.

| Concurrency | Flag off, median total tokens/s | Flag on, median total tokens/s | On versus off |
| --- | ---: | ---: | ---: |
| 1 | 3,885 | 3,837 | −1.23% |
| 32 | 20,075 | 19,293 | −3.89% |

This confirms that locality alone does not close the remaining gap. These
controls use the same pre-correction implementation; no production affinity policy
or new tuning option was added.

### Workload controls

A single wave of 32 requests with the original 2,048-input/512-output lengths
fits within the enabled Full cap, including all its unique prefixes. Two runs
measured 14,913–15,011 total tokens/s enabled versus 15,452–15,544 disabled,
about a 3.5% median gap. Thus the different capacity policy does not explain
the whole regression.

A longer decode control used 32 requests with 256 input and 3,072 output tokens,
also within that Full cap. Its first run on each server included substantial
shape warmup and is excluded. The next pair measured 9,098 enabled versus 9,244
disabled total tokens/s, a 1.6% gap, with equal acceptance length of 10.0057.
This is one primed pair, not a replacement for the repeated main comparison.

An earlier 2,048-input/4,096-output control exceeded the enabled Full cap at
32 active requests. It produced different admission and acceptance behavior
(acceptance lengths 8.5601 enabled and 6.9237 disabled). Its apparently faster
enabled result is excluded from steady-decode cost attribution. All these
workload controls preceded the admission correction below.

## Retained implementation changes

### Provision and reuse draft page bindings without per-step GPU counts

[`DFlashCacheResidency`](../python/sglang/srt/speculative/dflash_cache.py) provisions
a bounded set of SWA bindings from Full-owned virtual pages. Host allocator counts
and the window/page geometry determine the required capacity. Existing request
allocation bounds cover the overlap scheduler's delayed commit metadata.

The fused
[`dflash_update_window`](../python/sglang/kernels/ops/attention/dflash_window.py)
kernel gathers the active windows, identifies shared pages, and reuses retired
bindings for newly needed pages. At the measured geometry, ordinary decode uses
one kernel and needs no GPU count readback. Larger geometries retain a multi-CTA
fallback with the same ownership contract.

Physical ownership stays with the unified allocator. The kernel changes bindings,
not physical capacity. Since the parity pass, Full-page frees leave draft bindings
in place as spares. Compaction uses the allocator's known live count; no new request-indexed
KV pool is introduced. Epoch arguments remain runtime values so advancing the
decode iteration does not create new Triton specializations.

Held bindings can include spare pages. Consequently, shared prefixes reduce the
number of distinct materialized pages but do not necessarily reduce the held
physical allocation. The fixed reservation is unchanged and is not lent to Full KV.
After radix deduplication, a nonzero binding alone is not proof of valid draft
contents: the radix insert records the rows it moved onto canonical pages, and the
next window preparation rebuilds them from the live tail.

### Fuse decode capture and reuse owned prefill chunks

[`capture_dflash_features`](../python/sglang/kernels/ops/kvcache/dflash_features.py)
copies reusable target outputs into owned storage and emits start, accepted-row
count, and checkpoint-boundary metadata in one launch. Only that small metadata
is copied to the CPU. Decode result consumption appends accepted rows; rejected
rows never become part of a live or checkpoint tail.

Prefill captures own independent chunks of at most 256 rows. Consumption adopts
these chunks where possible, avoiding a second prefill copy. Checkpoints and live
windows share immutable views; appends only fill unused chunk suffixes.
Canonical-page repair concatenates the live tail only for rows a radix insert
recorded. Decode still has a packed capture followed by accepted-row
appends; that remaining copy has not been eliminated.

Target prefill is enqueued before independent draft-pool preparation. The target
uses Full KV, and draft bindings are ready before the draft first translates SWA
locations. Short prefill extensions avoid an unnecessary tail-selection gather.

### Transfer strided recurrent state directly

The existing
[`Mamba transfer kernels`](../python/sglang/kernels/jit/csrc/kvcacheio/transfer_mamba.cuh)
now accept a slot stride in bytes. Unified recurrent-state views can therefore
transfer their contiguous per-slot payload directly, without gathering every
layer into temporary storage and uploading a new pointer table.

The fast path requires contiguous rows and aligned pointers/strides. Unsupported
layouts, mixed layer allocations, and unaligned layer strides retain the staging
path. Tests check both neighboring slots and padding; they include the subtle case
where the first layer is aligned but a later layer is not.

Isolated H200 D2H measurements for 30 layers:

| State payload per layer | Checkpoints | Staged transfer, ms | Strided transfer, ms |
| --- | ---: | ---: | ---: |
| 32 KiB convolution | 1 | 0.2206 | 0.0256 |
| 1 MiB temporal | 1 | 0.8014 | 0.6148 |
| 32 KiB convolution | 32 | 0.8698 | 0.6163 |
| 1 MiB temporal | 32 | 20.3092 | 19.2158 |

These are transfer microbenchmarks, not end-to-end speedups. The reproducer is
[`bench_transfer_mamba.py`](../test/registered/kernels/benchmark/mamba/bench_transfer_mamba.py).
Since the parity pass, page-first CUDA backups use the staged copy-engine path
instead, because the direct kernels occupy SMs for the whole transfer. Loads and
unstaged layouts still use the strided kernels.

### Preserve asynchronous transfer and rank ownership

HiCache readiness consensus is pipelined for the supported single-stage,
storage-disabled DFlash path. A pending consensus retains the exact acknowledgment
objects it counted. A blocking eviction drain cannot make its old counts apply to
new acknowledgments. Other HiCache protocols retain their existing behavior.

Raw host-slot release now returns **logical capacity** immediately. Reusing that
slot queues a GPU wait on its previous occupant's final reader or writer, before
the next D2H write. Abandoning an unfilled reservation preserves that fence.
Allocation availability therefore does not depend on rank-local DMA completion,
allowing removal of the synchronous per-donation capacity collective.

The actual GPU transfer-source queue remains bounded at `2 × max_requests` across
D2H and H2D. Saturation waits for the oldest completion; unfilled host reservations
own no transfer source. Normal submission stays asynchronous. This retains the
chosen checkpoint instead of dropping it to meet a temporary transfer limit.

Raw-tail backups now share the controller's existing D2H stream with target KV
and recurrent-state backups. The binding happens during cache construction,
before any transfer can start; raw H2D retains its independent stream. This
serializes competing backups without adding a buffer or changing checkpoint
ownership. Two paired prototype runs improved C32 throughput by about 0.95%,
with C1 essentially unchanged. The retained initialization passed the five GPU
transfer-ordering tests and 46 CPU tests with 83 subtests. The independent
state/tail restore test now uses the shared backup stream and separate load streams.

### Keep recurrent-slot admission separate from its target-token cost

The bounded allocator reserves recurrent-state bytes independently of its Full
token cap. It inherited a planner cost intended for pools where each new state
consumes bytes otherwise available to Full KV. Charging that cost against the
bounded Full budget counted the same reservation twice.

The bounded allocator now reports zero incremental Full-token cost for a Mamba
slot. The prefill planner carries **slot counts** independently of that cost:
request and host-checkpoint slots still require capacity and are debited when
admission commits. Shared-budget allocators retain their existing byte-derived
Full-token charge. No new server option or public kernel API was introduced.

The regression tests cover host restore with sufficient target-token space but
insufficient state slots, zero-cost slot accounting, and construction of the
separate slot budget. A real bounded-pool test fills every recurrent slot and
still allocates its full advertised target budget. The planner tests fail
against the previous implementation. The correction also passed the TP2 warm
reuse, long-decode, host-pressure recovery, and 32-branch serving reproducer.

### Report realizable capacity consistently under automatic sizing

Removing the small `--max-total-tokens 131072` benchmark cap exposed a startup
accounting failure. The enabled allocator had a fixed admission budget of
4,472,832 Full tokens, but the live draft region split the arena's free space,
leaving 2,679,232 immediately available Full tokens and 64 cached tokens. Its
reported total still used the fixed budget, so the scheduler reported a leak.

The allocator now returns its existing `current_full_capacity` for `size`.
This pairs the same availability view with the allocated count, following the
existing unified Mamba allocator's accounting contract. The fixed admission cap
is unchanged; allocation can still move the draft region to realize that cap.
The fix does not disable or loosen the scheduler's invariant checker.

A regression test fails before the fix and passes afterward at page sizes 1
and 4. It exercises the real allocator and invariant checker, verifies that
omitting one allocated page from the accounted cache still reports a leak,
and allocates the entire remaining fixed budget after moving the draft region.
The surrounding allocator and invariant suites pass **143 tests with 924
subtests**. Both corrected TP2 configurations passed warm reuse, long decode,
and 32 concurrent branches. The fixed-cap test also forced host-only recovery:
4,096 cached prefix tokens came from host and only the 37-token suffix required
target extension. The automatic-sizing run used no pressure requests and does
not establish host-only recovery by itself. Two fixed-cap throughput repeats
completed with unchanged acceptance lengths (5.1512 at C1 and 4.9906 at C32);
their results and source hashes are in the JSON's `capacity_accounting_fix`
entry. Automatic-sizing throughput was unfinished at the user's pause and was
subsequently completed in the separate series above.

## What the warmed Nsight traces establish

Temporary NVTX wrappers live outside production under `/tmp/dflash-parity/python`.
The warmed traces that motivated stream serialization are
`/tmp/dflash-parity/warm_profile_whole_local.{1,2}.nsys-rep`; the flag-off controls
have the prefix `off_warm_profile_whole_local`. Capture 1 uses concurrency 32,
and capture 2 uses concurrency 1. SQLite exports and summaries sit beside them.
Both servers use local CPU affinity. Each ran two complete C32 warmups and one
C1 warmup before collection; each capture starts with an empty radix cache.
Collection used Nsight Systems 2026.3.2, CUDA/NVTX/OS runtime tracing, and CUDA
graph-level tracing. Neither rank in any capture has a module/library-loading
or kernel-attribute API call longer than 1 ms. Earlier traces included such
first-use stalls and remain archived. These warmed traces precede the retained
stream-binding change; its improvement is established by the unprofiled comparisons.

The following totals cover TP rank 0 across a mixed prefill/decode workload.
CPU intervals can overlap GPU work, GPU durations can overlap each other, and
some NVTX ranges are nested. **Do not add these rows to explain the throughput
gap.** GPU busy time also includes collective waits. Profiling itself changes
timing: the C1 forward span is 23.28 s on versus 21.49 s off, a much larger
difference than the unprofiled benchmark. These profiles diagnose behavior;
they do not replace the qualified throughput results.

### Kernel and host-work table

| Range | Calls | CPU range total, ms | Sum of correlated GPU durations, ms |
| --- | ---: | ---: | ---: |
| Draft residency, C32 decode | 120 | 29.5 | 3.2 |
| Raw decode capture | 684 | 92.3 | 3.6 |
| Raw result consumption, mixed batches | 765 | 502.6 | 117.7 |
| All HiCache D2H submissions | 504 | 157.4 | 1,294.3 |
| Recurrent backup, nested in D2H | 190 | 24.9 | 422.8 |
| Raw-tail backup, nested in D2H | 314 | 81.0 | 713.7 |

The initial residency implementation launched roughly 219 kernels per steady
decode round and frequently synchronized the host. Ordinary decode now uses one
fused residency kernel. Growth, frees, and canonical-page repair remain separate
operations; the warmed C32 residency trace contains 2.77 ms of synchronization
API time across 120 calls.

### Overlap opportunity table

| Path | Measured evidence | Implication |
| --- | --- | --- |
| Raw result consumption | 502.6 ms CPU range; 63.3 ms overlaps GPU-idle intervals | Much of its CPU work is hidden; removing all of it would not recover 503 ms of wall time |
| HiCache D2H submission | 157.4 ms CPU range; 11.7 ms overlaps GPU-idle intervals; 1,294.3 ms summed GPU durations | Submission is mostly hidden, but copy/inference interference can still slow GPU work |
| HiCache readiness polling | 116.8 ms CPU range; 4.1 ms overlaps GPU-idle intervals | The old synchronous polling barrier has largely been hidden at C32 |
| Prefill CPU work | Enabled C32 prefill ranges overlap 801 ms of GPU idle time; off ranges overlap 527 ms | Both configurations have substantial CPU gaps; these ranges alone do not isolate the feature's added cost |
| Raw donation | Per-donation CPU collective removed; actual slot reuse is GPU-fenced | Cleaner ownership and lower CPU work did not materially change C32 throughput by themselves |

There is a more specific lead in the C32 target-verification graphs. On rank 0,
79 enabled graphs with no overlapping HiCache backup averaged **20.18 ms**,
close to **20.22 ms** for 132 disabled graphs. The 41 enabled graphs overlapping
backups averaged **21.76 ms**, with 3.74 ms of backup activity overlapping each
graph on average. This is a correlation, not a causal estimate: sequence lengths,
checkpoint tracking, and batch composition can differ. It supports investigating
transfer/inference interaction before adding more capture machinery. The second
rank shows the same overall target-graph timing shift.

In contrast, the raw decode capture range overlaps just 0.06 ms of GPU idle time
over the entire C32 profile. Its isolated kernel cost is not the main unresolved
lead. At C1, profiling substantially amplifies CPU gaps, so those totals should
not be used to promise an unprofiled speedup.

### Fusion and boundary table

| Operation | Retained implementation | Boundary preserved |
| --- | --- | --- |
| Window gathering and rebinding | One Triton launch at the measured geometry | Unified allocator owns physical pages and fixed capacity |
| Raw decode capture and commit metadata | One Triton launch | Capture owns reusable model outputs until ordered consumption |
| Recurrent-state transfer | Existing CUDA kernel reads strided slots directly | HiCache controls submission, completion, and source lifetime |
| Accepted-row appends | Existing individual copies | Immutable checkpoint views and bounded chunk ownership |

### Current implementation after stream serialization

Fresh captures of the retained implementation are
`/tmp/dflash-parity/serial_profile_whole_local.{1,2}.nsys-rep`, with flag-off
controls prefixed `off_serial_profile_whole_local`. Both servers were primed
with two complete C32 runs and one C1 run, then flushed before each capture.
The enabled server used GPUs 4/5 and off used 2/3, each with local CPU affinity.
The instrumented snapshot is identical to production after removing the temporary
NVTX wrappers. Neither rank has a module/library-loading or kernel-attribute API
call longer than 1 ms during these captures.

| Current C32 range, TP rank 0 | Calls | CPU range total, ms | Sum of correlated GPU durations, ms |
| --- | ---: | ---: | ---: |
| Draft residency at batch size 32, including transitions | 120 | 42.0 | 18.2 |
| Raw decode capture | 684 | 89.0 | 3.6 |
| Raw consumption, including nested backups | 765 | 502.8 | 118.2 |
| All D2H submissions | 504 | 153.3 | 998.9 |
| Recurrent backup, nested in D2H | 190 | 24.2 | 358.5 |
| Raw-tail backup, nested in D2H | 314 | 78.5 | 514.1 |

The raw-consumption row contains **37.6 ms of GPU-to-GPU appends** and
**80.6 ms of nested D2H copies**. Treating its entire GPU total as append work
would overstate the opportunity for another append kernel. There were 13,805
D2D copies totaling 1.62 GB. These use copy-engine DMA in the retained path.

All D2H durations sum to about 0.30 s less than in the earlier enabled trace.
That difference is not a wall-time speedup: transfers overlap model execution.
The current C32 forward span is 17.208 s enabled versus 16.665 s off. Both
captures contain 684 draft and 684 target decode graphs; summed target-graph
time is 10.401 s enabled versus 10.280 s off. Total GPU-idle time is actually
lower enabled, 1.208 s versus 1.344 s, because backup activity can occupy gaps
that would otherwise count as idle.

The transfer correlation remains: the 79 batch-32 target graphs without backups
average 20.15 ms, while 41 with backups average 21.60 ms. The disabled control's
120 batch-32 target graphs average 20.24 ms. The affected enabled graphs overlap
raw, recurrent, and target-KV backups; this does not isolate any one transfer.
Prefill CPU ranges overlap 779 ms of GPU idle time enabled versus 695 ms off.
Raw consumption overlaps 61 ms, and HiCache event polling only 4.5 ms. These
nested ranges and overlapping GPU durations must not be added as independent costs.

C1 profiling still amplifies the difference: its forward span is 23.052 s enabled
versus 21.482 s off, much larger than the qualified unprofiled gap. The throughput
comparison at the top remains the performance result. These new traces, the
source manifest, and an analysis of individual GPU-idle gaps are recorded in
the measurement JSON and local artifacts.

## Experiments excluded from the final implementation

- **Batched raw appends:** `torch._foreach_copy_` was faster than a custom
  pointer-copy kernel in isolation, but its serving runs were about 19.0–19.3k
  total tokens/s at C32. No demonstrated serving improvement justified the
  deferred-copy and donation-ordering machinery. It was removed.
- **Direct decode capture:** a separate prototype writes accepted rows directly
  into owned chunks and only commits their lengths during result consumption.
  Exact checks passed for 480 queued overlap pairs, rejected rows, chunk
  boundaries, preserved checkpoints, and canceled owners; TP2 warm reuse,
  long decode, host-pressure recovery, and 32 branches also passed. Two repeated
  comparisons gained 0.36% at C1 and 0.61% at C32 over the retained implementation,
  while still trailing flag off by 1.08% and 3.74%. A second prototype reusing
  destination tables gained only about 0.20% at either concurrency in its own
  comparison. The two prototypes ran on different GPU pairs, so this does not
  isolate the table-cache change. Neither was retained: the serving gain was small,
  and chunks allocated ahead of CPU commits lacked a general memory bound.
  Queued captures can retain those destinations after their owner closes;
  the existing packed-capture staging bound cannot simply be reused. The later
  bounded-reservation retest above accounts for this storage and finds no gain.
- **CPU prefix-remap version:** insertion metadata avoided a GPU readback on
  newly inserted branches while preserving duplicate-prefix repair. Tests passed,
  but C32 still measured about 19.0–19.3k and C1 about 3.84k. The extra version
  state was removed during minimization.
- **No decode capture:** a temporary diagnostic disabled new decode recovery
  captures. It measured 19.27–19.54k at C32 versus 18.95–19.23k for its nearby
  enabled control, roughly a 1–2% improvement. This loses recovery coverage for
  new decode checkpoints and is **not a valid implementation**. It shows why a
  complete decode-copy rewrite alone is unlikely to recover the entire gap.
- The flag-off Python TreeCore control remained close to the Rust control;
  changing TreeCore alone does not account for the remaining loss.
- **Fused raw-tail D2H kernel:** a temporary JIT CUDA prototype passed exact
  data checks for full, partial-boundary, and short tails. For a 2,048-row,
  12,288-feature BF16 tail, existing copies took 0.933 ms; fused mapped-host
  copies took 1.103–1.117 ms with 2–8 CTAs. CPU submission fell from about
  36 to 6 microseconds, but the device transfer became slower. A C++ loop
  retained the 0.933 ms device time with about 12 microseconds of CPU work.
  Neither justified a new production operator. CUDA's batched-copy call
  returned an invalid-argument error in this prototype and is
  excluded from the timing comparison. These are isolated, repeated copies
  on one H200 with local pinned host memory, not serving measurements.
- **Omitted duplicate raw copies:** a diagnostic recognized overlapping views
  of the same live GPU allocations and skipped about 46% of raw rows. Two C32
  runs reached 19,423 and 19,417 total tokens/s, versus 19,231 and 19,303 for
  the retained implementation and 20,136 and 20,143 for flag off. This is only
  about a 0.8% serving gain despite the large byte reduction. The diagnostic
  deliberately leaves incomplete host payloads and raises on any H2D restore;
  it is **not a valid host deduplication implementation**. A real implementation
  would additionally need shared host-chunk ownership, capacity accounting,
  and reader/writer fences. That added machinery has not been retained.
- **Omitted decode capture and all D2H payloads together:** two diagnostic C32
  runs reached 19,932 and 19,845 total tokens/s under the same cold workload.
  This still trails the 20,136–20,143 off control. It removes required recovery
  behavior and changes checkpoint donation work, so it is an upper-bound
  experiment, not an additive attribution of individual costs. H2D restores
  were forbidden here too.

The latter two diagnostics used zero warmup requests, flushed before every run,
and excluded a separate priming run. Their controls used the same procedure.
They must not be mixed into the four-warmup qualified tables above.

Diagnostic snapshots, benchmark logs, and source manifests remain in
`/tmp/dflash-parity`. Wrong-port controls and overlapping benchmark runs are
explicitly excluded in `invalid-controls.txt` there. No production NVTX markers,
new tuning environment variables, or no-capture paths remain.

### Follow-up experiments after stream serialization

A new pair of temporary diagnostics omitted either raw-tail or recurrent-state
backup payloads while preserving capture, index preparation, completion events,
and cache bookkeeping. H2D restores raised explicitly because their host payloads
were incomplete. Two C32 repeats measured 19,570 tokens/s with raw copies omitted
and 19,627 with recurrent copies omitted, versus 19,527 for the retained control
and 20,139 for flag off. These are about 0.2% and 0.5% gains over the enabled
control, respectively. The effects are not additive and do not explain the whole
remaining gap. C1 showed no gain: both omission variants measured about 3,835,
versus 3,847 enabled and 3,885 off. The variants used different GPU pairs, all with
local CPU affinity; the first C1 pass can also include remaining shape warmup.
These diagnostics used zero benchmark warmups and must remain separate from the
qualified four-warmup comparison.

A second C32 diagnostic pair omitted target-KV payloads alone or every D2H
payload. Two repeats measured 19,462 tokens/s without target-KV copies and
19,852 with all copies omitted, against 19,429 enabled and 20,128 off. The all-copy
diagnostic gained about 2.2% over that enabled control and still trailed flag off
by 1.4%. These figures also span different GPU pairs; the omitted-copy runs are
invalid recovery implementations. They support investigating transfer scheduling
and interaction, rather than treating each isolated omission as an additive cost.

A **fused recurrent-state gather followed by copy-engine DMA** passed exact
30-layer payload checks, unchanged host neighbors, device padding, repeated
asynchronous backups, and strided H2D restore. The TP2 serving test also passed
warm reuse, long decode, host-pressure recovery, and 32 branches. The temporary
kernel gathers a GPU-indexed checkpoint into contiguous GPU scratch, then copies
it to the existing CPU-selected host slot. One 30 MiB temporal checkpoint took
about 0.608 ms versus 0.614 ms with the existing direct transfer kernel. CPU
submission was more expensive. Two serving repeats measured 19,519 versus 19,491
C32 tokens/s, and 3,843 versus 3,851 at C1. The candidate and control used GPUs
0/1 and 4/5 respectively, so these small differences are not a clean causal
estimate. This does not justify retaining the extra staging path. The parity
pass retained a staged copy-engine backup after removing the blocking host
synchronizations; it then closed about 0.9 point at C32.

A second prototype **kept draft allocations across target-page frees**, allowing
the window kernel to reuse them without the release path's dynamic filter.
It used the existing reservation and introduced no separate pool. It passed
200 Full-ID recycling cycles across page sizes 1 and 4, byte accounting,
canonicalization, and the TP2 recovery test. A fresh control on the same CPU NUMA
node measured 19,458 C32 tokens/s versus 19,409 for the prototype; C1 was effectively
unchanged at 3,840 versus 3,838. This prototype was rejected at the time. The
parity pass revisited it after removing the other blocking synchronizations, and
it then closed about 1.4 points at C32; production now keeps the bindings.

Raising the target and draft forward streams to **CUDA priority −1** also passed
the TP2 recovery test, but did not establish a useful throughput gain. Two repeats
measured 19,437 C32 tokens/s versus 19,411 for the enabled control (+0.13%), and
3,836 versus 3,839 at C1 (−0.08%). Flag off measured 20,155 and 3,890 respectively.
The streams used for backups stayed at priority 0. This constructor-only
experiment was rejected; production retains its existing stream priorities.

Replacing each immediate accepted-row `Tensor.copy_` with PyTorch's existing
**multi-tensor copy kernel** also showed no gain. An operator trace confirmed
that it used GPU SMs instead of copy-engine DMA, and exact partial-copy checks
plus the full TP2 recovery test passed. Two repeats measured 19,378 C32 tokens/s
versus 19,380 for the enabled control (−0.01%), and 3,835 versus 3,837 at C1
(−0.05%). This experiment kept append and donation ordering intact; it did not
reintroduce the earlier deferred-batch machinery. It was rejected. A batched
version for decode appends and prefill chunks was measured again in the parity
pass, with the same result.

The JSON records the individual runs, source manifests, launch configurations,
and correctness artifacts. These prototypes changed neither production code nor
its memory reservation.

## Correctness, memory, and remaining work

The retained implementation passed the TP2 serving reproducer for a warm prefix,
a 2,400-token decode followed by reuse of an older checkpoint, host-pressure
eviction and recovery, and 32 concurrent branches. The host recovery reported
4,096 host-cached tokens and zero device-cached tokens, then extended only the
37-token suffix. A separate test launched 32 identical cold branches and checked
every generated token ID against flag off.

Before the admission correction, targeted checks passed **231 CPU tests with
215 subtests**, covering the feature, allocator, HiCache acknowledgment handling,
and kernel namespaces, plus
**43 CUDA tests with 62 subtests**, covering window kernels, raw capture, strided
Mamba transfers, and host-tail stream ordering. The CUDA tests ran on H200.
The admission correction passed a further **229 CPU tests with 888 subtests**
across planner, DLLM, and allocator suites, plus the TP2 serving reproducer again.
These counts describe separate validation rounds and include overlapping coverage.

The host-slot race test delays an old reader, releases its slot, and immediately
backs up a replacement. Both restores must return the correct data without a CPU
wait. A temporary negative control that removes only the slot fence fails this
test, confirming that the dependency is necessary. Unit tests also cover canceled
reservations, delayed result consumption, checkpoint replacement, shared pages,
allocator bounds, and rank-independent logical slot availability.

Admission still reserves **9.74267578125 GiB of raw GPU headroom per rank** at
32 requests and a 4,096-token prefill ceiling. The materialized draft pool is
69,632 rows, **0.796875 GiB per rank at TP2**. Raw host storage is **12 GiB per rank
at 256 checkpoint slots**, in addition to ordinary target/recurrent HiCache pools.
These are admission limits, not measured resident allocations. The projection
term now explicitly covers a concatenated window and selected repair rows held
together; the baseline value is unchanged. The
[memory calculator](draft-kv-memory-budget.html) uses the same formula.

The unresolved task is to recover the remaining C1 and C32 throughput gaps
without removing recovery behavior or increasing target replay. After the
parity pass, the same-node gaps are about 0.5% at C1 and 0.6–1.3% at C32; the
[remaining-gap analysis](#remaining-gap) lists the measured contributors.
Direct capture with its complete destination-memory bound and queued raw-backup
submission remain rejected after measurement. Asynchronous request-map uploads
are retained in the parity pass. The
[staging-memory accounting cleanup](#hicache-staging-memory-accounting-2026-10-22)
now reserves the bounded path's late transfer scratch before sizing the cache.

Live draft-weight revision orchestration, lending unused reservations, Rust
TreeCore bundle hooks, and raw-tail L3 persistence remain separate design work.
