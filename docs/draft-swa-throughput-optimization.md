# DFlash bounded-cache throughput optimization

Measured on 2026-10-21. This report supersedes the implementation and performance
status in [the initial fused-gather report](draft-swa-throughput-analysis.md).
The earlier report remains useful as the initial diagnosis.

**Parity is not achieved yet.** The retained implementation is 0.99% below flag
off at concurrency 1 and 3.23% below at concurrency 32 in the repeated comparison
below. The original large regression has been reduced, but the remaining gap
should not be described as noise or attributed entirely to raw-feature bandwidth.

Optimization is **paused for an intermediate work checkpoint**. The latest
accounting fix passed serving checks and two further fixed-cap comparisons:
concurrency 1 measured 3,837 enabled versus 3,885 off (−1.25%), and concurrency
32 measured 19,407 versus 20,094 (−3.42%). These later runs used GPUs 0/1 enabled
and 6/7 disabled, with local CPU affinity; the earlier three-repeat table below
remains a separate series. Automatically sized startup and serving passed, but
its throughput run was interrupted during excluded priming. No automatically
sized throughput result is claimed. All benchmark servers have been stopped.

The feature still preserves checkpoint recovery, keeps materialized draft KV,
and uses the existing unified allocator and HiCache transfer engine. It remains
opt-in through `--enable-dflash-swa-cache`.

## Qualified throughput results

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
not physical capacity. Full-page frees still remove the corresponding draft
bindings. Compaction uses the allocator's known live count; no new request-indexed
KV pool is introduced. Epoch arguments remain runtime values so advancing the
decode iteration does not create new Triton specializations.

Held bindings can include spare pages. Consequently, shared prefixes reduce the
number of distinct materialized pages but do not necessarily reduce the held
physical allocation. The fixed reservation is unchanged and is not lent to Full KV.
After radix deduplication, a nonzero binding alone is not proof of valid draft
contents: the retained prefill features still repair changed canonical IDs.

### Fuse decode capture and reuse owned prefill chunks

[`capture_dflash_features`](../python/sglang/kernels/ops/kvcache/dflash_features.py)
copies reusable target outputs into owned storage and emits start, accepted-row
count, and checkpoint-boundary metadata in one launch. Only that small metadata
is copied to the CPU. Decode result consumption appends accepted rows; rejected
rows never become part of a live or checkpoint tail.

Prefill captures own independent chunks of at most 256 rows. Consumption adopts
these chunks where possible, avoiding a second prefill copy. Checkpoints and live
windows share immutable views; appends only fill unused chunk suffixes.
Canonical-page repair concatenates a retained tail only when reprojection is
actually required. Decode still has a packed capture followed by accepted-row
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
entry. Automatic-sizing throughput remains unfinished at the user's pause.

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
  isolate the table-cache change. Neither is retained: the serving gain is small,
  and the general memory bound for chunks allocated ahead of CPU commits remains
  unproved. Queued captures can retain those destinations after their owner closes;
  the existing packed-capture staging bound cannot simply be reused.
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
estimate. This does not justify retaining the extra staging path.

A second prototype **kept draft allocations across target-page frees**, allowing
the window kernel to reuse them without the release path's dynamic filter.
It used the existing reservation and introduced no separate pool. It passed
200 Full-ID recycling cycles across page sizes 1 and 4, byte accounting,
canonicalization, and the TP2 recovery test. A fresh control on the same CPU NUMA
node measured 19,458 C32 tokens/s versus 19,409 for the prototype; C1 was effectively
unchanged at 3,840 versus 3,838. This prototype was rejected. Production continues
to remove draft bindings when their Full pages are freed.

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
reintroduce the earlier deferred-batch machinery. It was rejected.

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

The unresolved task is to recover the remaining C32 throughput gap without
removing recovery behavior or increasing target replay. Further work should
measure the exposed transfer/inference interaction and CPU scheduling costs
before adding a direct-to-chunk decode capture protocol. Its measured prototype
gain and the no-capture diagnostic put the potential benefit in context. A
production version would still require a complete lifetime proof and memory
bound for destinations prepared ahead of CPU commits.

Live draft-weight revision orchestration, lending unused reservations, Rust
TreeCore bundle hooks, and raw-tail L3 persistence remain separate design work.
