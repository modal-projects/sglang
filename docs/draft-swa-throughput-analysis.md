# DFlash SWA throughput diagnosis and fused page gathering

Historical report: the [subsequent optimization report](draft-swa-throughput-optimization.md)
describes the current implementation, repeated C1/C32 measurements, and remaining gap.

Measured on 2026-10-21. This follows the raw-tail storage work in
[draft-kv-checkpoint-design.md](draft-kv-checkpoint-design.md).

The main regression is in the implementation of draft-window residency:
GPU-dependent page bookkeeping blocks the CPU before it can enqueue the next
forward. The timeline shows extra GPU idle time while the main model kernels
remain similar. Raw-tail maintenance adds CPU work and copies, but the trace
does not support attributing the entire loss to moving the raw feature payload.

A fused page-gather kernel now removes the per-request tensor-operation loop.
It preserves the existing unified allocator and recovery behavior. Its isolated
speedup is large; the serving improvement is modest because the remaining
allocation operations still synchronize with the GPU.

## Controlled throughput comparisons

The target was the local Qwen3.6-35B-A3B snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`; the draft was
`/tmp/autospec/baseline`. Each server used two H200 GPUs, BF16, FA3 for target
and draft, unified memory, CUDA graphs, page size 64, draft block size 16,
32 maximum active requests, 131,072 configured target tokens, 256 recurrent
slots, and 4,096-token chunked prefill. HiCache variants used ratio 2.

Each throughput run flushed the cache, performed four warmups, then served
128 random-token requests with 2,048 input and 512 output tokens at concurrency
32, seed 20261021 and greedy decoding. Every listed run completed all 128
requests. The reported throughput is **input plus output tokens per second**;
output-only throughput is exactly one fifth for this fixed workload.

Three runs per configuration used rotated order. Temporary NVTX instrumentation
was present, with Nsight armed but CUDA tracing inactive during throughput
measurement. Profiling runs were separate and are not used as throughput results.

| Configuration | Median total tokens/s | Range across three runs |
| --- | --- | --- |
| Flag off; HiCache off; Rust TreeCore | 19,917 | 19,897–20,183 |
| Flag off; HiCache off; Python TreeCore | 19,826 | 19,825–20,079 |
| Flag on; write-through HiCache; Python TreeCore | 17,091 | 16,764–17,154 |
| Flag on; write-back HiCache; Python TreeCore | 16,918 | 16,748–17,114 |

The Rust and Python controls differed by 0.46% in the median. Write-back was
1.0% slower than write-through; it did not recover the
regression. Both enabled variants still retain raw checkpoint tails, so this
comparison tests ordinary HiCache write policy, not removal of raw backups.
The enabled implementation before fusion was 14.2% below the
Rust flag-off control in this series. The earlier Step 10 result was 13.4%;
these are separate measurements.

The original enabled server and both fused iterations used the same GPU pair,
4/5. Flag-off Python ran on 2/3 and was rechecked during the fused runs. The
original Rust control used 0/1; write-back used 6/7. The feature-on/off comparison
is not a GPU-pair crossover. The initial fused iteration used synchronous
metadata upload; the final implementation uses pinned, asynchronous upload.
The first run of the initial fused iteration also overlapped a CPU namespace test,
so the final series is the result to use for the implementation.

| Final comparison | Median total tokens/s | Range across three runs |
| --- | --- | --- |
| Fused gather + asynchronous metadata upload | 17,422 | 17,188–17,443 |
| Flag-off Python control, rechecked | 19,627 | 19,596–19,664 |

The final fused path improved by **1.9%** over the original enabled median on
the same GPU pair. It remained **11.2% below** the contemporaneous flag-off
Python control. That control was 1.0% lower than its original median, so
the small before/after gain should not be treated as a precise causal estimate
or a general serving-speedup guarantee. The Nsight and isolated-kernel results
establish that the intended bookkeeping work decreased.

This workload measures the overhead of maintaining recovery state with little
prefix reuse. It does not measure the benefit of avoiding historical target
replay after a prefix hit or draft update, or gains from fitting more target
KV into memory. The feature remains opt-in.

## What Nsight measured

Temporary wrappers marked draft-window residency, target-feature capture,
result consumption, draft projection, raw host backup/restore, radix insertion,
and HiCache submission. All wrappers live in an overlay under
`/tmp/dflash-throughput-cause/python`; no NVTX instrumentation was added to
production modules.

Collection used Nsight Systems 2026.3.2 with CUDA, NVTX and OS runtime tracing,
CUDA graph node tracing, and `cudaProfilerStart/Stop` through the existing
`/start_profile` endpoint with `activities=["CUDA_PROFILER"]`. The decode
capture started after the scheduler reported 32 active requests and covered
20 forward steps. Separate traces cover 32 cold 2,048-token prompts with
32 output tokens to inspect prefill and checkpoint donation.

The following numbers use TP rank 0 only. Round timing excludes the first and
last capture edges: 17 interior start-to-start intervals. Range means exclude
the first and last instances: 18 complete ranges. GPU busy time is the union
of kernel, memcpy and memset intervals on that rank, including overlapping
streams only once. It includes communication kernels and is not a measure
of useful arithmetic utilization.

| Decode trace | Round interval (ms) | GPU busy union (ms) | GPU idle (ms) |
| --- | --- | --- | --- |
| Flag off, Python | 22.137 | 21.793 | 0.344 |
| Flag on, before fusion | 26.539 | 23.071 | 3.468 |
| Flag on, final fused gather | 25.695 | 23.141 | 2.555 |

In the original trace, the additional idle time was about 3.12 ms per round,
compared with a 4.40 ms increase in round interval. Main compute was similar:
20 rounds of `fused_moe_kernel` took 166.35 ms off versus 166.49 ms on;
the recurrent verify kernel family took 103.66 versus 103.63 ms. These are
sums within each family, not additive wall-clock components. Incremental draft
context projection used the same kernel family and similar durations. There
was no full-window reprojection or host raw-tail transfer in this steady
decode slice.

Profiler overhead changes absolute timing, and these short slices do not
provide an exact percentage decomposition of the unprofiled end-to-end loss.
They identify the blocking path and show why reducing its launch count alone
has a smaller serving effect than its microbenchmark suggests.

### Kernel and host-work table

| Range, per round | Kernels | Attributed GPU union (ms) | CPU excluding sync APIs (ms) | Stream sync calls |
| --- | --- | --- | --- | --- |
| Residency, before | 219.0 | 0.443 | 3.967 | 8.78 |
| Residency, final | 45.1 | 0.167 | 1.381 | 7.67 |
| Raw capture, original trace | 10.0 | 0.019 | 0.306 | 0 |
| Raw consumption, original trace | 0.0 | 0.117 | 1.375 | 0 |

The original residency range lasted 14.26 ms, but 10.30 ms was inside CUDA
stream synchronization calls. Much of that wait was for already-enqueued
model work. It would be incorrect to call all 14.26 ms newly added compute,
or to add those waits to GPU kernel durations. The relevant regression is
that the CPU must finish these waits and allocation bookkeeping before
submitting the next forward, creating gaps in GPU work.

The old path built 32 token-position arrays, clamped them, indexed the request
table separately, divided token IDs into pages, concatenated roughly 66,000
entries and computed their union. It then found dead and missing pages with
GPU set operations and dynamic masks. CUDA needs host-visible output sizes
for several of these operations. Freeing and allocating through the existing
unified allocator introduces further small operations and synchronization.

Raw consumption appended newly committed rows with about 33 D2D copies per
round in the original trace, taking about 0.117 ms of GPU copy time and
1.38 ms of CPU range time. Most of that CPU range overlapped ongoing GPU
work in the measured decode slice. Decode capture copied one 12 MiB packed
buffer plus small position metadata; its GPU operations occupied about
0.019 ms. These are implementation costs, but not evidence that HBM bandwidth
for raw tails accounts for the full throughput loss.

### Overlap opportunity table

| Path | Evidence | Implication |
| --- | --- | --- |
| Next-forward submission versus prior GPU work | Dynamic page operations force stream synchronization; GPU gaps grow in the enabled path | Highest-priority follow-up: avoid host-dependent set rebuilding on the ordinary decode path |
| Raw feature consumption | Many small copies and Python calls; CPU range mostly overlaps GPU work in the decode slice | Batch copies if they remain exposed after the residency stalls are removed; do not assume the full CPU range is recoverable wall time |
| Checkpoint D2H | No transfers in the steady decode capture; 32 raw backups in the cold-prefill/donation capture | Offload is episodic in these workloads, not a whole-window copy on every decode round |

The cold-prefill/donation trace copied 32 raw windows, 1.5 GiB per rank, in
about 53.7 ms of summed D2H operation time. Host submission averaged 0.315 ms
per window. The raw transfer-capacity ranges contained **no blocking
synchronizations**. The queue limit was not responsible for the measured
steady-state loss. Different checkpoint rates, prefix reuse and host pressure
can change these costs; this trace does not establish a general offload ceiling.
Ordinary target/recurrent HiCache backups also appeared in the prefill trace.

### Fusion table

| Operation | Change or remaining boundary | Relationship to existing infrastructure |
| --- | --- | --- |
| Window bounds, page positions and virtual-page gather | Implemented as one Triton launch for all active windows | Same metadata-fusion family as the existing attention metadata helpers; this contract includes retained windows and radix canonicalization |
| Global union and dead/missing-page selection | Remains in PyTorch and the unified allocator | Removing these barriers needs an allocator/residency change; the gather kernel alone cannot safely replace ownership accounting |
| Raw append copies | Unchanged | Existing shared chunks and transfer leases keep their lifetime guarantees |

## Implemented kernel and invariants

The public kernel is
[`attention.dflash_window_pages`](../python/sglang/kernels/ops/attention/dflash_window.py),
registered lazily in the attention group. Runtime integration is confined to
[`DFlashCacheResidency`](../python/sglang/srt/speculative/dflash_cache.py).
There is no new server option, allocation policy, pool, transfer path, or
change to raw-tail payloads.

The kernel records the current batch's logical bounds and gathers one virtual
page ID per intersecting logical page. Retained requests keep their bounds but
reread the request table, so canonicalization remains visible even when a
request is absent from the current batch. CPU metadata identifies active
request rows and their current batch indices; its upload is pinned and
asynchronous.

The correctness conditions are:

- Within each logical request page, tokens belong to one virtual page. The
  existing paged allocator and page-aligned radix operations provide this
  invariant. The first gather position is clamped to the window start, so a
  partial first page does not require reading outside the interval.
- Prefill keeps `[max(0, end-W), end)`. Decode keeps
  `[max(0, prefix-W), prefix+B)`, including the speculative block.
- Old and current active windows can have different block lengths. Output
  width uses the largest active block length and includes both partial ends.
- Request-row reuse initializes new bounds before reading them. The CPU
  ownership map removes released/replaced requests; clearing it makes the
  next admitted windows initialize their bounds again. The GPU bounds table
  is only 16 bytes per request-pool row.
- A guaranteed zero sentinel and nonnegative page entries allow
  `torch.unique(pages)[1:]` to remove padding without another dynamic mask.
- Shared virtual pages are deduplicated by the existing union; eviction,
  allocation, prefill repair, compaction and raw-tail lifetimes keep their
  existing owners.

The minimization pass kept the existing allocator interfaces and prefill
repair, added no compatibility path, and put the CPU reference in test support
rather than introducing a production fallback for this CUDA/HIP-only feature.
CUDA was exercised here; no AMD run was available.

## Validation and isolated benchmark

The GPU test compares page unions and updated bounds against a token-wise CPU
oracle for 18 combinations of page sizes 1/4/64, windows 1/8/2048 and int32/int64
indices. Each combination checks initial decode windows, a prefill update
with retained decode windows and canonical remapping, and recycled request
rows. It includes zero/short prefixes, page boundaries, non-contiguous outer
request-table strides, shared pages, and IDs above int32 range.

The existing CPU allocator fixture still verifies shared physical pages,
canonical-page repair, freeing and accounting; it substitutes the CPU oracle
only at the GPU-kernel boundary. All 61 tri-pool tests passed. The namespace
and dispatch suites passed 121 tests; kernel correctness passed, along with
registered-test validation and the applicable pre-commit checks.

The final serving check passed warm 4,096-token reuse plus a 37-token suffix,
2,400-token decode followed by reuse of the older checkpoint, host recovery
after 20 pressure requests, and 32 concurrent branches. Warm, long-prefix
reuse and host-restored outputs matched their corresponding reference token
IDs. Branch requests completed with the expected cache hits.

The isolated benchmark includes input metadata construction/upload, gathering,
unique-page extraction and a final GPU drain in its wall-time measurement.
It excludes allocation/free, model execution, raw capture and offload:

| Requests | Token-wise path (µs) | Fused path (µs) | Ratio |
| --- | --- | --- | --- |
| 1 | 123.1 | 68.4 | 1.8× |
| 8 | 422.2 | 95.3 | 4.4× |
| 32 | 1209.2 | 74.3 | 16.3× |

It is reproducible with
[`benchmark_dflash_window.py`](../test/manual/kernels/benchmark/attention/benchmark_dflash_window.py).
The large isolated speedup must not be substituted for the measured serving
gain: much of the original gather submission overlapped the preceding GPU
forward, while the later synchronization and next-forward launch gaps remain.

## Next optimization boundary

The next useful change is to avoid rebuilding variable-length GPU page sets
on every ordinary decode round. Candidates include updating residency only
when page coverage or ownership changes, using conservatively reserved page
coverage when overlap hides the exact current length, or using fixed-shape
page bookkeeping that does not return dynamic counts to the CPU.

That work needs explicit treatment of speculative acceptance, pending
forwards, shared pages, request-row reuse, canonicalization and pool-cap
geometry. It should retain the existing allocator as the ownership authority.
The current measurements do not justify promising that any one approach will
recover the entire remaining throughput gap.

## Reproduction artifacts

Machine-readable results and launch commands are in
[draft-swa-throughput-analysis.json](draft-swa-throughput-analysis.json).
Full local artifacts are under `/tmp/dflash-throughput-cause/`: before/after
server logs and benchmark JSONL files, instrumentation generator, analysis
script, serving outputs and Nsight reports. The final decode report is
`fused_async.1.nsys-rep`; `on.1.nsys-rep` and `off_python.1.nsys-rep` are the
original enabled and flag-off decode reports. The corresponding `.2` reports
cover cold prefill/donation. These local trace files are not committed into
the documentation tree.
