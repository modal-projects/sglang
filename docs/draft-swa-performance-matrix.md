---
title: "DFlash deployment performance matrix"
description: "Repeated fixed-capacity cold and warm-prefix throughput measurements for materialized KV and optional raw taps, with GPU and HiCache storage."
---

# DFlash deployment performance matrix

Measured on 2026-10-22 after the [deployment split](draft-swa-deployment-shapes.md).
The bounded path still has a throughput cost: materialized-only GPU storage is
4.65% below the original unified flag-off baseline on cold prompts at concurrency
32. Raw tap retention adds little cost when GPU-resident, but loading both payloads from
HiCache is more expensive on warm hits: 24.65% below flag off at
concurrency 32, and 13.34% below host materialized KV alone.
These are measurements of this configuration and synthetic workloads, not
claims about every workload or the optimal deployment policy.

The [JSON record](draft-swa-performance-matrix.json) contains every measured
sample, per-launch results, configuration, acceptance, latency, memory measurements,
and the frozen source fingerprint. Runtime code was not changed during this
benchmark. The existing staged checkpoint was preserved.

A [flag-off HiCache follow-up](#flag-off-with-hicache) below adds paired
separate-pool controls and records why the unified variant cannot start.

The KV rows enable `--enable-dflash-swa-cache`; the tap rows also enable
`--enable-dflash-tap-cache`. HiCache rows additionally enable
`--enable-hierarchical-cache`.

## Total throughput

Median total input-plus-output tokens/s across six measured runs: three repeats
on each of two fresh launches. Percentages compare pooled medians with flag off
for the same workload/concurrency. Prime runs are excluded.

| Configuration | Cold C1 | Cold C32 | Warm C1 | Warm C32 |
| --- | ---: | ---: | ---: | ---: |
| Flag off | 3,908 | 21,035 | 50,363 | 408,410 |
| KV only, GPU | 3,884 (-0.60%) | 20,056 (-4.65%) | 49,230 (-2.25%) | 375,716 (-8.01%) |
| KV only, HiCache | 3,874 (-0.87%) | 19,912 (-5.34%) | 48,590 (-3.52%) | 355,112 (-13.05%) |
| KV + taps, GPU | 3,880 (-0.71%) | 19,995 (-4.94%) | 48,978 (-2.75%) | 377,679 (-7.52%) |
| KV + taps, HiCache | 3,869 (-0.98%) | 19,874 (-5.52%) | 47,689 (-5.31%) | 307,753 (-24.65%) |

For warm hits, total input includes **4,096 cached tokens per request**. The
large warm numbers measure logical tokens served; they do not imply that those
target tokens were recomputed. All measured warm requests hit exactly 4,096
tokens, and all measured cold requests report zero cached tokens.

For comparison, warm output-only throughput and throughput counting only
uncached input plus output exclude those cached input tokens:

| Configuration | C1 output tok/s | C1 new input + output tok/s | C32 output tok/s | C32 new input + output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Flag off | 1,513 | 1,950 | 12,269 | 15,815 |
| KV only, GPU | 1,479 | 1,906 | 11,286 | 14,549 |
| KV only, HiCache | 1,460 | 1,882 | 10,668 | 13,751 |
| KV + taps, GPU | 1,471 | 1,897 | 11,345 | 14,625 |
| KV + taps, HiCache | 1,433 | 1,847 | 9,245 | 11,917 |

## First-token latency

Median of each run's mean TTFT, in milliseconds. The same HTTP serving client
and concurrency limit are used in every mode.

| Configuration | Cold C1 | Cold C32 | Warm C1 | Warm C32 |
| --- | ---: | ---: | ---: | ---: |
| Flag off | 49.93 | 322.84 | 39.54 | 84.92 |
| KV only, GPU | 56.15 | 383.72 | 41.75 | 97.45 |
| KV only, HiCache | 57.36 | 394.13 | 42.82 | 109.51 |
| KV + taps, GPU | 56.26 | 386.21 | 42.10 | 101.45 |
| KV + taps, HiCache | 57.55 | 395.84 | 44.56 | 144.94 |

## Repeat and launch variation

Each percentage below compares that launch's median with the flag-off median
from the same pass. The second pass reverses mode order on the same GPU pair.
The last column includes all six measured throughput samples, including
outliers; no measured run was removed based on its timing or acceptance.

| Configuration | Workload | First launch delta | Second launch delta | Six-run tok/s range |
| --- | --- | ---: | ---: | ---: |
| KV only, GPU | cold, C1 | -0.58% | -0.61% | 3,881–3,887 |
| KV only, GPU | cold, C32 | -4.59% | -4.66% | 19,974–20,100 |
| KV only, GPU | warm, C1 | -2.39% | -2.28% | 49,173–49,340 |
| KV only, GPU | warm, C32 | -9.36% | -7.78% | 369,573–385,071 |
| KV only, HiCache | cold, C1 | -0.90% | -0.89% | 3,873–3,877 |
| KV only, HiCache | cold, C32 | -5.27% | -5.41% | 19,882–19,947 |
| KV only, HiCache | warm, C1 | -3.94% | -3.26% | 48,299–48,835 |
| KV only, HiCache | warm, C32 | -16.37% | -12.61% | 344,765–358,807 |
| KV + taps, GPU | cold, C1 | -0.78% | -0.68% | 3,875–3,882 |
| KV + taps, GPU | cold, C32 | -5.03% | -4.91% | 19,963–20,510 |
| KV + taps, GPU | warm, C1 | -2.52% | -2.78% | 48,901–49,363 |
| KV + taps, GPU | warm, C32 | -9.25% | -6.92% | 367,790–386,678 |
| KV + taps, HiCache | cold, C1 | -1.06% | -0.94% | 3,866–3,871 |
| KV + taps, HiCache | cold, C32 | -5.68% | -5.51% | 19,806–20,292 |
| KV + taps, HiCache | warm, C1 | -5.53% | -5.25% | 47,548–47,744 |
| KV + taps, HiCache | warm, C32 | -25.89% | -24.01% | 306,233–312,227 |

Flag-off warm C32 samples span 347,994–421,054 tok/s; its two launch medians
are 414,720 and 407,199 tok/s.
Warm C32 has more variation than cold throughput or warm C1, especially in the
flag-off baseline. Treat small differences there cautiously. Speculative
acceptance is recorded per request: some C32 batches have different acceptance,
so those throughput changes are not pure kernel-cost measurements. Two launches
and these ranges expose some variability; they are not a population confidence
interval. The synthetic prompts measure serving/cache behavior, not task accuracy.

## Memory at the same capacity

GiB per rank, averaged across the two ranks. Startup comes after model/graph
warmup; the largest sample comes from `nvidia-smi` immediately before/after
benchmark runs, including primes. This is sampled device usage, **not an
instrumented peak**. It includes weights, graphs, pools, and allocator caches;
it is distinct from the conservative admission reserves in the design document.

| Configuration | Startup GPU GiB | Largest sampled GPU GiB | Checkpoint pool GiB | Checkpoint storage |
| --- | ---: | ---: | ---: | --- |
| Flag off | 86.16 | 86.16 | — | — |
| KV only, GPU | 81.04 | 81.42 | 6 | GPU |
| KV only, HiCache | 75.22 | 76.20 | 6 | host |
| KV + taps, GPU | 93.04 | 94.89 | 18 | GPU |
| KV + taps, HiCache | 75.24 | 77.09 | 18 | host |

The host column describes only the DFlash checkpoint payload. Other HiCache host
allocations are additional. With fixed Full KV and recurrent capacities, this
matrix measures runtime overhead and observed allocation. It does not measure
the maximum target-cache capacity that each mode could admit automatically, nor
throughput under target-cache pressure.

## Protocol and scope

- Hardware/model: two H200 GPUs, TP2, BF16 Qwen3.6-35B-A3B, the local six-layer
  `/tmp/autospec/baseline` draft, 2,048-token draft window, 16-token draft block.
- The five original modes use unified memory, FA3, `extra_buffer_lazy`, page size 64, chunked
  prefill 4,096, breakable prefill graphs, decode graph batch limit 32, and
  static memory fraction 0.8. Full KV is fixed at **1,048,576 tokens**, recurrent
  capacity at **256 slots**, and maximum running requests at **32**.
- Python TreeCore is fixed in every mode, including flag off. This controls for
  backend differences; it is not the earlier comparison against flag off's
  default Rust TreeCore. Flag off has both DFlash cache flags and HiCache off.
- Cold: the existing serving CLI's seeded `random-ids` dataset, nominally 2,048
  input / 512 output tokens; 32 prompts at C1 and 128 at C32. Its default text
  decode/re-tokenization is unchanged, and throughput uses the CLI's dataset
  input counts. Cache is flushed after four CLI warmup requests. Seed: `20261021`.
- Warm: eight distinct 4,096-token parents are seeded outside timing after a
  cache flush. Every measured request adds a distinct 37-token suffix and
  generates 128 tokens; 128 requests at C1 and 512 at C32. The same CLI's timed
  HTTP requests and metric calculation are used with this custom dataset.
- HiCache modes use ratio 2 and write-through backups. Target KV and recurrent
  states stay resident in the warm workload. Cached draft checkpoint payloads
  use host storage, so their restore is timed even though target cache-hit
  metadata reports device residency. Forced target/recurrent host recovery was
  correctness-tested previously; it is not this performance workload.
- All measurements run serially on GPUs 4/5; server CPUs are pinned to 48–95,
  client CPUs to 0–7. There are no concurrent benchmark servers or profiling
  sessions. Each mode gets two fresh launches in reverse order, one full prime
  per cell and an additional prime for each warm cell, then three measured
  repeats per cell. This yields **120 measured runs and 60 excluded primes**.
- Weights stay fixed. The matrix does not time draft weight updates, stale-KV
  reprojection, training, or live update orchestration.

The short warm pilot was excluded before this matrix: its approximately
2-second C32 timing was too variable. The final workload is larger, although
warm C32 still exhibits measurable variation.

Artifacts and reproducibility:

- Full server/client commands, logs, startup configuration, GPU telemetry and
  per-run files: `/tmp/dflash-performance-matrix-20261022/stable/`.
- Driver and client: `stable/run.py` and `stable/client.py` under that directory's
  parent. Their hashes are in the full results file.
- Validated frozen runtime: `/tmp/dflash-deployment-split-20261022/final-source/python`.
- The full runtime tree was verified unchanged before and after the matrix.

## Flag off with HiCache

A follow-up on 2026-10-22 measured flag off with HiCache using the existing
separate-pool path. Both `--enable-dflash-swa-cache` and
`--enable-dflash-tap-cache` remain off.

The original unified-memory combination was also retried unchanged and failed
at startup: its draft KV is fused into target pages, while the existing HiCache
draft sidecar requires a separate device pool. The
[pool configurator](../python/sglang/srt/mem_cache/kv_cache_configurator.py)
rejects that combination to avoid allocating an unbudgeted private draft pool.
The successful control therefore removes `--enable-unified-memory` in **both**
arms and toggles only HiCache. No runtime code or validation guards were changed.

All other explicit capacities, GPUs, CPU affinity, models, Python TreeCore,
workloads and benchmark mechanics match the original matrix. Each mode has two
fresh launches, in reversed order, with three measured repeats per cell:
**48 measured trials, 24 excluded primes, and 9,600 measured requests**.

Median total input-plus-output tokens/s; percentages compare HiCache on with
HiCache off using the same separate-pool layout:

| Configuration (both DFlash flags off) | Cold C1 | Cold C32 | Warm C1 | Warm C32 |
| --- | ---: | ---: | ---: | ---: |
| Separate pools, HiCache off | 3,827 | 20,233 | 51,819 | 418,337 |
| Separate pools, HiCache on | 3,810 (-0.43%) | 20,108 (-0.62%) | 49,843 (-3.81%) | 407,244 (-2.65%) |

As above, warm totals include 4,096 cached input tokens per request. All warm
requests hit exactly that device-resident target prefix; cold requests report
zero cached tokens. The flag-off draft KV also stays on GPU for these hits.
This measures HiCache's serving/backup cost, not forced host recovery. The
bounded HiCache modes in the original matrix instead restore checkpoint draft
windows from host even when target KV remains on GPU.

| Workload | HiCache delta, first launch pair | HiCache delta, second launch pair | HiCache six-run tok/s range |
| --- | ---: | ---: | ---: |
| Cold C1 | -0.34% | -0.58% | 3,807–3,813 |
| Cold C32 | -0.57% | -0.67% | 20,093–20,145 |
| Warm C1 | -3.66% | -3.83% | 49,241–50,197 |
| Warm C32 | -2.66% | -5.20% | 377,324–416,658 |

Warm C32 has more variation than the cold cells. The full JSON retains every
sample and per-request acceptance summaries. No measured sample was excluded.

| Separate-pool configuration | Startup GPU GiB/rank | Largest sampled GPU GiB/rank |
| --- | ---: | ---: |
| HiCache off | 86.02 | 86.05 |
| HiCache on | 86.28 | 86.28 |

These are before/after GPU samples, not instrumented peaks. Host allocations
are additional. The separate-pool HiCache configuration backs up ordinary draft
KV through the existing sidecar, rather than the bounded checkpoint-window pool.

This adds a usable flag-off HiCache baseline, but **it is not a pure SWA-flag
ablation against the original unified-memory modes**: allocation/layout changes
between those configurations. HiCache alone produces a much smaller warm-hit
loss here than the bounded host-cache configurations in the original matrix;
these timings do not isolate the operations responsible for the remaining gap.

Evidence: `flag_off_hicache_followup` in the [JSON record](draft-swa-performance-matrix.json).
Full commands, logs, samples and startup metadata are under
`/tmp/dflash-flag-off-hicache-20261022/separate-matrix/`; the rejected unified
launch is recorded in `unified-probe.json` and `unified-startup.log` in its parent.

## What this suggests for the next discussion

At cold C32, the separate-pool flag-off control without HiCache is 3.81% below
the earlier unified flag-off baseline; adding HiCache costs another 0.62%
within the separate-pool comparison. Allocator and layout differences therefore
merit profiling alongside bounded-cache bookkeeping. These measurements do not
isolate the individual operations responsible for the gap.

For the raw-history policy, the warm HiCache comparison is directly relevant:
valid materialized KV avoids projection, but loading the accompanying raw
history still costs throughput. Deferring or overlapping that history load is
worth evaluating before choosing a default policy. Independent eviction of
materialized KV is a separate capacity tradeoff and was not exercised here.
