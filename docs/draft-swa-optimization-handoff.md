---
title: "DFlash bounded-cache optimization handoff"
description: "Current implementation, reproduced throughput, staging-memory accounting cleanup, validation, and retained checkpoint state."
---

# DFlash bounded-cache optimization handoff

State updated on **2026-10-22**, in `/home/modal/sglang`. The subsequent
[deployment split](draft-swa-deployment-shapes.md) is now implemented:
`--enable-dflash-swa-cache` retains materialized checkpoint KV, the separate
`--enable-dflash-tap-cache` adds raw target features, and HiCache is optional.
See that document for the current memory costs and validation. The performance
numbers below were measured **before** this split and must not be attributed
to its new payloads or storage modes.

The [deployment performance matrix](draft-swa-performance-matrix.md) now records
120 measured runs across two fresh launches per mode, holding Python TreeCore
and pool capacities fixed. It finds a remaining cold C32 cost in the bounded
path and an additional warm-hit cost when raw taps are loaded through HiCache.
Use that report for the new deployment shapes; the results below remain the
history of the preceding implementation.

The preceding optimization state is recorded below. The completed
[core minimization pass](draft-swa-core-minimization.md) has removed redundant
checkpoint-donation checks and an unused page-only kernel specialization:
**50 net lines in two runtime files**.
The larger 171-line cleanup repeatedly lost about 0.21% at C1, so the fused
serving kernel and its shared helper were restored. The short-tail guard is
also retained: removing two lines adds identity gathers without a substantial
simplification. Final throughput changes were **+0.053% at C1** and **+0.030%
at C32**, within observed variation. The final source passed **48 tests and
132 subtests**, plus forced-host recovery of 4,096 host-cached tokens with
zero device-cached tokens and a 37-token new suffix. All validation servers
are stopped.
Test minimization is deferred, and the checkpoint index is preserved.

The preceding staging-memory cleanup reserves HiCache staging payload before
sizing bounded DFlash pools and shares the geometry with allocation. The
ablations include that corrected budget, with fixed pool capacities in both
arms. They compare individual removals, not flag on against flag off.

**Flag-on/off parity has not been established.** The latest such reproduction,
before staging accounting, measured **−0.49% total throughput at concurrency 1**
and **−0.56% at concurrency 32**. Earlier launches of that code measured about
−0.5% at C1 and −0.6% to −1.3% at C32. Those figures are historical evidence,
not measurements of the current cleanup.

Use this document for current state. The
[optimization report](draft-swa-throughput-optimization.md) contains the parity
pass, independent reproduction, accounting cleanup, and dated historical
experiments. [draft-swa-parity-pass.json](draft-swa-parity-pass.json) preserves
compact evidence for the parity pass and reproduction. The larger
[draft-swa-throughput-optimization.json](draft-swa-throughput-optimization.json)
records the earlier work; its `retained_flag_gpu_crossover` entry is the
**2026-10-21** comparison, not the current result. That older comparison measured
−1.03% at C1 and −3.46% at C32 with a different harness and code state.

## Working tree and checkpoint

| Item | Current state |
| --- | --- |
| Branch | `dcw02/autospec-v1` |
| Local HEAD | `6f168a98d8438f3cb8bfd7ecfbfd9c2efa15b5e2` |
| Pushed checkpoint | `4f895d40ed8fc0f698040e1f31e21eb3f6c2e4c4` — `WIP: checkpoint bounded DFlash KV and raw-tail recovery` |
| Remote | `origin`, `git@github.com:modal-projects/sglang.git` |
| Index | Exactly the pushed checkpoint tree; **75 staged files** |
| Later tracked changes | Earlier cleanup plus the deployment split; use `git diff --stat` for the current count |
| Untracked documents | This handoff, parity/core-minimization reports, and the deployment-shape report/validation record |

The checkpoint was committed, pushed, then soft-reset at the user's request.
Preserve this arrangement. No subsequent commit or push has been made. The
later work includes CPU tracking-slot ownership, the four parity-pass changes
below, the staging-accounting cleanup, their tests, and measurement records.
The core minimization added no test changes; the subsequent deployment split
adds flag, revision, sizing and GPU-slot-lifetime coverage. Working-tree changes
overlap the staged checkpoint file list.

Useful read-only checks from the repository root:

```bash
git status --short --branch
git diff --stat
git diff --cached --stat
git diff --cached --quiet 4f895d40ed8fc0f698040e1f31e21eb3f6c2e4c4
```

The last command should exit zero while the checkpoint index is preserved.
The independent reproduction also verified that all 5,857 frozen `python/`
files and the test files still matched the working source after measurement.

## Feature and invariants to preserve

The agreed design is in [draft-kv-checkpoint-design.md](draft-kv-checkpoint-design.md).
The legacy compact draft cache is excluded. Native sliding-attention layers
bound draft context; physical storage uses the existing unified Full/SWA/Mamba
arena and page mappings.

`--enable-dflash-swa-cache` defaults off and selects bounded active draft KV
plus persistent materialized checkpoint windows. It requires DFLASH, unified
memory, a compatible Full + Mamba target and Python TreeCore. HiCache is now
optional: cached context windows reside on GPU without it and on host with it.
`--enable-dflash-tap-cache` separately adds raw history for compatible draft
updates and requires the SWA flag. Active context history stays on GPU.

A valid materialized checkpoint avoids historical draft projection. A stale
revision rebuilds from taps when enabled, preserving the target prefix hit.
Both representations currently share a bounded checkpoint lifetime. Recurrent
state residency remains independent of the context payload's residency.
The internal revision-invalidation hook is implemented; the live update
coordinator and its cross-rank drain barrier remain outside this change.

Ownership contracts:

- A checkpoint and its context window describe the same exact token boundary.
  Pending captures, replaced tracking slots, cancellation, and donation preserve
  that association. Rejected speculative rows never enter live history.
- Immutable chunks can be shared by overlapping windows. Appends may fill unused
  suffixes but cannot overwrite data held by a checkpoint or transfer.
- GPU sources stay alive through transfer completion. Host-slot reuse waits for
  the previous occupant's final reader/writer, including after logical release
  or an abandoned reservation. Logical host availability stays independent of
  rank-local DMA completion so TP ranks make the same donation decisions.
- A nonzero draft binding does not prove valid materialized contents. Full-page
  frees now leave bindings as spares. Each page entering a live window must be
  written, restored, or repaired after canonicalization.
- Full/SWA/Mamba admission caps remain independent. Unused draft or recurrent
  reservations are not lent to Full KV. Recurrent slots need admission checks
  even when their incremental Full-token charge is zero.

The user prioritizes correct v1 foundations, a small SGLang diff, and a
minimization pass after each feature. Preserve useful checks and explicit
ownership. Dropping recovery payloads or weakening transfer fences is not a
valid throughput improvement.

## Retained implementation

The pushed checkpoint contains bounded draft provisioning, fused accepted-row
capture and immutable chunks, exact host-tail retention/recovery, strided
recurrent transfers, shared HiCache D2H scheduling, and independent recurrent-slot
admission. The fused serving window kernel remains; its page-only sibling and
fast dispatch are removed because the production caller cannot reach them.
The shared rebinding helper stays because the fused kernel calls it. The report
preserves implementation entry points and the earlier validation rounds.

CPU recurrent tracking-slot ownership was added after that checkpoint:
`ReqKvInfo.mamba_ping_pong_allocated` records allocation bits while device slot
indices remain authoritative for kernels. Allocation, replacement, donation,
retraction, and cleanup maintain both. Planning avoids GPU scalar reads and
boolean filtering; integer slot clears use `fill_`. This common optimization
also benefits flag off.

The 2026-10-22 parity pass retains four further changes:

| Change | Implementation and reason |
| --- | --- |
| Host-recorded canonical remaps | `unified_radix_cache.py` records affected ranges in `DFlashFeatureState`; `dflash_cache.py` repairs those rows from the live tail on the next preparation. Replaces per-prefill table copies and a blocking device comparison. |
| Prefill synchronization removal | `schedule_batch.py` keeps tracking indices on-device; `memory_pool.py` and `dflash_worker_v2.py` use pinned asynchronous uploads; GDN checkpoint planning uses existing host mirrors. Tensor bounds and slot clears avoid scalar transfers. Applies to both controls where relevant. |
| Copy-engine HiCache backups | `pool_host/mamba.py` stages two recurrent slots at a time. The existing staged MHA/MLA transfer kernels now accept source strides, allowing unified target-KV rows to use the same path. Unsupported layouts keep their fallback paths. |
| Spare bindings survive target-page frees | `allocator/dflash_swa.py` releases Full pages without filtering draft bindings on the GPU. Existing bindings remain inside the fixed reservation and are reused; canonical-remap repair preserves content validity. |

Review the binding lifecycle and canonical repair together. Keeping a binding
removes the release-path readback only because contents are validated by the
window lifecycle. The common synchronization changes and generic HiCache
transfer changes can be reviewed as separate groups.

## Throughput reproduction and validation

Reproduction before the staging-accounting cleanup on GPUs 4/5 enabled and 6/7
disabled, both on the same NUMA node. Metric: total input plus output tokens/s.
Each cell is the median of three serial repeats in alternating order, after
excluded primes.

| Concurrency | Flag on | Flag off | On versus off |
| --- | ---: | ---: | ---: |
| 1 | 3,882.87 | 3,902.01 | −0.49% |
| 32 | 20,911.85 | 21,029.17 | −0.56% |

C1 median TTFT was 53.156 versus 50.629 ms; time per output token differed by
0.12%. Acceptance length was 5.1512 for both C1 variants and 4.9893/4.9902 at C32.
All measured request and token counts matched. This reproduction confirms the
smaller gap on fresh servers; it is not another GPU-assignment crossover or a
confidence interval. The parity pass observed enabled launch-to-launch variation
up to a −1.30% C32 gap.

The independent check passed 680 unit tests, 214 CUDA transfer tests, and five
GPU host-tail tests, with 59 unit cases skipped. Default and forced-host TP2
serving runs passed warm reuse, 2,400-token decode, and 32 branches. All five
recorded output hashes in each run matched
[draft-swa-serving-validation.json](draft-swa-serving-validation.json).
Forced recovery restored 4,096 host-cached tokens with zero device-cached tokens
and extended the 37-token suffix.

The preceding parity pass separately recorded 693 unit and 214 CUDA tests plus
serving checks. These rounds overlap; their counts are not additive. Preserve
ownership, remap, no-readback, fallback-layout, and host-slot race coverage during
cleanup. A negative control for the old-reader fence failed the race test in an
earlier round, establishing that the fence is necessary.

After staging-accounting cleanup, **143 unit tests and 214 CUDA transfer tests**
passed. The new budget regression fails on the pre-cleanup source by the exact
missing reserve. Automatic-size and 131,072-token-capped TP2 serving checks
passed long decode, reuse, and 32 branches, with all five reference output
hashes matching. The capped run forced recovery of 4,096 host tokens with zero
device tokens. Its cap and arena size were preserved; the automatic-size launch
held 4,462,464 Full-KV tokens. These are correctness/capacity checks, not a new
throughput comparison. All validation servers were stopped.

## Environment and reproduction artifacts

| Setting | Current reproduction |
| --- | --- |
| Hardware | NVIDIA H200, CUDA, TP2 |
| Target | `Qwen/Qwen3.6-35B-A3B`, snapshot `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Draft | `/tmp/autospec/baseline`; six sliding-attention layers, hidden width 2,048, eight KV heads, head dimension 128 |
| Raw features | Six target taps `[10, 15, 21, 26, 32, 37]`; width 12,288 |
| Attention and precision | FA3 for target/draft; BF16; window 2,048; draft block 16; page size 64 |
| Limits | 32 running requests; 256 recurrent slots; chunked prefill 4,096; context 16,384 |
| Cache and graphs | `extra_buffer_lazy`; static memory fraction 0.8; decode graph maximum batch size 32; breakable prefill graphs |
| Enabled | DFlash SWA cache, HiCache ratio 2 with `write_through`, Python TreeCore |
| Disabled | No HiCache, Rust TreeCore; unified memory remains enabled |
| CPU affinity | Both timed servers use CPUs 48–95 |
| Workload | `random-ids`, 2,048 input/512 output tokens, range ratio 1, seed 20261021, request rate `inf` |
| Counts and warmup | 32 requests at C1, 128 at C32; flush per run, four warmup requests, excluded prime per variant/concurrency |
| Capacity | No `--max-total-tokens` cap in throughput runs; forced-host validation alone uses 131,072 |

This compares the combined bounded-cache configuration, including HiCache and
TreeCore choice. It does not isolate raw-tail bandwidth. Automatic sizing can
give the variants different capacities. The synthetic workload does not measure
agentic workload distributions or the benefit of avoiding historical target
replay after a prefix hit.

Artifacts under `/tmp/dflash-review-repro-20261022/`:

| Artifact | Purpose |
| --- | --- |
| `results.json`, `pipeline.log` | Completed tests, serving checks, all timing runs, comparisons, server shutdown |
| `on-server.json`, `off-server.json`, `on_fixed-server.json` | Exact launch commands, GPU assignments, historical PIDs |
| `*-startup.json` | Effective server arguments |
| `*-c*-repeat*.jsonl`, `*-c*-prime.jsonl` | Measured repetitions and separately excluded primes |
| `on-serving.json`, `on_fixed-serving.json` | Generated tokens and prefix residency evidence |
| `run.py`, `source-manifest.json`, `python/` | Reproducer and full frozen Python source; digest `f866ad8c07a81a98cd7b8486e14427a02f449efc7f9f66a742f61658c0452f86` |
| `final-verification.json` | Source/test and result hashes, HEAD/index/status checks before documentation reconciliation |

Compact results and hashes are also retained under `independent_review_reproduction`
in [draft-swa-parity-pass.json](draft-swa-parity-pass.json). Full `/tmp` artifacts
are host-local and are not embedded by those references.

The subsequent accounting cleanup has its own
`staging_memory_accounting_cleanup` entry in that JSON and artifacts under
`/tmp/dflash-staging-cleanup-20261022/`. They include the regression's negative
control, per-file test logs, serving outputs, launch commands, edited source
copies, and final verification. The capped launch was retried on a new port
after the harness's immediate port reuse failed; both serving runs passed.

The old `/tmp/dflash-parity/manage.py` selectors can replace files with historical
snapshots or apply prototypes. In particular, `reqmap_control_cross` restores old
runtime files. Use each core-minimization case's frozen source and manifest for
the current comparisons; the earlier reproduction remains historical evidence.
Verify source manifests and actual flags before any future comparison. Run
benchmarks serially without tests, startup, profiling, or heavy analysis alongside
them. Confirm process ownership before stopping a saved PID.

## Memory contract and staging accounting

For the measured 32-request, TP2, 4,096-token-prefill configuration:

| Resource | Per-rank amount |
| --- | ---: |
| One BF16 raw window | 48 MiB |
| Latest windows for 32 active requests alone | 1.5 GiB |
| Total raw GPU admission reserve | 9.74267578125 GiB |
| Materialized draft pool | 69,632 rows; 0.796875 GiB |
| Raw host pool | 256 windows; 12 GiB, additional to ordinary HiCache pools |
| Enabled Full-token budget in the pre-cleanup throughput run | 4,472,832 tokens |
| Enabled Full-token budget in the cleanup validation launch | 4,462,464 tokens |
| Recurrent GPU staging payload, now reserved before sizing | 61.40625 MiB |
| Target K/V GPU staging payload, now reserved before sizing | 40 MiB |

The raw reserve bounds overlapping ownership, including recurrent candidates,
queued captures, transfers, and projection temporaries. It is not a measured
resident allocation. Shared immutable chunks reduce duplication but cannot
assume all valid checkpoints overlap. Transfer sources are bounded at
`2 × max_requests` across D2H/H2D, or 64 here. Raw features are replicated across
TP ranks; draft KV is sharded. See `DFlashCacheSizing.raw_gpu_reserve_bytes` and
[the memory calculator](draft-kv-memory-budget.html) for that separate formula.

The short-tail selection guard is retained. The core cleanup does not change
temporary allocations, the raw reserve, or cache capacities.

The new staged transfer paths allocate another **101.40625 MiB per rank** of
persistent GPU tensor payload. This is shared scratch: two recurrent checkpoint
slots and 64 target-KV pages per transfer chunk, reused across requests. It is
independent of per-request ping-pong checkpoint candidates and raw-tail storage.

The HiCache host-pool constructors still own these allocations after initial KV
sizing. The bounded DFlash path now reserves their payload before selecting pool
capacity, alongside its separate raw-feature reserve. Allocation and budgeting
use the same staging-shape helpers and target Full/Mamba specifications.
The maximum KV chunk is reserved before host capacity is known; very small host
pools can allocate less. Layout and kernel fallbacks retain their behavior.

Bounded DFlash already rejects post-capture sizing, so this charge cannot overlap
with the existing MHA preparation-before-profiling path. Other sizing paths keep
their existing behavior. Transfer capacities, ownership, and the raw-tail budget
were preserved. The report contains the
[allocation trace, byte calculation, and scope](draft-swa-throughput-optimization.md#hicache-staging-memory-accounting-2026-10-22).
No OOM was observed and this is not an established cause of the throughput gap.

## Profiling conclusions and experiment disposition

The parity pass supersedes the older dependency ranking:

- Host synchronizations compounded. A readback was much slower when queued
  HiCache copies were present; removing one wait in isolation could hide its gain.
- Direct GPU-to-host transfer kernels occupied SMs during PCIe traffic and slowed
  overlapping model graphs. Staged copy-engine backups reduced that interference.
- The remaining C32 cost is concentrated after prefills, with additional CPU work
  for HiCache submission, raw-tail donation/consumption, checkpoint bookkeeping,
  and readiness consensus. Common prefill-launch CPU code also ran 1–2% slower
  with the flag enabled; the cause remains unknown.
- C1's remaining cost is predominantly TTFT. The fresh unprofiled reproduction
  confirms the small decode-time difference but does not independently re-profile
  the causal contributions.

| Experiment or mechanism | Current disposition |
| --- | --- |
| One asynchronous request-index upload shared by both maps | Retained: the current C01 removal costs 2.02% at C32. This supersedes the October 21 isolated result. |
| Keep draft bindings across Full-page frees | Retained with canonical-remap repair after the other waits were removed. |
| Staged recurrent gather followed by copy-engine DMA | Retained in the parity pass; supersedes the earlier isolated rejection. |
| CPU prefix-remap version guard | Rejected. Current host-recorded remap ranges are a different mechanism. |
| Direct capture into persistent chunks with complete destination reserve | Rejected; the bounded retest added 0.844 GiB/rank without a useful gain. |
| Queue raw backups until cleanup | Rejected; raw D2H still starts during donation. |
| Larger raw-host/recurrent capacities or changed write policy | Not adopted; measured tradeoffs remain in the historical report. |
| Higher stream priority, fused mapped-host raw D2H, immediate or batched multi-tensor raw copies | Not retained; no useful serving improvement. |
| Pinned count readback during release | Replaced by retaining spare bindings; still waited for the fenced forward. |
| Skipping readiness consensus when the local queue is empty | Not retained; no TTFT gain and no rank-safety guarantee. |

Diagnostic omission of capture or host payloads invalidates recovery. Its timings
cannot count as parity, and isolated omission gains are not additive. Older
wrong-port, overlapping, or unprimed runs remain excluded as documented.

## Cleanup and remaining work

Documentation is reconciled and the bounded path's staging-memory accounting is
implemented. Its minimization pass kept the change in the existing configurator
and host-pool modules, with no new allocation owner or configuration option.
The common synchronization changes, generic HiCache transfer support, and
DFlash binding/remap lifecycle remain coherent groups for any broader cleanup,
preserving the invariants and useful tests above.

The [core minimization pass](draft-swa-core-minimization.md) completed the
individual comparisons and rejected the larger cleanup after combined testing.
The final 50-line cleanup passed throughput and forced-host validation. The
5,846-file working Python tree matches the measured source, all 2,762 test files
are unchanged, and HEAD plus the 75-file checkpoint index are preserved. Ruff
lint/format and diff whitespace checks pass. Test minimization remains deferred.
Historical harness selectors and October 21 graph-analysis plans remain
provenance, not current instructions.

Live draft-weight revision orchestration, lending unused reservations, Rust
TreeCore bundle hooks, and raw-tail L3 persistence remain separate design work.
