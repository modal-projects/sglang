---
title: "DFlash cache deployment shapes"
description: "Materialized checkpoint KV, optional target taps, GPU and HiCache storage, revision recovery, and validation."
---

# DFlash cache deployment shapes

Implemented on 2026-10-22. This document supersedes the raw-only checkpoint
policy in the earlier [design investigation](draft-kv-checkpoint-design.md)
and [optimization handoff](draft-swa-optimization-handoff.md). The earlier
performance reports describe an older implementation; their throughput numbers
do not measure this deployment split.

The subsequent [performance matrix](draft-swa-performance-matrix.md) measures
all four shapes against flag off at concurrency 1 and 32, with cold prompts and
warm prefix hits. It records the remaining throughput costs and observed GPU
allocation at fixed capacities.

## Flags and behavior

Both flags default to false. Neither model compatibility nor HiCache enables
them implicitly.

| SWA cache | Tap cache | HiCache | Active request | Cached checkpoint |
| --- | --- | --- | --- | --- |
| Off | Off | Either | Existing DFlash path | Existing behavior, subject to its configuration restrictions |
| On | Off | Off | Bounded materialized draft KV and checkpoint capture history | Materialized window on GPU |
| On | Off | On | Same | Materialized window on pinned host memory |
| On | On | Off | Same, plus raw target taps on GPU | Materialized window and raw taps on GPU |
| On | On | On | Same, plus raw target taps on GPU | Materialized window and raw taps on pinned host memory |

`--enable-dflash-swa-cache` selects the bounded attention pool and saves
materialized draft KV with recurrent checkpoints. `--enable-dflash-tap-cache`
adds unprojected target features and requires the SWA flag. The tap flag by
itself is a configuration error.

The bounded path still requires DFLASH, unified memory, an all-sliding draft
with unquantized half/BF16 KV, and a Full-attention + Mamba target with recurrent
tracking buffers. It requires an explicit maximum running-request count and
uses Python TreeCore. This change removes the HiCache prerequisite; it does
not expand support to legacy compact draft caching, disaggregation, sessions,
or arbitrary model geometries.

For this Full-attention + Mamba target, flag-off DFlash with HiCache uses
the existing separate-pool path: omit `--enable-unified-memory`. The unified
fused-draft configuration rejects HiCache. The
[performance matrix](draft-swa-performance-matrix.md#flag-off-with-hicache)
records the startup check and paired flag-off measurements.

For fixed draft weights, enable only the SWA flag. Raw target hidden outputs
still exist transiently when projecting new tokens, but no raw history is
retained. For compatible draft updates, enable both flags. HiCache independently
chooses the cached checkpoint payload's storage tier.

## What is retained

At a reusable prefix boundary C, a complete bundle associates:

1. The existing target prefix identity and target Full KV.
2. The recurrent state at C, including its convolution state.
3. Materialized draft KV for the window ending at C, tagged with a draft revision.
4. The corresponding raw target taps, only when the tap flag is enabled.

Materialized KV includes the draft's projections, key normalization and rotary
positions. Raw taps are the configured target hidden features before draft
projection. A draft-only update can invalidate the former while preserving the
latter. Target weight or tap-layout changes are outside this recovery contract.

Active attention continues to use the existing bounded unified SWA pool and
page mappings. Persistent checkpoints own copied context snapshots; they do
not keep historical SWA pages pinned. The request also retains an append-only
snapshot history for exact checkpoint capture and repair after canonical page
remapping. Consequently, the materialized representation has a snapshot cost
in addition to the active attention pool.

The existing capture, donation and transfer lifecycle now handles a context
row containing each layer's K and V, followed by optional raw features. Byte
packing preserves the target and KV dtypes separately. Overlapping live and
checkpoint windows share immutable chunks; appending never overwrites rows
already held by a checkpoint or transfer.

This implementation retains both representations together when taps are on.
They share one checkpoint slot and one eviction lifetime. It does not yet
provide independent eviction of materialized KV while retaining raw taps.

## Reuse and draft revisions

On a prefix hit, the target computes only the uncached suffix. For old context:

- If the saved KV revision matches, restore those rows and project only the new
  target outputs needed by the current or checkpoint window.
- If the revision is stale and taps are present, project the retained raw
  window under current draft weights, restore the required active KV rows, and
  refresh the saved checkpoint. Its next reuse can take the materialized path.
- A complete new window can make loading the old payload unnecessary. An
  interior recurrent checkpoint may still need old rows even when the final
  active window does not.

Matching validates a usable context payload at the exact recurrent boundary.
A target-only prefix without a complete bundle is not advertised as a reusable
bounded-DFlash checkpoint. Recovery from a complete bundle never requests
historical target replay.

`DFlashWorkerV2.invalidate_draft_context()` advances the shared draft revision
and refreshes any pre-stacked projection helper. Active windows and saved
checkpoints recover lazily from their taps. This is an internal hook, not a
server endpoint or an autospec update coordinator. Its caller must drain
in-flight forwards and captures, update compatible draft weights, and invoke
the hook on every participating rank before admitting more work. Targets and
tap identities must remain fixed. Calling it without tap retention fails.

Active windows recover before their next draft use. Existing checkpoint
snapshots keep their own revisions even if the live request advances. This
prevents an old checkpoint from being mislabeled with newly projected weights.

## Storage tiers and bounds

The checkpoint pool has `max_mamba_cache_size` logical slots. This is a fixed
capacity, not an unbounded cache and not a promise that every host recurrent
checkpoint can keep a context window. A bundle that cannot obtain context
capacity is not published as a complete reusable checkpoint.

Without HiCache, the pool is a fixed GPU tensor. With HiCache, it uses pinned
host storage and the existing L2 transfer engine. Recurrent state residency
remains independent: a request may need a context load, a recurrent-state load,
both, or neither. GPU slot reuse and host slot reuse both wait for the previous
occupant's final access. Logical capacity release does not wait for rank-local
copy completion, preserving consistent TP admission decisions.

GPU-only checkpoint storage is charged when sizing recurrent capacity, including
both explicitly configured and automatically fitted capacities. Host checkpoint
storage is charged to HiCache's host budget. The active context reserve is
charged in both cases; HiCache staging is charged only when HiCache is enabled.

For the local six-layer draft, a 2,048-token window, BF16, TP2, 32 active
requests, a 16-token block, page size 64, chunked prefill 4,096, overlap enabled,
and 256 checkpoint slots, the costs **per rank** are:

| Allocation or admission reserve | SWA only | SWA + taps |
| --- | ---: | ---: |
| Materialized KV per checkpoint window | 24 MiB | 24 MiB |
| Raw taps per checkpoint window | 0 | 48 MiB |
| Fixed checkpoint pool, 256 slots | 6 GiB | 18 GiB |
| Active attention KV, 69,632 token slots | 0.797 GiB | 0.797 GiB |
| Context capture/ownership GPU reserve | 4.989 GiB | 14.778 GiB |

The fixed checkpoint pool is GPU memory without HiCache and host memory with
HiCache. The context reserve is a conservative admission bound, not an eagerly
allocated tensor or a measurement of live occupancy. It covers one live and
two candidate checkpoint windows per active request, boundary-chunk slack,
bounded transfer leases, overlapped captures and projection temporaries. The
current bound retains the same transfer-lease allowance in GPU-only mode.
That conservative sizing can be revisited separately from payload retention.

Raw taps are replicated across TP ranks; draft KV is sharded. The 48 MiB raw
window therefore stays constant at TP1/TP2/TP4, whereas materialized windows
are 48/24/12 MiB per rank. The existing HiCache staging reserve for this
configuration is another approximately 101.4 MiB per rank when enabled.
Target Full KV, recurrent state, verify intermediates, weights, and graph
allocations are separate from the table. These figures are costs, not a
claim of net memory savings for every deployment.

## Validation

The targeted CPU suite passes 214 tests and 290 subtests. It covers flag
validation, pool sizing, capture/donation/cancellation, canonical remapping,
checkpoint reuse and revision invalidation. A worker-level test changes draft
projection weights: a valid hit projects only two new rows; a stale hit
projects four retained rows plus the two new rows; the next hit again projects
only the two new rows. It also checks mixed raw/KV dtypes and scale handling.

The broader unified-tree suite passes 544 tests and 39 subtests (59 skipped).
The CUDA suite passes 13 tests and 62 subtests, covering capture/window kernels,
host transfers and GPU-only slot reuse with a previous reader still in flight.
Serving validation uses the local Qwen3.6-35B-A3B and DFlash baseline on H200,
BF16, TP2. All four SWA-enabled shapes pass warm reuse, a 2,400-token decode,
shorter-prefix reuse, and 32 concurrent branches. Both HiCache shapes also
recover 4,096 cached tokens entirely from host after capacity pressure. All
37 output sequences per shape match the preceding implementation's reference.
See [the validation record](draft-swa-deployment-validation.json) for the final
source, deployment matrix and results.

Live update orchestration is not exercised by the serving harness; revision
recovery is tested through the worker and residency interfaces. Throughput is
measured separately in the [performance matrix](draft-swa-performance-matrix.md);
the new bounded path has not reached flag-off parity.

## Questions reserved for discussion

1. **Loading raw history on a valid materialized hit.** With taps enabled,
   active requests need raw history for a possible later update. The current
   policy loads the complete payload, so saved KV avoids projection but not
   necessarily raw transfer. Deferring that transfer would need a retained
   source lease and a rule for when an active request obtains its raw history.
2. **Independent retention under pressure.** Both payloads are bounded and
   retained together initially. Evicting only materialized KV could save space
   while keeping update-safe recovery, at the cost of projection on reuse. It
   would require separate validity and capacity accounting; it is not part of
   this implementation.

## Code map and minimization

- `dflash_cache.py`: active SWA residency, revision repair and sizing.
- `dflash_feature_tail.py`: opaque context layout, immutable windows and capture.
- `pool_host/dflash.py`: one checkpoint pool with GPU or HiCache storage.
- `dflash_worker_v2.py`: capture materialized KV, prefer saved rows, reproject
  stale windows, expose the drained-update hook.
- Mamba cache component: exact-boundary association, usable matches and donation.
- KV configurator/builder and HiCache autosizing: construction and accounting.

The minimization pass kept the existing attention allocator, window kernels,
capture machinery and donation lifecycle. KV and optional taps are packed once
per projected batch instead of launching a separate snapshot copy per layer.
No new attention allocator, eviction policy, training coordinator, or public
update API was introduced. Module filenames remain unchanged to avoid a
separate relocation diff.
