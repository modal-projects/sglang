# DFlash clean-room rewrite spec v4

## 0. Purpose

This document is the implementation spec for a second DFlash rewrite attempt.
It is intentionally **clean-room with respect to the failed rewrite source code**.
It may use:

- the optimized old SGLang DFlash v2 behavior and profiling
- the vLLM DFlash implementation and profiling
- cross-tree correctness and performance notes
- explicit new experiments proposed in this document

It must **not** encode any branch-specific structural assumptions from the failed rewrite as fixed truths.
Kernel shape, graph boundaries, and fusion decisions are treated as an **experiment space** unless this document explicitly locks them.

The goal is:

- maximum DFlash throughput on the primary benchmark path
- clean correctness on the primary backend pair
- aggressive overlap with late host correction
- GPU-first state ownership
- minimal Python in the hot path
- enough structure and method signatures that Codex can implement it directly

---

## 1. Source-of-truth findings that are locked

These findings are treated as design constraints.

### 1.1 Primary benchmark and backend target

Primary bring-up and benchmark pair:

- target attention backend: `trtllm_mha`
- draft attention backend: `fa4`

Secondary comparison path only after the primary path is correct and competitive:

- `fa4` target + `fa4` draft

### 1.2 Fixed architectural facts

1. Draft execution must go through **draft-local model handles**.
2. Draft `embed_tokens` and `lm_head` are **aliased to target weights**, but execution is still draft-local.
3. Draft vocab equals target vocab.
4. Target verify stays **greedy top1-first** on the performance path.
5. DFlash overlap output processing is keyed by **`req_pool_idx + generation`**, never by row order.
6. DFlash owns its own graph rings, host-packet rings, and replay metadata.
7. Scheduler must not reach into draft-runtime internals.
8. CPU request fields are mirrors only. The GPU loop never trusts CPU `seq_lens`, CPU optimistic token ids, or CPU row order for next-step truth.

### 1.3 Performance facts to respect

1. Target-side forward time is already in the same performance class between optimized old-v2 and vLLM.
2. The remaining gap is mostly DFlash-specific plumbing: block prep, state handoff, materialization, draft sampling, overlap integration, and graph-friendly ownership.
3. Old-v2 improvements that held up were:
   - smaller DFlash-v2 setup cleanup
   - slimmer overlap payloads
   - fused dense draft block prep
   - dense commit compute with GPU-side write filtering
4. Rejected old-v2 ideas that should not be assumed good:
   - Python-side valid-row compaction before append
   - valid-only sequential writes after compaction
   - stacked direct KV scatter as a default answer
   - target-verify top1 micro-optimizations as the next main lever

### 1.4 Overlap facts to respect

1. Overlap should be **aggressive and optimistic**, with host correction applied later.
2. GPU tensors and the GPU request-state table are the source of truth.
3. The CPU scheduler is allowed to optimistically keep requests in flight, but it must never be the source of truth for:
   - the last committed token
   - the next verified token
   - the exact committed sequence length used by the next GPU step
4. The design must explicitly avoid the class of bug where optimistic CPU sequence lengths point past the committed token tail.

---

## 2. Success criteria

### 2.1 Correctness gates

The rewrite is not considered alive until all of these are true on `trtllm_mha + fa4`:

1. Greedy single-request prompts produce correct output text.
2. First drafted block is correct on a simple prompt before any later-step commit append could matter.
3. `spec_accept_length` and `spec_accept_rate` are sane and non-collapsed.
4. Prefill + first decode step + second decode step are all correct with overlap enabled.
5. `flush_cache()` preserves the permanent DFlash dummy request lane and dummy token reservation.
6. On request finish, CPU `kv_allocated_len` is synchronized from GPU `reserved_len` before release.

### 2.2 Performance gates

Primary benchmark target is B200 GSM8K with the same model pair and methodology already in use.

Minimum launch target for the rewrite to remain the mainline direction:

- at least old-v2 parity at concurrency 1 and 32 on the primary backend pair
- after parity, next target is beating vLLM on the same backend pairing where possible

### 2.3 Profiling gates

Every hot-stage optimization must be justified by:

1. stage timers with CUDA events for iteration speed
2. at least one validating `nsys` run for ground truth
3. e2e tok/s on concurrency 1 and concurrency 32

No kernel is accepted purely because it wins a microbench.

---

## 3. Design principles

1. **Keep semantic truth on GPU.**
2. **Keep Python off the hot path.**
3. **Overlap any unavoidable D2H copies.**
4. **Prefer fixed-shape dense GPU work over dynamic Python tensor surgery.**
5. **Do not assume a specific materializer kernel shape is optimal; prove it.**
6. **Prefer small, high-leverage fused jobs over a full-model megakernel.**
7. **Use page-aware reservation and mapping; do not bypass existing paged-cache semantics.**
8. **Design for graph capture from day one, but do not force all stages into one graph without evidence.**

---

## 4. Top-level runtime architecture

### 4.1 Objects

#### `DFlashCoordinator`
The only scheduler-facing execution object.

Owns:

- one `TargetRuntime`
- one `DraftRuntime`
- one `DFlashReservationManager`
- one `DFlashRequestStateTable`
- one `DFlashGraphRing`
- one `DFlashHostPacketRing`
- DFlash-specific replay metadata
- DFlash-specific batch handles

#### `TargetRuntime`
Rank-local target runtime.

Runs:

- target prefill with hidden capture
- target verify forward
- target top1 path

#### `DraftRuntime`
Rank-local draft runtime.

Runs:

- draft block forward (non-causal)
- draft embedding path
- draft lm-head top1 path

#### `DraftKVMaterializer`
First-class runtime component.

Runs:

- prompt hidden -> draft KV materialization
- committed verify hidden -> draft KV materialization

It is not a helper buried inside another worker.

#### `DFlashReservationManager`
Owns DFlash-specific reservation/headroom policy and page-aware req-to-token growth.

#### `DFlashRequestStateTable`
Persistent per-request GPU state keyed by `req_pool_idx`.

#### `DFlashGraphRing`
Owns graph buffers and graph execs by `(bucket_bs, ring_slot)`.

#### `DFlashHostPacketRing`
Owns pinned CPU packets for late correction.

### 4.2 What does **not** exist

- no scheduler-facing draft worker
- no old-style cross-worker DFlash boundary
- no Eagle-derived carried tensor bundle
- no generic speculative future map carrying semantic DFlash tensors

---

## 5. Authoritative state model

### 5.1 Persistent per-request GPU state

The source of truth is device-resident and keyed by `req_pool_idx`.

```python
committed_len:   int32[num_req_slots]
reserved_len:    int32[num_req_slots]
next_verified_id:int32[num_req_slots]
generation:      int32[num_req_slots]
status_flags:    int32[num_req_slots]   # bitfield: active, eos_seen, finished, canceled, etc.
```

Optional additional tables if they prove useful:

```python
max_new_tokens_left: int32[num_req_slots]
stop_token_state:    int32[num_req_slots]
optimistic_epoch:    int32[num_req_slots]
```

### 5.2 Exact semantics

For a request `r`:

- logical token positions `[0, committed_len[r])` are already committed
- `next_verified_id[r]` is the token at logical position `committed_len[r]`
- `next_verified_id[r]` is **not yet counted** inside `committed_len[r]`
- `req_to_token[r, :reserved_len[r]]` is valid mapping space
- only `:committed_len[r]` is semantically committed
- reserved future slots may contain provisional draft KV

### 5.3 Shared mapping invariant

Target and draft share logical slot ids but have separate KV stores.

Shared:

- `req_to_token`
- token-slot allocator / page allocator

Separate:

- target KV tensors
- draft KV tensors

This means:

- target verify and draft block refer to the same logical future positions
- committed target hidden states overwrite the committed prefix of draft KV slots
- draft may later overwrite the remaining provisional tail it reuses
- no target<->draft slot remap tables are needed

---

## 6. Batch handles, graph rings, and host packets

### 6.1 `DFlashBatchHandle`

This is carried by the scheduler/runtime between overlap iterations.
It is intentionally tiny.

```python
@dataclass
class DFlashBatchHandle:
    bucket_bs: int
    ring_slot: int
    generation: int
    active_bs: int
    compute_ready_event: object
    host_ready_event: object
```

The handle does **not** carry semantic per-request tensors.
Those live in the request-state table and graph buffers.

### 6.2 `DFlashGraphBuffers`

Owned by `(bucket_bs, ring_slot)`.

```python
req_pool_indices:    int32[bucket_bs]
req_generation:      int32[bucket_bs]

# prep / draft
query_input_ids:     int32[bucket_bs, B]
query_positions:     int32[bucket_bs, B]
query_slot_ids:      int64[bucket_bs, B]
query_sample_idx:    int32[bucket_bs, B-1]      # optional if used by sample kernel

# outputs from draft
emit_ids:            int32[bucket_bs, B]        # lane 0 = next_verified_id, lanes 1.. = draft tokens

after_draft_hidden:  fp16_or_bf16[bucket_bs, B, H]  # optional, usually not persisted

# verify
target_top1:         int32[bucket_bs, B]
verify_hidden:       bf16[bucket_bs, B, H]

# accept/publish
commit_lens:         int32[bucket_bs]
bonus_ids:           int32[bucket_bs]
gpu_stop_flags:      int32[bucket_bs]

# optional
deficit_flags:       int32[bucket_bs]
```

No persistent `accepted_mask` buffer is part of the final design.

### 6.3 `DFlashHostPacket`

Pinned CPU packet for late correction.

```python
req_pool_idx_cpu:    int32[bucket_bs]
generation_cpu:      int32[bucket_bs]
commit_lens_cpu:     int32[bucket_bs]
emit_ids_cpu:        int32[bucket_bs, B]
stop_flags_cpu:      int32[bucket_bs]
active_bs_cpu:       int32[1]
```

Keep the packet fixed-shape unless profiling proves D2H copy is material.

### 6.4 Ring depth

Default:

- graph ring depth: 3
- host packet ring depth: configurable, default 6

These are separate knobs.

---

## 7. Streams and readiness semantics

Use at least three streams:

- `plan_stream`
- `compute_stream`
- `copy_stream`

Optional fourth stream after bring-up if needed:

- `materialize_stream`

### 7.1 `compute_ready`

`compute_ready` means:

- committed hidden -> draft KV materialization has completed
- request-state table publish has completed
- next GPU DFlash step may safely read authoritative state

### 7.2 `host_ready`

`host_ready` means:

- pinned host correction packet is safe for CPU processing

### 7.3 Overlap rule

The next decode step depends on `compute_ready`, not `host_ready`.

CPU correction is late and non-blocking, keyed by `req_pool_idx + generation`.

---

## 8. Reservation and headroom policy

### 8.1 Hard correctness invariant

Let `B = block_size`.

For every active request during steady-state overlap:

```python
reserved_len - committed_len >= 2 * B
```

This guarantees one-step overlap safety.

### 8.2 Growth policy

Do **not** reserve exactly `2 * B` each time.
That is the correctness floor, not the growth policy.

Use page-aware larger growth:

```python
if reserved_len - committed_len < soft_watermark:
    grow_by = page_aligned_quantum(max(4 * B, policy_quantum))
```

Default starting point:

- `soft_watermark = 3 * B`
- `grow_by = max(4 * B, page_quantum)`

Also benchmark:

- `soft_watermark = 4 * B`
- `grow_by = 8 * B`

### 8.3 Reservation manager contract

`DFlashReservationManager` must:

1. use page-aware extension semantics
2. fill any newly valid `req_to_token` entries
3. preserve dummy DFlash lane reservation invariants
4. support a lightweight deficit query for batch prep

It must not use a raw allocator path that bypasses paged-cache semantics.

---

## 9. Exact decode-step semantics

## 9.1 Prefill

1. Run target prefill with hidden capture.
2. Materialize prompt hidden states into draft KV immediately.
3. Initialize request-state table:

```python
committed_len   = prompt_len
next_verified_id= first_verified_token
reserved_len    = prompt_len + initial_dflash_tail
status_flags    = active
```

4. Publish first `compute_ready` event.

Prompt hidden states are never carried in the overlap handle.

### 9.2 Steady-state decode timeline

#### Stage A: CPU optimistic scheduling

Scheduler chooses candidate requests and bucket size.
It may optimistically keep requests in flight before host correction arrives.
But it does **not** compute token truth.

CPU owns only:

- request lifecycle
- detokenization and stop-string logic
- batch roster proposal
- late correction application

#### Stage B: `plan_stream`

1. Wait on the prior `compute_ready` for the consumed handle.
2. Ensure reservation headroom through `DFlashReservationManager`.
3. Fill ring-local `req_pool_indices` / `generation` buffers.
4. Optionally precompute any non-graphed backend metadata.

#### Stage C: `compute_stream` fixed-shape execution

Run the fixed-shape DFlash step for `(bucket_bs, ring_slot)`.
This execution must read authoritative GPU state from `DFlashRequestStateTable`.

Core logical order:

1. block prep
2. draft embedding path
3. draft forward
4. draft lm-head top1
5. emit candidate write
6. target verify
7. target top1
8. accept / bonus / stop kernel

#### Stage D: `copy_stream`

After accept/bonus is available, copy the host correction packet asynchronously.

#### Stage E: committed-hidden materialization + publish

1. materialize committed target hidden into draft KV
2. publish request-state table updates:

```python
committed_len   += commit_lens
next_verified_id = bonus_ids
status_flags    |= gpu_stop_flags
```

3. record next `compute_ready`

---

## 10. Aggressive overlap contract (zero-bubble with late correction)

This rewrite explicitly targets aggressive overlap similar in spirit to modern async speculative scheduling.

### 10.1 What the CPU may assume optimistically

The CPU scheduler may optimistically assume:

- requests remain batch-eligible until a correction packet says otherwise
- a request still needs the next DFlash step
- a request stays in the same bucket for one more step

### 10.2 What the CPU may **not** assume

The CPU may not be the source of truth for:

- last committed token id
- exact committed length used by GPU prep
- bonus token id used for next lane-0 input
- exact number of rejected tokens from the prior step

### 10.3 GPU source-of-truth rule

All of these live on GPU and are read by the next prep kernel from the request-state table:

- `committed_len`
- `next_verified_id`
- `status_flags`
- any GPU-known stop state

### 10.4 Generation rule

Every host packet and every carried batch handle is keyed by `req_pool_idx + generation`.
Packets for stale generations are dropped.

### 10.5 Lane filtering rule

The scheduler may over-include rows within a chosen bucket, but the GPU prep path must support marking rows inactive through:

- canceled generation
- GPU stop bits
- dummy-lane fallback

This makes optimistic overlap robust without requiring exact CPU correction before the next GPU step.

---

## 11. Correctness contracts that must be written into code comments

### 11.1 `emit_ids`, `target_top1`, `commit_len`, `bonus`

For each request row:

```python
emit_ids[0]      = next_verified_id
emit_ids[1:B]    = draft_top1[0:B-1]
target_top1[j]   = target-predicted next token after emit_ids[j]
```

Then:

```python
accept_len = longest prefix where target_top1[0:B-1] == emit_ids[1:B]
commit_len = accept_len + 1
bonus_id   = target_top1[accept_len]
```

### 11.2 Stop semantics

- EOS or stop condition inside the committed prefix stops the request.
- EOS or stop condition in `bonus_id` does **not** stop it yet, because `bonus_id` is not committed this step.

### 11.3 Dummy lane semantics

Inactive lanes must be safe no-ops.
They must not mutate real request state or write into real request slots.

---

## 12. Required fused jobs

These are the fused jobs that should exist in the initial implementation.
They are intentionally small and high leverage.

### 12.1 `dflash_prepare_block_kernel`

Purpose:

- read authoritative GPU request state
- write fixed-size query input ids for the draft block
- write query positions
- write query slot ids / cache locs
- optionally write query token indices for draft sampling
- optionally null out dummy lanes

Inputs:

- `req_pool_indices[bucket_bs]`
- `generation[bucket_bs]`
- `committed_len[req_pool_idx]`
- `next_verified_id[req_pool_idx]`
- `status_flags[req_pool_idx]`
- `req_to_token`

Outputs:

- `query_input_ids[bucket_bs, B]`
- `query_positions[bucket_bs, B]`
- `query_slot_ids[bucket_bs, B]`
- optional sample indices

Must support:

- bucket padding / dummy lanes
- GPU stop-bit filtering
- page-aware shared mapping

### 12.2 `dflash_accept_bonus_kernel`

Purpose:

- compare draft candidates vs target top1
- compute `commit_lens`
- compute `bonus_ids`
- compute GPU stop bits for committed prefix
- optionally compute `num_rejected` or accepted-count derivatives used later

Inputs:

- `emit_ids[bucket_bs, B]`
- `target_top1[bucket_bs, B]`
- optional stop-token tables

Outputs:

- `commit_lens[bucket_bs]`
- `bonus_ids[bucket_bs]`
- `gpu_stop_flags[bucket_bs]`

### 12.3 `dflash_publish_state_kernel`

Purpose:

- update `committed_len`
- update `next_verified_id`
- set or clear GPU-known stop flags
- optionally mark rows for future compaction or retirement

Inputs:

- `req_pool_indices`
- `commit_lens`
- `bonus_ids`
- `gpu_stop_flags`

Outputs:

- updated request-state table

### 12.4 `draft_top1_fastpath`

Purpose:

- compute greedy top1 from draft hidden states without materializing more than necessary

Must support:

- TP=1 fast path
- TP>1 path with correct shard reduction

### 12.5 `target_top1_fastpath`

Same idea as draft top1, but for target verify logits selection.
This is not the main optimization target, but it should exist as the default fast path.

---

## 13. Experiment matrix across the DFlash lifecycle

The rest of the kernel shape is deliberately left open and must be selected by measurement.

Each experiment family has:

- a control
- variants
- required metrics
- a keep / reject rule

### EXP-0: Bring-up correctness experiments

Goal: lock the prompt + first-block contract before any throughput work.

#### Variants

- C0: prefill + first decode step only
- C1: prefill + two decode steps
- C2: overlap off, same kernels
- C3: overlap on, same kernels

#### What to validate

- prompt hidden materialization correctness
- `committed_len` / `next_verified_id` contract
- first draft block ids, positions, slot ids
- first target verify candidates
- first `commit_len` and `bonus_id`
- second block correctness after one publish

#### Required assertions

- first draft block tokens are sensible
- `emit_ids` lane 0 equals authoritative `next_verified_id`
- `bonus_id` becomes the next step’s lane 0
- no request reads beyond committed tail because of optimistic CPU seq-lens drift

Keep rule: no further optimization work until all C0-C3 pass on `trtllm_mha + fa4`.

---

### EXP-1: Reservation and mapping experiments

Goal: choose the lowest-overhead correct reservation strategy.

#### Control

R0: page-aware extend using existing paged-cache semantics, soft watermark `3*B`, grow quantum `4*B`.

#### Variants

- R1: soft watermark `4*B`, grow quantum `8*B`
- R2: soft watermark `3*B`, grow quantum page-size rounded exact `2*B`
- R3: per-step exact top-up to `2*B` (control only; likely reject)
- R4: fused deficit-detect + new-range fill kernel vs separate kernels

#### Metrics

- allocator / mapping time in `nsys`
- number of top-ups per 1k tokens
- tok/s c1 and c32
- any correctness instability under batch churn

Keep rule:

- no variant that increases top-up frequency significantly unless it wins e2e
- per-step exact `2*B` top-up should be rejected unless it surprisingly wins

---

### EXP-2: Draft block prep experiments

Goal: minimize `draft_setup` and make the block prep graph-friendly.

#### Control

P0: dense GPU prep kernel writing `input_ids`, `positions`, and `slot_ids` from authoritative GPU state.

#### Variants

- P1: write `input_ids` + `positions` + `slot_ids` + `emit_ids[:,0]` in one kernel
- P2: same as P1 plus direct write of sample indices for draft top1 gather
- P3: direct write of draft input embeddings instead of input ids
- P4: hybrid path: input ids only, embeddings gathered by the draft model
- P5: CPU-prepared control path for correctness only

#### Metrics

- stage time (`draft_setup`)
- graph replay compatibility
- total launches added
- tok/s c1 and c32

#### Notes

`P3` exists because the query pattern is always `[bonus, mask, mask, ...]`.
A direct-embedding kernel can gather the bonus embedding and broadcast the mask embedding.
This must be measured rather than assumed.

Keep rule:

- prefer the smallest stable path that wins e2e
- `input_ids` path remains the fallback because it is simpler and graph-friendly

---

### EXP-3: Draft embedding path experiments

Goal: reduce work around the draft model entry.

#### Control

E0: draft runtime owns the embedding path and reads `input_ids`.

#### Variants

- E1: direct-embedding block prep (`P3`) plus draft forward on `input_embeds`
- E2: cache the mask embedding tensor and only gather bonus-token embedding each step
- E3: fuse bonus gather + mask broadcast + optional hidden combine pre-step

#### Metrics

- stage time (`draft_setup`)
- any graph restrictions introduced
- e2e tok/s

Keep rule:

- only keep direct-embedding variants if they beat `input_ids` consistently on both c1 and c32

---

### EXP-4: Draft model forward experiments

Goal: improve draft runtime without changing model semantics.

#### Control

D0: graph-captured fixed-shape draft forward using the normal draft runtime.

#### Variants

- D1: bucket-specific CUDA graphs only
- D2: bucket + ring-slot-specific graphs
- D3: draft backend metadata reuse vs rebuild per ring slot
- D4: alternate FA4 wrapper/config choices if correctness-equivalent

#### Metrics

- `draft_forward` stage time
- graph launch overhead
- memory footprint
- correctness on primary backend pair

#### Important rule

Do **not** spend the next rewrite cycle on a full draft-model megakernel first.
The draft model itself is not yet the dominant gap.

Only after materializer + overlap are competitive may we consider a research branch for a larger draft-model fusion.

---

### EXP-5: Draft sampling / lm-head top1 experiments

Goal: reduce `draft_sample`.

#### Control

S0: draft-local greedy top1 fast path.

#### Variants

- S1: TP=1 specialized direct linear+argmax path
- S2: TP>1 custom top1 kernel with local `(max, id)` reduction followed by cross-rank reduction
- S3: full logits materialization control path (correctness only)
- S4: chunked vocab scan vs fused top1 kernel

#### Metrics

- `draft_sample` stage time
- NCCL overhead under TP>1
- tok/s c1 and c32

Keep rule:

- any variant must beat the control on the actual TP configuration in use
- full logits path is correctness-only and should not remain on the fast path

---

### EXP-6: Target verify integration experiments

Goal: minimize non-forward overhead while not over-optimizing the already-competitive target path.

#### Control

V0: target verify with top1-only output path and captured hidden states.

#### Variants

- V1: different graph boundaries around verify prep vs verify forward
- V2: direct outputs into DFlash-owned buffers vs intermediate staging
- V3: verify metadata reuse by `(bucket_bs, ring_slot)`

#### Metrics

- `verify_prep`
- `target_verify`
- graph replay stability
- hidden capture cost

Keep rule:

- do not spend further cycles here unless `nsys` says this is now a top blocker

---

### EXP-7: Commit materialization experiments (the main open search)

Goal: find the best committed-hidden -> draft-KV materialization shape.

This family is deliberately left open. It is the main kernel search space.

#### Common inputs

- `verify_hidden[bucket_bs, B, H]`
- `query_positions[bucket_bs, B]`
- `query_slot_ids[bucket_bs, B]`
- `commit_lens[bucket_bs]`

#### Common rules

- all candidates must preserve exact semantics
- all candidates must support prompt materialization separately
- all candidates must be benchmarked both as isolated stage time and e2e

#### Control candidates

- M0: dense per-layer compute + prefix-valid GPU writes
- M1: dense per-layer compute + masked full write (control only)
- M2: valid-only GPU compaction + valid-only write
- M3: valid-only CPU compaction + valid-only write (rejected control)

#### Fused candidates to test

- M4: grouped-layer GEMM with group size `G=2`, then per-layer prefix-valid writes
- M5: grouped-layer GEMM with group size `G=4`, then per-layer prefix-valid writes
- M6: grouped-layer GEMM with group size `G=8`, then per-layer prefix-valid writes
- M7: full-stack GEMM over all layers, then per-layer prefix-valid writes
- M8: full-stack GEMM + grouped prefix-valid write kernel
- M9: grouped-layer GEMM + grouped prefix-valid write kernel in Triton
- M10: grouped-layer GEMM + grouped prefix-valid write kernel in `jit-kernel` / cute DSL

#### Why grouped variants exist

Old experiments showed that **full stacked direct scatter** was slower, but that does not rule out an intermediate grouped optimum.
Search `G in {1,2,4,8,all}` explicitly.

#### Additional write-style variants

For each compute variant above, benchmark:

- W0: existing prefix-valid write kernel
- W1: new Triton prefix-valid tiled write kernel specialized for small `B`
- W2: `jit-kernel` / cute prefix-valid write kernel specialized for the KV layout

#### Metrics

- `commit_append`
- split into `commit_project`, `commit_norm_rope`, `commit_write` where possible
- workspace bytes
- register pressure / occupancy
- tok/s c1 and c32

#### Keep rule

A new materializer variant is only accepted if it wins both:

1. stage time by a meaningful margin
2. e2e tok/s on the primary benchmark pair

Do not keep a microbench-only win.

---

### EXP-8: Prompt materialization experiments

Goal: treat prompt bursts separately from steady-state commit append.

Prompt materialization is different because token count can be large.

#### Control

PM0: chunked prompt materialization using the same underlying materializer family as steady-state, with bounded chunk sizes.

#### Variants

- PM1: per-layer sequential materialization, chunked by token rows
- PM2: grouped-layer materialization, chunked by token rows
- PM3: full-stack materialization, chunked by token rows
- PM4: backend-native per-layer cache update if available and faster
- PM5: chunk sizes `{128, 256, 512, 1024}` rows

#### Metrics

- prompt materialization wall time
- peak memory
- launch-time stability / OOM risk
- effect on prompt->first-token latency

Keep rule:

- prompt path may use a different winner than steady-state commit path
- never use an unchunked path that risks Triton launch-time OOM

---

### EXP-9: Publish and stop-state experiments

Goal: keep publish tiny and GPU-authoritative.

#### Control

U0: dedicated publish kernel updating request state and stop flags.

#### Variants

- U1: fuse publish into accept/bonus kernel
- U2: fuse publish into the materializer tail kernel
- U3: separate stop-kernel vs fused stop detection in accept kernel

#### Metrics

- launch count
- gap between end of target verify and next compute-ready
- correctness under EOS / max_len / canceled-request churn

Keep rule:

- prefer the smallest launch count that preserves debuggability and correctness

---

### EXP-10: Graph partition experiments

Goal: choose the best graph boundary.

#### Candidates

- G0: core graph only; eager materializer/publish tail
- G1: core graph + tail graph
- G2: full-step graph including materializer + publish
- G3: graph block prep inside core graph vs outside on plan stream

#### Metrics

- replay overhead
- bubble time between steps
- graph capture complexity
- e2e tok/s

#### Keep rule

Do not force a one-graph design if it reduces overlap.
Start from the graph split that minimizes bubble time on `nsys`.

---

### EXP-11: Overlap and late-correction experiments

Goal: maximize overlap while preserving GPU truth.

#### Control

O0: conservative overlap where CPU waits for host correction before rebuilding next batch.

#### Variants

- O1: optimistic CPU carry; GPU request table remains source of truth
- O2: optimistic CPU carry + GPU dummy-lane filtering for stale/canceled rows
- O3: optimistic carry + larger host packet ring depth
- O4: optimistic carry + different graph ring depths `{2,3,4}`

#### Metrics

- bubble time between `compute_ready` and next core-graph replay
- D2H overlap ratio
- scheduler CPU time
- stale-packet drop rate
- tok/s c1 and c32

Keep rule:

- the winner must not reintroduce optimistic-CPU token-truth bugs
- GPU lane-0 token truth must always come from the GPU request table

---

### EXP-12: Host packet minimization experiments

Goal: reduce D2H cost only if it matters.

#### Control

H0: fixed packet carrying `req_pool_idx`, `generation`, `commit_lens`, and `emit_ids`.

#### Variants

- H1: packed fixed packet carrying only committed prefix tokens plus bonus
- H2: split packets: minimal immediate correction packet + later optional token packet
- H3: GPU-packed variable-length packet (research only)

#### Metrics

- D2H bytes
- CPU decode simplicity
- overlap quality
- e2e tok/s

Keep rule:

- keep the simple fixed packet unless D2H copy is shown to matter materially

---

## 14. Candidate implementation shapes for the materializer

The materializer should be coded behind a pluggable interface so the experiment matrix above is easy to run.

```python
class DraftKVMaterializer(nn.Module):
    def materialize_prompt(
        self,
        hidden: torch.Tensor,      # [N, H]
        positions: torch.Tensor,   # [N]
        slot_ids: torch.Tensor,    # [N]
    ) -> None:
        ...

    def materialize_commit(
        self,
        verify_hidden: torch.Tensor,    # [bs, B, H]
        positions: torch.Tensor,        # [bs, B]
        slot_ids: torch.Tensor,         # [bs, B]
        commit_lens: torch.Tensor,      # [bs]
    ) -> None:
        ...
```

### 14.1 Required implementation families

Implement the following families behind one switchable interface:

- `per_layer_dense_prefix_write`
- `grouped_dense_prefix_write`
- `stacked_dense_prefix_write`
- `valid_only_compact_write` (control only)

### 14.2 Group size search

For grouped variants, benchmark:

```python
G in {1, 2, 4, 8, all}
```

### 14.3 DSL guidance

Preferred implementation order:

1. Triton first for bring-up speed and easy iteration
2. if the grouped write kernel plateaus, add a `jit-kernel` / cute implementation for the prefix-valid write stage
3. do not jump to a fully custom C++ path before the grouped search is exhausted

The most likely place for `jit-kernel` / cute to pay off is **prefix-valid KV writes into paged KV layout**, not the full draft transformer.

---

## 15. Draft-model optimization guidance

The draft model is not the first thing to rewrite into a megakernel, but there are still safe optimizations to include.

### 15.1 Must-have

- draft-local `embed_tokens`
- draft-local `lm_head`
- graph-captured fixed-shape draft forward
- greedy top1 fast path

### 15.2 Worth benchmarking

- input-id path vs direct-embedding path for `[bonus, mask, mask, ...]`
- TP-aware top1 kernel
- ring-local backend metadata reuse
- bucketed graphs tuned for the actual DFlash block sizes in use

### 15.3 Not recommended for attempt two

- a full draft-model megakernel
- custom fused transformer layer rewrite before materializer and overlap are solved

### 15.4 Research-only branch after parity

Only after parity or better on the primary benchmark pair, consider a research branch for:

- fused embedding + first layernorm entry path
- tiny-query non-causal attention specialization for fixed `B`
- persistent-kernel style draft forward for very small `B`

This is explicitly not on the critical path for the second rewrite attempt.

---

## 16. GPU-known stop logic

To support late host correction safely, these stop conditions should be visible to GPU state publish:

- EOS token id
- max_new_tokens / max length
- any GPU-checkable stop-token set

CPU-only stop checks such as stop strings remain in host correction.

A request that is GPU-stopped may still appear in an optimistic CPU batch proposal, but GPU prep must null it out safely.

---

## 17. Backend metadata ownership rules

Because backend wrapper state is correctness-sensitive, all backend plan / wrapper state must be owned by `(bucket_bs, ring_slot)` where necessary.

Rules:

1. draft backend plan objects are DFlash-local
2. verify backend plan objects are DFlash-local
3. any custom-mask buffers are wired only when a real custom mask is required
4. wrapper reuse must never cross ring slots if it changes semantics

---

## 18. Instrumentation plan

### 18.1 Stage timers

Record at minimum:

- `draft_setup`
- `draft_forward`
- `draft_sample`
- `verify_prep`
- `target_verify`
- `accept_bonus`
- `commit_append`
- `commit_project`
- `commit_norm_rope`
- `commit_write`
- `publish_state`
- `d2h_copy`
- `bubble_time`

### 18.2 `nsys` checkpoints

Mandatory `nsys` runs:

1. single-request greedy steady-state decode
2. concurrency-32 steady-state decode
3. prompt-heavy short-output run
4. one run with overlap disabled as a control

### 18.3 Microbench harnesses

Add dedicated harnesses for:

- block prep
- lm-head top1
- prompt materializer
- commit materializer
- publish kernel
- host packet copy and overlap

These harnesses should be selectable independently from the full server benchmark.

---

## 19. File layout proposal

```text
python/sglang/srt/speculative/dflash/
  coordinator.py
  request_state.py
  reservation.py
  batch_handle.py
  graph_ring.py
  host_packet.py
  target_runtime.py
  draft_runtime.py
  materializer.py
  interfaces.py
  correctness_debug.py

python/sglang/srt/speculative/dflash/kernels/
  prepare_block.py
  accept_bonus.py
  publish_state.py
  lm_head_top1.py
  kv_prefix_write.py
  kv_materialize_grouped.py
  stop_state.py

python/sglang/srt/speculative/dflash/experiments/
  exp_prepare_block.py
  exp_materializer.py
  exp_prompt_materializer.py
  exp_lm_head_top1.py
  exp_overlap.py
  exp_graph_partition.py
```

---

## 20. Implementation order

### Phase 1: correctness-first skeleton

1. implement `DFlashRequestStateTable`
2. implement `DFlashReservationManager`
3. implement `DFlashCoordinator` + batch handles + rings
4. implement target prefill + prompt materialization path
5. implement simple block prep kernel
6. implement draft forward + draft top1
7. implement target verify + accept/bonus + publish
8. validate first-block correctness on `trtllm_mha + fa4`

### Phase 2: overlap bring-up

1. add separate `compute_ready` and `host_ready`
2. add fixed host packets keyed by `req_pool_idx + generation`
3. enable optimistic CPU carry with GPU source-of-truth prep
4. validate stale-packet and canceled-request handling

### Phase 3: graph bring-up

1. add core graph
2. benchmark core graph vs eager
3. add tail graph candidate
4. benchmark core-only vs core+tail vs full-step graph

### Phase 4: experiment sweep

1. run the full experiment matrix from sections 13-15
2. lock winners based on e2e + `nsys`
3. remove losing code paths from the fast path
4. keep correctness control paths under debug flags only

---

## 21. Decision rules

1. Do not lock a kernel shape because it feels elegant.
2. Do not assume prefix-filtering, compaction, or stacked writes are optimal without the experiment family proving it.
3. Do not optimize target verify first unless new profiling says it became the main blocker.
4. Do not reintroduce Python-side committed-row compaction into the fast path unless it wins clearly.
5. Do not rely on optimistic CPU sequence lengths or CPU backup-token lookup for next-step truth.
6. Prefer the simplest graph split that minimizes bubble time.
7. Prefer grouped kernel searches before attempting a full-model megakernel.

---

## 22. Default starting choices for attempt two

These are the starting choices for the clean branch before experiments begin.
They are defaults, not truths.

- block prep: dense Triton kernel writing `input_ids`, `positions`, `slot_ids`
- draft entry: `input_ids` path, draft-local embedding
- draft forward: graph-captured fixed-shape
- draft sample: draft-local greedy top1 fast path
- target verify: top1-first output path with hidden capture
- publish: separate publish kernel
- overlap: optimistic CPU carry, GPU source of truth, late host correction
- materializer starting control: dense per-layer compute + prefix-valid GPU writes
- prompt materializer starting control: chunked per-layer or grouped path, not unbounded stacked path
- graph strategy starting control: core graph first, then benchmark tail graph

The second rewrite attempt should start from these defaults and then let the experiment matrix decide what survives.
