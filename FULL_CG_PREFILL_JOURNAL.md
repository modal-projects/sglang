# Full-graph prefill for Inkling (`--cuda-graph-backend-prefill full`) — Engineering Journal

- **Branch:** `cheng/full-cg-prefill-inkling` (rebased on `origin/dev`).
- **Goal:** Make the upstream **full** prefill CUDA-graph backend correct + usable for the TML/Inkling model, and compare input throughput vs the breakable (BCG) path.
- **Outcome:** ✅ **FIXED.** Full-graph prefill is now correct for batched inference. gsm8k **P64 = 0.940 / 0% invalid** (was 0.245 / 72.5%), P1 = 0.95. Input throughput is competitive with breakable (roughly even; +14–17% at launch-bound shapes). Root cause was a single pointer-keyed cache staleness in the CUTE FA4 sheared-bias kernel under CUDA-graph capture.

---

## TL;DR

The bug was **not** numerical fragility (an earlier journal draft wrongly concluded a "kernel-rounding accuracy ceiling ~0.48" — that number was entirely this bug **cascading through the radix-shared prefix**). The real bug:

> The CUTE FA4 **sheared-bias** path (used by Inkling because it always passes `rel_bias` = `rel_logits`) caches its per-tile→sequence block schedule (`blocks_to_batch_idx` / `cu_total_m_blocks_bias`) in `rel_bias_prep_cache`, **keyed on `cu_seqlens_q.data_ptr()`** (`jit_kernel/flash_attn/cute/interface.py:750`).

Under full-CG prefill, `cu_seqlens_q` is a **pointer-stable** metadata buffer whose **values** change every replay. Warmup+capture populate the cache from the dummy single-sequence capture layout `[num_tokens, 0, …, 0]`; the unchanging data_ptr then produces a **cache hit at every replay**, so the block-schedule kernel is **skipped** (never recorded in the graph) and the schedule stays frozen at the single-seq layout.

Result: in any bs>1 batched prefill, **sequence 0 (matching the frozen layout) is correct and sequences 1+ get the wrong tile→seq bias mapping → garbage**. The garbage KV/state then poisons the **radix-shared mamba conv-state**, cascading the whole batch (why gsm8k P64 collapsed to ~0.24 while bs=1 was ~0.90).

**Fix** (`flashattention_backend.py::forward_extend`): when the metadata is the full-CG prefill metadata, pass `rel_bias_prep_cache=None`, so the block-schedule kernel is recorded in the captured graph and refreshes from the live `cu_seqlens` at every replay. It's the **only** `data_ptr`-keyed runtime-data cache in the CUTE path (verified; the other `compile_cache`s key on dtypes/shapes/flags). Decode is layout-invariant (1 query/seq) and never matches the branch; eager/BCG use fresh per-call `cu_seqlens`, so all are unaffected.

---

## Always-on changes (the fix + enablers)

1. **`flashattention_backend.py::forward_extend`** — the fix: `rel_bias_prep_cache=None` for the full-CG prefill metadata (identity-checked via `metadata is self.full_cg_prefill_metadata`; `self.full_cg_prefill_metadata` is initialized to `None` in `__init__`).
2. **`flashattention_backend.py::_init_full_cg_prefill_metadata`** (from the original branch, still needed):
   - **Page-size generalization** — build the block table in *pages* via a strided `req_to_token` gather instead of raising on `page_size != 1` (identity at page_size 1).
   - **SWA support** — populate `swa_page_table` + a pointer-stable `swa_out_cache_loc`; zero its tail beyond the live tokens (mirrors decode's `swa_out_cache_loc_buf[n:].zero_()`) so padded KV writes land on the dummy slot, not stale live SWA slots.

---

## Investigation path (how it was found)

The earlier session was stuck behind a **Heisenberg wall**: any probe/break that could observe the broken monolithic state did a host sync, which itself masked the corruption. This session broke through:

1. **Non-Heisenberg device probe** (temporary): wrote per-layer stats to a stable device buffer with plain device ops inside the captured loop (no host sync), dumped from the eager outer forward. It showed a **compounding blow-up to NaN** — but that was later found to be dominated by *padding-row* garbage, a measurement artifact.
2. **bs isolation** was the key: **bs=1 prefill is correct (gsm8k 0.90); bs>1 corrupts**, robustly **"seq 0 ok, seqs 1+ garbage"**. That pattern maps exactly to the single-sequence capture dummy `[num_tokens, 0, …]`.
3. **Ruled out** (each toggled, none was it): attention alt-stream, symm-mem all-reduce, AR-sconv fusion, overlap scheduler, padding amount (fine vs coarse buckets), `num_splits` (a red herring — one run passed, not reproducible), multi-seq capture layout (made it *worse* — bakes a different fixed layout). Verified that `cu_seqlens`, `page_table`, `positions`, sconv `si`/`has_initial_state` all **refresh** correctly at replay.
4. **The tell:** corruption tracked the **capture layout exactly** (single-seq capture → matches bs=1 & seq-0-of-bs>1; multi-seq capture → matches nothing → all corrupt). That meant a **baked, layout-specific schedule that doesn't refresh** — pointing into the CUTE varlen kernel.
5. Reading `jit_kernel/flash_attn/cute/interface.py` found the `rel_bias_prep_cache` keyed on `cu_seqlens_q.data_ptr()` — stable pointer + changing values under capture = stale. Passing `None` fixed every case.

## Verification

- Repros (fresh, flush between): `bs_narrow` (conc 2/3), `bs_wide` (conc 4/8/12), `prefix_extend` (conc 6) — **all `ok`** after the fix (all corrupt before).
- gsm8k: **P64 = 0.940 / 0% invalid** (was 0.245 / 72.5%), **P1 = 0.95**.
- Throughput (quick probe, noisy): full-CG vs breakable roughly even; +14–17% at launch-bound shapes (L4096 bsz1/bsz16), ≈ at L8192.

## Notes / follow-ups

- The corruption **poisons persistent state** (radix-shared mamba conv-state); once one bad prefill lands, later requests cascade until `/flush_cache`. This is why early accuracy measurements were so confusing.
- Debug scaffolding used during the hunt (device probe, multi-seq-capture / zero-pad / num_splits toggles) was **removed**; only the fix + enablers remain.
- Proper `bench_one_batch_server --profile --profile-by-stage` sweeps (input_len 4096/8192/16384 × bsz 1/4/16/64) are the next step for a clean throughput/latency number and traces.
