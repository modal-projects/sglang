# DCP + tokenspeed_mla + DFlash — design notes

Branch `jamesliu/dcp-dflash`: decode context parallelism (DCP) support for the
`tokenspeed_mla` / `trtllm_mla` attention backends, plus DFlash speculative
decoding under DCP. Base: upstream main + DCP refactor PR #29365
(`layers/cp/dcp/{comm,kernels,layout,metadata,planner}.py`) + fork ports.

Companion: tokenspeed_mla wheel from branch `jamesliu/decode-lse`
(`tokenspeed_mla_decode(..., return_lse=True)` → `(output, lse)`), required at
runtime — the backend fails fast at init if the installed wheel lacks
`return_lse` when DCP is enabled.

## KV layout recap (PR #14194, unchanged)

- KV is sharded round-robin by position: `pos % dcp_size == dcp_rank` owns the
  token; physical slot = `logical_loc // dcp_size`.
- The allocator is widened: capacity `max_total_num_tokens * dcp`, page size
  `page_size * dcp`. `req_to_token` holds LOGICAL locations.
- MLA writes are rank-filtered inside `set_mla_kv_buffer` (Triton kernel does
  the owner filter + logical→physical divide). The TMA bulk-store fast path is
  bypassed under DCP; the DSA fp8-with-scales write path now asserts it is not
  used under DCP (it would bypass the filter).

## Invariants

1. **Rank-invariant block tables.** A logical page of `page_size * dcp` tokens
   maps to exactly one contiguous physical page of `page_size` tokens on every
   rank, at the SAME page index (`logical_loc // (page_size*dcp) ==
   physical_slot // page_size`), and per-rank KV is dense and position-ordered.
   So trtllm/tokenspeed block tables are built ONCE with
   `create_flashmla_kv_indices_triton` using `PAGED_SIZE = page_size * dcp`
   over the logical `req_to_token`, and are identical across ranks. Only the
   per-request KV LENGTHS passed to the kernel are rank-local
   (`get_dcp_lens`, the `start=None` owner rule). Page-count identity:
   `ceil(ceil(L/dcp)/ps) == ceil(L/(ps*dcp))`, so global-length cdiv with the
   widened page size covers every rank's local page needs exactly.
2. **`_calc_padded_blocks` reconciliation.** The TRT-LLM
   `block_num % (128/block_size)` constraint keeps the PHYSICAL `page_size`
   (the kernel sees physical pages); the cdiv over `max_seq_len` and the
   Triton index-build constraint use the LOGICAL page size.
3. **Base-2 LSE everywhere.** The tokenspeed decode kernel returns
   `lse = log2(sum_k exp(scale * q.k))` — base-2, softmax scale (including the
   folded `k_scale`) already applied, shape `[B, q_len, H]` fp32, `+inf`
   sentinel for empty rows. `cp_lse_ag_out_rs_mla` /
   `_correct_attn_cp_out_kernel` work in base-2 (`exp2`/`log2`) and treat
   `+inf`/NaN (and `-inf`, which contributes `exp2(-inf)=0`) as empty — the
   kernel LSE feeds the merge with NO base conversion.
4. **fp8 scale consistency.** `softmax_scale = layer.scaling * k_scale` and
   `output_scale = k_scale` are identical on every rank, so per-rank LSEs are
   mutually consistent for the cross-rank merge. The torch block phase of the
   verify path applies the same two scales.
5. **Residue-class block sharding at verify.** The draft block's fresh K/V is
   computed replicated on every rank (before the rank-filtered cache write);
   rank `r` attends block position `j` iff `(seq_start + j) % dcp == r`, under
   causality `q_i >= j`. This partitions block tokens exactly once across
   ranks and matches the persistent shard layout for future decode steps.

## What was changed

### 1. Decode (q_len=1) — `trtllm_mla_backend.py`, `tokenspeed_mla_backend.py`
- `TRTLLMMLADecodeMetadata` gains `dcp_local_seq_lens` /
  `dcp_max_local_seq_len` (decode) and `dcp_local_prefix_lens` /
  `dcp_max_local_prefix_len` (verify). Eager path computes them in
  `init_forward_metadata`; CUDA graphs allocate persistent buffers in
  `_init_cuda_graph_metadata` and refill them on every capture/replay in
  `_apply_cuda_graph_metadata` (invoked by
  `decode_cuda_graph_runner` replay-prep via `init_forward_metadata_out_graph`
  — same hook DFlash's `prepare_for_verify` → `load_batch` uses, so the
  overlap scheduler path participates automatically).
- `forward_decode` under DCP: q arrives all-gathered along heads
  (`attn_mqa_for_dcp_decode`, `num_local_heads * dcp` heads); the kernel runs
  with local lens + `return_lse=True` and returns
  `(out.view(-1, H*512), lse.view(-1, H))` for `cp_lse_ag_out_rs_mla`
  (out `[B,H,D]`-viewable, lse `[B,H]`).
- `_run_decode_kernel` grew `return_lse` / `causal_mask` params. Base
  (trtllm-gen) raises `NotImplementedError` for either — flashinfer's
  `trtllm_batch_decode_with_kv_cache_mla` exposes neither an LSE output nor a
  non-causal mode, so **DCP is tokenspeed-only**; the base class carries the
  shared metadata plumbing.
- tokenspeed workspace sized for `num_q_heads * dcp` (gathered heads),
  `MAX_Q_LEN=8` covers DFlash verify. q_len 5–8 with LSE is FP8-kernel-only —
  this backend is FP8-KV-only, so that's the only path.

### 2. DFlash target verify (q_len = num_draft_tokens = 8)
- `server_args._handle_dcp_validation`: CUDA + spec is allowed only for
  `DFLASH` + explicit `--attention-backend tokenspeed_mla`.
- `forward_mla.py`: the three DCP decode branches (q gather in
  `forward_absorb_prepare`, `attn_mqa_for_dcp_decode` call and
  `cp_lse_ag_out_rs_mla` merge in `forward_absorb_core`) now also cover
  `is_target_verify()`. Verify is checked BEFORE `is_extend()` (it is a
  sub-mode of extend — previously it would have fallen into the extend-DCP
  branch and crashed on the missing `attn_dcp_metadata`). Gather/merge are
  per-token and handle `bs*8` tokens unchanged.
- `TRTLLMMLABackend._forward_target_verify_dcp` — the two-phase attention:
  - (a) prefix: decode kernel, all 8 q tokens × gathered heads over the
    rank-LOCAL committed prefix (lens from PRE-draft seq_lens),
    `causal_mask=False`, `return_lse=True`. The block tokens' KV has already
    been written to the cache at this point, but local prefix lens exclude
    them (logical→physical is monotonic, so local block copies sit at
    physical positions ≥ local_prefix_len).
  - (b) block: `[bs, 8, H_gathered] × [≤2 owned kv]` masked attention in
    torch fp32 over the fresh `k`/`k_rope` produced by
    `mla_quantize_and_rope_for_fp8` (exactly the replicated block K, no
    re-projection or threading needed). Scores computed directly in the
    base-2 domain (`* scale * log2(e)`); value = the latent (`kv_a`);
    all-masked rows → `lse=-inf`, `out=0`.
  - (c) local merge in base-2 with max-subtraction; phase-(a) `+inf`/NaN
    sentinels normalized to `-inf` and outputs `nan_to_num`ed. Returns one
    normal partial `(out, lse)`; `forward_mla` never knows it was two-phase.
  - No double KV write: the cache write happened once via the rank-filtered
    `set_mla_kv_buffer`.
  - Everything is static-shape tensor ops → verify CUDA graphs capture fine
    (no eager fallback needed).
- `draft_extend_v2` + DCP is explicitly rejected (EAGLE-only mode; spec under
  DCP is restricted to DFLASH anyway).

### 3. DFlash draft KV under DCP — replicated
- `model_runner_kv_cache_mixin._apply_memory_pool_config`: for
  `is_draft_worker and spec_algorithm.is_dflash() and dcp_size > 1`, the
  draft's `max_total_num_tokens` (hence its MHA pool size) is widened to
  `config.max_total_num_tokens * dcp_size` — the LOGICAL allocator capacity.
  Logical locations index the draft pool directly; zero fa4 changes.
- Draft writes are NOT rank-filtered: `MHATokenToKVPool.set_kv_buffer` only
  filters via `dcp_kv_mask`, which is set in `ForwardBatch.init_new` under
  `is_hip()` only — always `None` on CUDA. The fused-KV-materialize helper and
  the worker's direct `set_kv_buffer` calls likewise write unmasked.
- `dflash_info_v2.prepare_for_decode` watermark math audited: all quantities
  (`kv_committed_len`, `kv_allocated_len`, reserved lens) are per-request
  TOKEN LENGTHS, not pool capacities; allocation goes through the shared
  widened allocator (`alloc_paged_token_slots_extend` uses the allocator's
  own widened page size). Consistent — no change needed.

### 4. Idle / edge paths
- IDLE: model-side gather/merge conditions exclude idle (mirrors the
  flashinfer DCP path); backend metadata build treats idle like decode
  (harmless — no real rows). All DCP ranks in a group share one batch, so
  collective participation is uniform.
- DP-attention padding backstop in `forward_decode`/`forward_extend`
  (`batch_size < forward_batch.batch_size` → re-plan) rebuilds the DCP lens
  fields because it goes through `init_forward_metadata`.
- `get_verify_buffers_to_fill_after_draft` machinery is EAGLE-worker-only;
  DFlash v2 initializes verify metadata via `prepare_for_verify` →
  `load_batch` (graph) / `init_forward_metadata` (eager), both of which run
  the new DCP lens computation.

## Known risks (GPU validation required, ranked)

1. **Zero local KV rows.** With `seq_len < dcp_size` (decode) or a short
   prefix (verify phase a), some ranks get `cache_seqs[i] == 0`. The kernel
   docs promise `+inf` LSE from the split-KV reduction's `sum==0` guard, but
   the non-split path with a zero-length row is unverified. Verify phase (a)
   sanitizes lse AND output; **plain decode does not sanitize the kernel
   output** — if the kernel emits NaN output (rather than 0) for empty rows,
   `_correct_attn_cp_out_kernel` computes `NaN * 0 = NaN`. Test with prompts
   shorter than dcp_size tokens early.
2. **Verify phase (a) with max local prefix 0** (whole batch short): the
   kernel is called with `max_seq_len` clamped to ≥1 and all-zero lens.
   Same zero-row question as above, batch-wide.
3. **fold_sq path for gathered heads.** With TP4×DCP4 on a 64-head model,
   H_gathered = 64 < 128 → `fold_sq` with q_len=8 → `q_chunk=2` folding.
   Supported per the kernel's divisibility rule, but this exact
   (H<128, q_len=8, return_lse, non-causal) combination needs a kernel-level
   smoke test.
4. **`k_scale_float` on `attn_mqa_for_dcp_decode`.** The DCP layer is a
   second `RadixAttention` with the same weight prefix as `attn_mqa`; if the
   checkpoint carries a non-unit k_scale, confirm the scale attributes are
   populated on BOTH layers (the LSE/scale consistency argument assumes the
   same `k_scale` on every rank — it is, per-layer, but must be non-default
   on the dcp layer too). Most fp8-KV deployments here use k_scale=1.0.
5. **CUDA-graph capture of the DCP collectives for TARGET_VERIFY.**
   Decode-mode DCP graphs (all_gather + reduce_scatter with symmetric memory)
   are validated upstream; the verify-mode graphs capture the identical
   collectives with 8× tokens — expected to work, unvalidated. Fallback if it
   misbehaves: skip verify graph capture when `dcp_enabled()` (decode graphs
   matter more).
6. **Non-causal + LSE prefill-phase compile.** Phase (a) uses a decode-kernel
   variant (`is_causal=False, with_lse=True`, q_len=8) that will be
   cute.compile'd on first use (~1–2 min JIT); consider pre-warming like the
   prefill kernels if the scheduler watchdog complains.
7. **`maybe_detect_oob` bounds under DCP** (pre-existing): `set_mla_kv_buffer`
   checks logical locs against the physical `size + page_size` bound; with
   `SGLANG_ENABLE_ASYNC_ASSERT` enabled this false-positives under DCP.
   Same for the flashinfer DCP path — not introduced here, not fixed here.
8. **Draft pool memory** = dcp_size × draft KV. Fine for a few-layer GQA
   draft, but check `mem-fraction-static` headroom on 4×B200 with dcp=4.

## GPU validation checklist

- [ ] Launch: `--attention-backend tokenspeed_mla --kv-cache-dtype fp8_e4m3
      --dcp-size 4 --tp-size 4 --speculative-algorithm DFLASH
      --speculative-num-draft-tokens 8 --speculative-draft-attention-backend fa4`
- [ ] Non-spec DCP decode first (drop the speculative flags): GSM8K vs
      non-DCP tokenspeed baseline; then logprob parity.
- [ ] Short-prompt decode (< dcp_size tokens) — risk #1.
- [ ] DFlash verify: avg_spec_accept_length ≈ non-DCP baseline (a broken
      merge shows up as accept length pinned near 1).
- [ ] temp=0 output parity DCP vs non-DCP with DFLASH enabled
      (`test/registered/dcp/test_dcp_tokenspeed_dflash.py`, parity class).
- [ ] CUDA-graph on/off parity (`--disable-cuda-graph`) for both decode and
      verify to isolate graph-capture issues.
- [ ] Radix-cache prefix hits under DCP + DFlash (draft pool replication
      interacts with prefix reuse through the shared logical allocator).
- [ ] Memory: confirm draft pool widening doesn't tip mem-fraction-static.
