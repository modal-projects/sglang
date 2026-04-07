# Plan Review: Disaggregated RL Policy Lifecycle

Quadruple-verified audit of `docs/developer_guide/disaggregated_rl_policy_lifecycle_plan.md`. Four rounds of targeted investigation. Every claim traced to line numbers.

---

## A. Where The Plan Mischaracterizes Existing Code

### A1. LoRA Registry Gap Is Narrow But Hard

**Already exists:** `LoRARegistry` (`lora_registry.py:79`) maps `lora_name` -> `LoRARef` in LRU `OrderedDict`. Fresh UUID `lora_id` per load (`:36`). Refcount drain via `ConcurrentCounter` + `wait_for_unload()` (`:175-192`). Non-blocking `acquire()`/`release()` (`:115-154`). Auto-reload of evicted adapters via `lora_ref_cache` (`tokenizer_manager.py:2276-2290`). Two-phase prepare/activate via `LoRAOverlapLoader` on dedicated CUDA stream (`lora_overlap_loader.py`; scheduler calls `try_overlap_load_lora()` at `scheduler.py:2405-2409`).

**Precise blocker:** `register()` rejects duplicate `lora_name` with `ValueError` (`:230-233`). Manager side also rejects (`:169-174`). Two versions of the same alias **cannot coexist**. This is the single structural blocker for drain-on-retire.

**Good news:** LoRA weights use indirection via `weight_indices` into a memory pool; CUDA graphs capture the indirection pattern, not specific weights (`cuda_graph_runner.py:909,975`). No graph recapture needed for LoRA switches. Overlay policy transitions are cheap at the GPU level.

**Where the abstraction fits:** Split `_registry` into `_versions: Dict[lora_id, LoRARef]` + `_alias_map: Dict[lora_name, lora_id]`. `acquire()` gains optional `pinned_version` parameter. Separate eviction (memory pressure) from retirement (policy superseded).

### A2. Sessions Are First-Class; Policy Pinning Is One Field Away

**Already exists:** `SessionController` (`session_controller.py:251-335`) with timeout, tree-structured `SessionReqNode`, `SessionAwareCache` for streaming sessions. `SessionParams` on every request (`io_struct.py:107-112`).

**Precise gap:** Each request independently resolves LoRA (`session_controller.py:217` passes `recv_req.lora_id` unchanged). No session-level `pinned_policy_version`. `Req` has no `weight_version` field.

**Where the abstraction fits:** Add `pinned_policy_version: Optional[str]` to `Session`. Check at `_resolve_lora_path` (`tokenizer_manager.py:2253`). Requires A1's coexistence fix.

### A3. Retract Mode Provides Partial `retract_recompute`

**What happens:** `release_req()` frees KV pool memory via `release_kv_cache(is_insert=False)` (`schedule_batch.py:2025-2026`) -- KV freed, NOT re-inserted into radix tree. `evict_from_tree_cache()` aggressively reclaims more (`:2029`). On re-entry, `match_prefix` finds whatever prefix survived from other requests (`schedule_batch.py:975-977`).

**Grammar state survives retraction.** `reset_for_retract()` (`:1183-1220`) does NOT clear `req.grammar`. Compiled grammar objects persist; no reparsing needed on resume. Token acceptance state is preserved via `grammar.current_token` check (`decode_schedule_batch_mixin.py:116-119`).

**`reuse_stale_kv`** has no implementation. `in_place` mode preserves KV but doesn't allow weight mutation while requests are held.

---

## B. Correctness Issues The Plan Must Address

### B1. LoRA Refcount Leak In Waiting Queue Aborts

`lora_registry.release()` is called in exactly 2 places (`tokenizer_manager.py:1683` for normal completion, `:1230` for scheduler-initiated abort with status 503/500). But waiting queue aborts (scheduler.py:3170, Method 1) send bare `AbortReq` **without** an error status code. The tokenizer's release path at `:1230` only fires on `SERVICE_UNAVAILABLE` or `INTERNAL_SERVER_ERROR`. Waiting queue aborts may **leak LoRA refcounts**.

This must be fixed before drain-on-retire can work. Every exit path must release exactly once.

### B2. HiCache Storage Keys Ignore `extra_key` -- Stale KV Correctness Risk

HiCache's storage-level hash (`radix_cache.py:225-255`, `hicache_storage.py:17-32`) computes keys from token IDs only, **not** from `RadixKey.extra_key`. Two requests with different `extra_key` values (different LoRA IDs or policy versions) but same token sequence will hash to the same storage key.

After a weight update: stale KV in host/disk cache from old policy could be served to requests under new policy. Device-level radix cache is correct (partitioned by `extra_key`), but storage-level cache is not. `flush_cache()` clears the in-memory radix tree but does **NOT** invalidate storage backend.

This is a latent correctness bug independent of the plan, but the plan's cache policy work must fix it.

### B3. Disaggregated Event Loops Do NOT Check `_engine_paused`

Standard event loops check `_engine_paused` and skip batch scheduling (`scheduler.py:1306-1308`). Disaggregated prefill (`disaggregation/prefill.py:384-455`) and decode (`disaggregation/decode.py:1139-1199`) event loops have **no `_engine_paused` check**. Pause/resume may not work correctly in disaggregated serving mode.

Weight updates dispatched via `process_input_requests()` happen **before** the pause check in standard mode (`:1599` before `:1306`), which is safe because they're synchronous. But in disaggregated mode, batch scheduling proceeds unconditionally after weight update dispatch.

### B4. `extra_key` From API Is Silently Dropped

`scheduler.py:1742-1774` constructs `Req()` but does not pass `recv_req.extra_key`. The API computes `extra_key` from `cache_salt` + `extra_key` (`serving_base.py:151-162`), tokenizes it, but the scheduler drops it. Final `Req.extra_key` is either `None` or just `lora_id`. Must be fixed before policy-version cache namespacing can work via `extra_key`.

---

## C. Things The Plan Misses

### C1. LoRA Updates Hard-Blocked For dp_size > 1

Unconditional `assert dp_size == 1` on `load_lora_adapter` (`tokenizer_communicator_mixin.py:654-656`), `load_lora_adapter_from_tensors` (`:730-732`), `unload_lora_adapter` (`:807-808`). TODO comment: "Remove after we verify..." (`:652-653`). No bypass.

Weight mutations bypass with `--enable-dp-attention`: `assert dp_size == 1 or enable_dp_attention` on `update_weights_from_tensor` (`:567-569`), `update_weights_from_distributed` (`:509-511`), `update_weights_from_ipc` (`:600-602`).

### C2. Four Dead Fields + One Dead Output Field

**Input fields** on `UpdateWeightFromDiskReqInput` (`io_struct.py`) documented in `sglang_for_rl.md` but never read: `is_async` (`:1232`), `keep_pause` (`:1236`), `token_step` (`:1240`), `torch_empty_cache` (`:1234`).

**Output field** `token_steps` on `BatchTokenIDOutput` (`io_struct.py:1012`) is defined but **never populated** by the scheduler output processor. Dead on both sides.

### C3. Draft Worker Stale On 3 of 4 Paths

`update_weights_from_disk` (`:50`), `update_weights_from_distributed` (`:78`), `update_weights_from_ipc` (`:110`) in `scheduler_update_weights_mixin.py` update only `self.tp_worker`. Only `update_weights_from_tensor` (`:91-94`) has `disable_draft_model` flag for conditional draft update. EAGLE workers override tensor path to update both (`:1019-1035`, v2 `:1045-1061`).

### C4. Engine API Lacks pause_generation / continue_generation

`EngineBase` abstract class does not define `pause_generation` or `continue_generation`. These are HTTP-only endpoints (`http_server.py:1356,1367`). RL integrations using the Python `Engine` API cannot pause before weight updates. They rely on `model_update_lock` (RWLock writer lock) instead, which blocks inference but doesn't retract/flush.

### C5. Multi-Tokenizer Mode: Separate LoRA Registries Per Worker

When `tokenizer_worker_num > 1`, each `TokenizerWorker` gets its own independent `LoRARegistry` instance (`tokenizer_manager.py:383-400`). No cross-worker synchronization. LoRA state can diverge between workers. Refcounts are per-worker. This interacts poorly with drain-on-retire if different workers have different refcount views.

### C6. Pipeline Parallelism: No Pause Coordination Across Stages

PP event loops (`scheduler_pp_mixin.py`) run separately. `pause_generation` has no PP-specific logic. `is_fully_idle()` requires all `running_mbs` empty (`scheduler.py:2911`) but retract doesn't coordinate across PP stages. No PP-specific test for weight updates exists.

### C7. `release_memory_occupation` / `resume_memory_occupation` Unmodeled

Central to RL GPU timesharing. Release offloads by tag (`kv_cache`, `weights`, `cuda_graph`) via `scheduler_update_weights_mixin.py:124-155`. Requires `is_fully_idle()`. Checkpoint engine's `--checkpoint-engine-wait-weights-before-ready` gates readiness on first load (`http_server.py:1118-1119`) -- a nascent "first activation" concept. RL tests show multi-stage release/resume sequences (`test_release_memory_occupation.py`).

### C8. Quantized RL: `flash_rl` Load Format

`QuantizedRLModelLoader` (`model_loader/loader.py:786-1192`) provides a dedicated path for FP8 RL training: receives BF16 from trainer, quantizes to FP8, updates scales. Special handling for stacked parameters (`qkv_proj`, `gate_up_proj`). Uses `torch.as_strided()` for in-place memory preservation. The plan doesn't mention quantization constraints; the mutation backend interface (Phase 5) should expose format requirements.

### C9. MoE + LoRA: 4D Expert-Aware Tensors

LoRA supports 4D expert-aware tensors (`lora/mem_pool.py:197-326`): `[num_loras, num_experts, rank, hidden_dim]`. Fused MoE+LoRA computation in `lora_moe_runners.py`. `return_routed_experts` captures which experts were routed (`layers/moe/routed_experts_capturer.py`). Policy versioning is model-level, not expert-level. Relevant for Phase 7 (sparse mutation) but the plan doesn't account for expert-level granularity.

### C10. No LoRA Provenance In Responses

`meta_info` includes `weight_version` (`tokenizer_manager.py:1553`) but NOT `lora_id`. `lora_id` is available in `state.obj` but never added to `meta_info`. The `customized_info` mechanism (`io_struct.py:1017`, `tokenizer_manager.py:1598-1600`) could carry it with no schema changes.

---

## D. Abstraction Mapping: Plan Phases -> Existing Code

### Phase 1: Policy Lifecycle Types

New module under `python/sglang/srt/`.

| Concept | Existing code | New work |
|---|---|---|
| `PolicyAlias` | `lora_name` / `weight_version` | Unified type |
| `PolicyVersion` | `lora_id` UUID / `weight_version` string | Common version type |
| `BackendFamily` | `lora_update_lock` vs `model_update_lock` | `OVERLAY` / `MUTATION` enum |
| `PreparedUpdate` | `LoRAOverlapLoader` (overlay, already two-phase) / none (mutation) | Generalize overlap loader |
| `RetirementPolicy` | `wait_for_unload()` + eviction / none | `EVICT_NOW` / `DRAIN` / `DRAIN_UNTIL_DEADLINE` |
| `CachePolicy` | `flush_cache` bool + pause modes | `FLUSH` / `RETRACT_RECOMPUTE` / `REUSE_STALE_KV` |
| `PolicyCapabilities` | Nothing | New struct: `supports_draft_update`, `requires_graph_recapture`, `supports_coexistence`, `supports_drain`, `format_constraints` |

### Phase 2: LoRA As Overlay Backend

**Files:** `lora_registry.py`, `tokenizer_communicator_mixin.py`

Split `_registry: OrderedDict[lora_name, LoRARef]` into `_versions: Dict[lora_id, LoRARef]` + `_alias_map: Dict[lora_name, lora_id]`. Add `superseded: bool` to track retirement state separately from eviction. `acquire()` resolves alias to current target unless a pinned version is requested.

**CUDA graph safety:** Confirmed no recapture needed for LoRA switches (weight indirection via `batch_info.weight_indices`). Overlay transitions are GPU-cheap.

**DP:** Either lift `dp_size == 1` assertions or scope to single-DP.

### Phase 3: Session Policy Pinning

**Files:** `session_controller.py`, `tokenizer_manager.py`

Add `pinned_policy_version: Optional[str]` to `Session`. Check in `_resolve_lora_path`. Requires Phase 2.

### Phase 4: Cache Policy

**Files:** `io_struct.py`, `scheduler_update_weights_mixin.py`

Add `CachePolicy` enum. Map: `flush_cache=True` -> `FLUSH`, `flush_cache=False` -> `RETRACT_RECOMPUTE`. New `REUSE_STALE_KV` requires allowing in_place weight mutation.

**Must fix first:** B4 (`extra_key` passthrough bug), B2 (HiCache storage key must include `extra_key`).

### Phase 5: Mutation Backend Interface

**Files:** `scheduler_update_weights_mixin.py`, `tp_worker.py`, `model_runner.py`

Four transports with different capabilities:

| | Disk | Tensor | Distributed | IPC |
|---|---|---|---|---|
| Draft update | No | Conditional | No | No |
| Graph recapture | Yes | No | No | No |
| Rollback | Yes | No | No | No |
| Partial params | Yes (name filter) | No | No | No |
| Quantized RL | Via `flash_rl` | Custom loader | `flattened_bucket` | No |

**Must also:** Fix draft worker staleness on 3 paths. Model `release_memory_occupation`/`resume_memory_occupation` as lifecycle transitions. Add `pause_generation`/`continue_generation` to `EngineBase` for Python API parity. Use `InputBlocker` pattern (`scheduler_input_blocker.py`) for cross-DP coordination.

### Phase 6: Provenance

Add `lora_id` to `meta_info` (trivial: available in `state.obj`, add one line at `tokenizer_manager.py:~1555`). Alternatively use `customized_info` mechanism for zero-schema-change path.

---

## E. Prerequisite Fixes (Before Any Phase)

These are bugs or gaps that must be resolved before the lifecycle work can proceed cleanly:

1. **Fix B1** (LoRA refcount leak in waiting queue aborts) -- drain-on-retire requires correct refcounting in all exit paths
2. **Fix B4** (`extra_key` not passed from API to `Req` in `scheduler.py:1742-1774`) -- blocks policy-version cache namespacing
3. **Fix B2** (HiCache storage hash must include `extra_key`) -- correctness risk independent of lifecycle work
4. **Clean up C2** (dead fields: `is_async`, `keep_pause`, `token_step`, `torch_empty_cache`, output `token_steps`) -- implement or remove before adding new API surface
5. **Fix B3** (add `_engine_paused` check to disaggregated event loops) -- pause must work in all modes

## F. Priority Adjustments

1. **Phase 1** should include DP-awareness. Use `InputBlocker` 3-state machine (`scheduler_input_blocker.py`) as template for cross-DP policy transitions.
2. **Phase 2** is smaller than described: overlap loader already provides prepare/activate, registry already has refcounting/drain. Real work: version coexistence, retirement vs eviction, DP restriction.
3. **Phase 3** is one field on `Session` + one check in `_resolve_lora_path`.
4. **Phase 4** requires fixing `extra_key` passthrough and HiCache storage keys first. After that, the enum is straightforward since `retract` mode already provides most of `RETRACT_RECOMPUTE`.
5. **Phase 5** should fix draft worker staleness, add Engine API parity (`pause`/`continue`), model memory release/resume, and expose quantization format constraints.
6. **Phase 5** must also address multi-tokenizer registry divergence (C5) and PP pause coordination (C6).
