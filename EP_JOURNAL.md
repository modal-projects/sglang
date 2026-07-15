# Inkling Expert-Parallelism (`--moe-a2a-backend none`) — Engineering Journal

- **Branch:** `cheng/dev/moe-ep` (off the merged FusedMoE-refactor `dev`)
- **Commit:** `d9f4df737f` — "[Inkling] Support expert parallelism (--moe-a2a-backend none) for InklingMoE"
- **Goal:** Support EP for InklingMoE with `--tp 4 --ep-size {2,4} --moe-a2a-backend none`; benchmark + verify accuracy on real NVFP4 weights.
- **Outcome:** ✅ Working. gsm8k **TP4+EP2 = 0.910**, **TP4+EP4 = 0.930** (0% invalid, matches TP4 baseline); bsz1 decode ~90 tok/s (perf-neutral).

---

## TL;DR

EP looked "almost free" (the stock `FusedMoE` + NVFP4 kernel already handle EP), but the model produced garbage (gsm8k ~5%, ~82% invalid). After eliminating every NVFP4-path hypothesis, a **TP-vs-EP per-layer output diff** revealed the corruption starts at the first MoE layer and **compounds**. That layer's routed experts turned out to be **bf16 (excluded from NVFP4)**, taking the `moe_tp_forward` grouped-GEMM path — which is **not EP-aware** (it indexes local expert weights with the gate's global topk ids). Fixing that path (plus regrouping the shared sink expert to full-TP) made EP correct.

---

## Investigation timeline

1. **Wired EP** assuming the NVFP4 routed path handles it: added `_slice_local_experts` (slice the checkpoint's fused 256-expert tensor to each rank's local ep block) and a shared-expert `1/ep_size` rescale. Loads clean, but gsm8k = 5% / 82% invalid. TP4 baseline (my code inert at ep=1) still 0.92 → EP-specific.
2. **Ruled out the NVFP4 routed path** across 6 variants — global-ids+offset, zero output buffer, dispatcher remap to local + offset 0, Inkling topk mask, local-space remap, and a *self-consistent 128-expert* view (global=local, offset 0). All failed identically (~82% invalid). Verified statically: expert partition, weights/scales sizing, `local_expert_offset`, `intermediate_size_per_partition`, and `StandardDispatcher.skip_local_expert_mapping` (intentionally True for trtllm_routed) were all correct.
3. **Shared sink expert** (per user hint + ARCHITECTURE.md: sink weights are jointly normalized with routed, so it's integral to every token): tested moe_tp-sharded+scale, replicated, and **full-TP shard** (user's guidance). Full-TP is the correct design and made the shared contribution exact — but gsm8k still 82% (something else was also wrong).
4. **Numerical TP-vs-EP diff** (user's suggestion): dumped each MoE layer's post-all-reduce output for identical (zeros) input. Divergence is **small at L2 and compounds** to garbage by L65; **routed** field diverges 36% at L2 while **shared** is 0.00% (exact) → the routed path is the bug, and the shared full-TP fix is correct.
5. **Root cause:** a per-layer weight dump showed L2's routed experts are `UnquantizedFusedMoEMethod` — **bf16, excluded from NVFP4**, interleaved w13. Those layers use `moe_tp_forward`, which I never made EP-aware. All prior NVFP4-path fixes were moot because these bf16 layers always corrupted the stream.

---

## The fix (`moe.py` + `inkling.py`)

1. **Routed NVFP4 path** — already EP-correct (kernel `local_expert_offset`). Only the load needed a fix: slice the fused 256-expert checkpoint tensor to this rank's contiguous ep block (`_slice_local_experts`, gated on `".experts." in name and shape[0]==n_routed_experts`).
2. **Routed bf16 path** (`moe_tp_forward`) — **the main bug.** In `_forward_routed`'s unquantized branch, remap the gate's global topk ids to local space (`topk_ids - ep_rank*local`) and zero non-local weights, so only local experts contribute and the full-TP all-reduce sums the disjoint subsets.
3. **Shared sink expert** — build with the **full tp group** (`get_tensor_model_parallel_rank/world_size`), not `moe_tp`; it's a replicated dense MLP, so the single full-TP all-reduce reconstructs it exactly (no rescale). Its NVFP4 block scales load with the full-TP rank (`_load_nvfp4_scale_param`: full-tp for `.shared_experts`, moe_tp for routed).
4. No changes to the gate, the NVFP4 kernel, or the dispatcher.

---

## Validation (bootstrap2, real NVFP4 weights, GPUs 4–7)

| Config | gsm8k acc | invalid | bsz1 | bsz16 | bsz64 (decode tok/s) |
|---|---|---|---|---|---|
| TP4 baseline | ~0.92 | — | ~91 | — | — |
| TP4+EP2 | **0.910** | 0.000 | 90.9 | 762 | 1861 |
| TP4+EP4 | **0.930** | 0.000 | 89.0 | 747 | 1829 |

Benchmark = `bench_one_batch_server` (input 4096, output 128). Representative subset; full CLAUDE.md sweep (4096/8192/16384 × 1/4/16/64) not yet run.

---

## Reproduce (bootstrap2, inside `sgl_cheng`)

Engine launch — EP2 (use `--ep-size 4` for EP4; pick a free 4-GPU block via `CUDA_VISIBLE_DEVICES`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
SGLANG_OPT_USE_FUSED_GATE_TOPK=1 \
python -m sglang.launch_server \
  --model-path /data/cache/model-share-nvfp4 \
  --trust-remote-code \
  --tp 4 --ep-size 2 --moe-a2a-backend none \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_trtllm \
  --moe-runner-backend flashinfer_trtllm_routed \
  --enable-torch-symm-mem \
  --attention-backend fa4 \
  --mamba-scheduler-strategy extra_buffer \
  --disable-custom-all-reduce \
  --page-size 128 \
  --reasoning-parser inkling --tool-call-parser inkling \
  --mem-fraction-static 0.85 \
  --max-mamba-cache-size 1024 \
  --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup \
  --host 127.0.0.1 --port 30055
```

Diff vs the canonical CLAUDE.md launch: add `--ep-size {2,4} --moe-a2a-backend none`; the model path here is `/data/cache/model-share-nvfp4`. One-time on a fresh `sgl_cheng`: `pip install helion`.

Accuracy + benchmark against the running server:

```bash
python -m sglang.test.few_shot_gsm8k --num-questions 200 --num-shots 8 --port 30055 --parallel 32
python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30055 \
  --model-path /data/cache/model-share-nvfp4 --batch-size 1 16 64 --input-len 4096 --output-len 128 --skip-warmup
```

---

## Infra notes

- **Access:** `ssh bootstrap1|bootstrap2` → `sudo docker exec sgl_cheng` (host container; not di1/di2). Repo bind-mounted at `/sgl-workspace/sglang`; local edits mirror in.
- **Model path:** `/data/cache/model-share-nvfp4` (CLAUDE.md's `/data/cache/huggingface/hub/...` is stale on these nodes).
- **Missing dep:** `sgl_cheng` on bootstrap2 lacked `helion` → "InklingForConditionalGeneration is not a registered model". Fixed with `pip install helion`.
- **Shared cluster:** di1/di2/bootstrap1/bootstrap2 heavily occupied by teammates (yanbin_sharedmoe, sgl_chunan, ...). Map GPU PIDs → containers via `sudo docker top`. Killed servers leak GPU memory via lingering `sglang::scheduler_TP*` children (`pkill -f "ep-size N"` misses them); relaunching before memory frees → OOM. An auto-launcher (watch for a free block of 4, then launch) was used to catch brief free windows.

---

## Follow-ups

- Run the full benchmark sweep (input 4096/8192/16384 × bsz 1/4/16/64) for EP2/EP4 vs TP4.
- Push `cheng/dev/moe-ep` / open a PR.
- AIME/GPQA accuracy under EP (optional; TP4 AIME on this cluster is slow/OOM-prone — see `inkling-aime-tp4-oom-config`).
