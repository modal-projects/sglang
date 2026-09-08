# GLM 5.3 Flash exact-count DSA graph variants

Experimental branch `willhu/glm53-dsa-variants`; no production deployment.
Workspace/environment: `modal-labs / glm-bringup`.

## Purpose and implementation

The DSA TRTLLM launcher chooses different arithmetic from the exact query count
and maximum context bound. Padding 1–80 live queries to BCG buckets or replacing
short context bounds with 2052 caused deterministic model numerical differences.
See `HANDOFF_DSA_NUMERICS.md` for the localization and conservative eager guard.

Commit `771af619ad` adds explicit startup graph variants keyed by live token count
and context bound. The surrounding model body and BMM remain padded to the
original BCG bucket. Only DSA attention/KV writes use exact live views; padded
attention output is zeroed. CPU batch metadata chooses the variant before replay.
All variants capture inside the existing startup capture session/global pool;
no capture is attempted against a live serving cache.

`SGLANG_DSA_PREFILL_CUDA_GRAPH_VARIANTS` accepts explicit `tokens:bound` entries.
Bounds are rounded up to multiples of 128, clamped to 2052. Captured KDA and DSA
must both be enabled for the intended comparison. The draft recipe is unchanged.

## Completed focused validation

- Exact-attention prototype: 90/90 valid layouts bitwise exact, all captured.
- Modified bridge: 275 layouts × 3 changed-input replays = 825/825 bitwise exact,
  finite outputs, zero padding, identical KV cache, reserved slot 0 untouched.
  Includes padded BMM, exact DSA subviews and actual KV scatter; every N=1..80 at
  fresh/384/8192 prefix, plus larger/query/context boundaries and 65536 prefix.
- Context coalescing: 360 captured plans; 5760/5760 bitwise exact changed-input
  comparisons for every N=1..80, all 17 context classes at their low/high bounds,
  and clamped 8192/65536 prefixes. FlashInfer 0.6.17, B300 148 SMs, Q16, KV/V512,
  qk_nope256, no RoPE, FP8 KV, KPool4, topk2048/page64.

These are kernel/bridge proofs, not integrated model parity.

Bridge app `ap-vyH7ybADQcVukeJRNaXroq`, call
`fc-01M213A8GPJNNDMAYQM97D08DN`, run
`20260908T180835.337329Z-dsa-variant-bridge-probe`.

Plan proof app `ap-F3SZWRoDon0qxeKpQoxCt0`, call
`fc-01M2130HDK1R6PYTYX8Z10XFRV`, run
`20260908T180316.775839Z-dsa-context-plan-probe`.
The finite map/raw results/source are in `glm53-kda-benchmarks` at that run path.

## Startup pilot (launched, not yet validated)

App `ap-nGcBSK5rhBWHjU5vqeYeYh`.
Driver `fc-01M213CKYMT5226Y0Q3BFFD8KQ` cleared the bridge dependency.
GPU worker `fc-01M213CT9H7A5PT0DT73RZ12X0`.
Run `20260908T180952.587214Z-dsa-exact-variant-startup-pilot`.
Immutable image `im-h3YPXRZ2O9n3GFse14i3vP`.

16 extra captures: N3,16,31,65 each at bounds128/512/2052, N74,80 at128/2052.
Measures incremental startup seconds, allocator deltas, and12segments pervariant.
Then profiles all16 selected serving cases on all4ranks; no parity claim.

Base source image pin b418d1c0387a0f4deb83796c2c0f217feb0be103 plus recorded
source overlays. Recipe 1816f3b6aa0186095f4f927e771234b7b5be532d; model revision
690b705278a3a58e538fcb37c2ca8b5f9511213c; DFlash2 epoch2 block8, Triton KDA,
TRTLLM DSA, TP4/EP4, seed479309393. The production server flags are unchanged.

## Next gates

1. Pilot completes with affordable startup/memory and12graph launches for small
   selected prefills on every rank.
2. Use validated finite map to reduce1360 combinations to360 startup variants,
   enforce the exact validated hardware/kernel/dimension configuration.
3. Same-GPU matched integrated comparison: captured KDA+DSA variants versus
   captured KDA/eager DSA. Require identical greedy generations and selected
   teacher logprob maxabs<=0.05, unchanged from the existing criterion.
4. Cover original535 cases, every1..80 short count, context boundaries, changed
   variants across successive requests, ragged multi-request totals, prefix,
   long-context and multimodal stress, plus all-rank profiles.

Do not call the full graph route model-validated before these gates pass.

## Durable artifacts

All Modal volume references are environment `glm-bringup` in `modal-labs`.
Per-run source, settings, logs/results: `glm53-kda-benchmarks/<run-id>`.
Profiles: `glm53-torch-profiles/<run-id>`.
Local full harness: `/home/ec2-user/artifacts/glm53-dsa-variants-20260908`.
A tar archive is uploaded under `dsa-variants-handoff-20260908` with SHA manifest.
