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

## Full finite-map pipeline (queued 2026-09-08 18:16 UTC)

Candidate source is `b0363d743e`; finite map introduced in `0193aa7c62`.
The table is finite data, not a runtime reimplementation of undocumented
FlashInfer heuristics. All 1,360 entries equal the validated GPU artifact;
164,400 host context resolutions and incompatible-configuration guards passed.
`all` mode requires FlashInfer0.6.17,148SM,Q16,topk2048,KPool4,KV/V512,
qk_nope256,RoPE0,page64; underlying captured-DSA checks require FP8 KV.

Full mode adds a runner-local token bucket1 before buffer/capture setup because
normal BCG starts at4 and rejects the4x padding for a single token. N>=2 keeps
its original bucket. The same-GPU control retains its original wholly eager
N=1 path, so final parity explicitly tests this difference.

Current app: `ap-oy2gzmIG6HBdeghbGiw8t4`.
Driver: `fc-01M213RPBS48W4XHH40908HRPC`.
Run: `20260908T181628.271898Z-dsa-exact-variant-full-smoke`.
Image: `im-8ohbrpsKyfZTWccCHxPWPn`.
This waits on pilot driver `fc-01M213CKYMT5226Y0Q3BFFD8KQ`; it does not allocate
GPUs until the pilot passes and measured cost passes the gate below.
Superseded waiting app `ap-GKdfFYyr6OUsFvVcrkKPJZ` was stopped before any GPU launch.

Gate: each rank's16 measured captures projected to360 must consume<=30minutes
and<=8GiB extra total device memory. A200ms NVML observer includes CUDA driver
and executable allocations; correlation uses conservative timestamp margins.
The observer is in the pilot container and writes `device-memory.csv` alongside
`server.log`. Dependency errors, missing memory evidence, or gate failures
produce a durable stopped report. During full capture the runner independently
checks actual CUDA mem_get_info growth and elapsed time against the same limits.

After admission the detached job runs2,208 numerical cases on the same4B300s
for candidate then control, with the same random seed and recipe/model/draft:

- Original535 boundary/ragged/prefix cases,32greedy output tokens each.
- Every exact1..80 live count at every17canonical context class:1,360cases,
 8greedy output tokens each, alternating context classes to change variants.
- Small ragged totals2..80:313individual request comparisons,8outputs each.

It requires2,208/2,208 equal greedy sequences and max selected teacher logprob
absolute difference<=0.05. It also runs the original33stress cases, long context,
multimodal validation and all-rank profiles; a separate profile requires12launches
for every1..80 live count and actual bs4/total74,75,80 prefills. The control clears
both DSA and variant environment flags but retains captured Triton KDA.

At queue time the pilot was still in normal model startup/DeepGEMM warmup.
Neither pilot serving profiles nor full model parity were complete.
Full harness: `glm53-kda-benchmarks/dsa-variants-handoff-20260908/full-harness.tar.gz`.
Each archive has its own SHA256 manifest. Final pass/fail lives in each run's
`run.json`; numerical parity lives in `matched-comparison.json`.


## Pilot passed; full validation started (2026-09-08 18:37 UTC)

The pilot passed published startup, all16 fresh/prefix serving cases, all-rank
12-launch profile assertions, and final health. Every one of64 startup variant
records reports12segments. This is finite/profile validation, not a candidate
versus control numerical comparison.

Per-rank incremental capture seconds:26.9242,26.9239,26.9247,26.9249.
Driver-inclusive peak device increase:160MiB on each GPU. Conservative360-plan
projection:605.8seconds (~10.1minutes),3600MiB (~3.52GiB) per GPU. Cost gate passed.
The raw pilot report is `<pilot-run>/run.json`, cost evidence is
`dsa-variants-handoff-20260908/pilot-capture-cost.json`, and profiles are
`glm53-torch-profiles/<pilot-run>/variants`.

Full pipeline automatically cleared both gates and launched GPU worker
`fc-01M214YQMXPGC6F431GNEJMHTX` under app `ap-oy2gzmIG6HBdeghbGiw8t4`.
It is in published startup;2,208-case matched parity remains pending.
The earlier startup delay was Triton KDA cache-file loading on the eager
multimodal/health warmup paths and resolved without code changes.
