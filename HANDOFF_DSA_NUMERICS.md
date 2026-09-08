# DSA prefill capture numerics — 2026-09-08

Workspace/environment: **modal-labs / glm-bringup**. Production is unchanged.

Source branch: `modal-projects/sglang`, `willhu/glm53-dsa-numerics`.
Base `333b53d48a`; narrow capture guard `d3d465beb2`.
Local checkout: `/home/ec2-user/work/sglang-glm53-dsa-numerics`.

## Cause

The previous captured DSA path substituted a fixed `max_seq_len=2052` for
the eager maximum context length and padded the launch's batch/token count
to the BCG bucket. Both host scalars affect the pinned FlashInfer 0.6.17
TRTLLM sparse MLA launch plan. This is reproducible eagerly and does not
require a graph replay race.

The pinned source is archived under `flashinfer-source/`. In
`data/include/flashinfer/trtllm/fmha/fmhaKernels.cuh`:

- Lines 700–722 select Q8 vs Q16 head tiles from the token count times
  `ceil(min(max_seq_len, topk) / 128)` and the SM count.
- Lines 529–565 select the KV split/reduction count from
  `ceil(min(max_seq_len, topk) / 256)`, limited by SM occupancy.
- Lines 615–632 further split the V head dimension if SMs remain unused.

On this B300 (148 SMs), padding 65–74 real tokens to an 80-token launch
switches from a multi-CTA KV reduction to the persistent single-CTA kernel.
75–80 real tokens already use the persistent kernel. Small fresh prefills
also change their head tiling and/or V splits when the context bound becomes
2052. These alter floating-point arithmetic/reduction order.

The 456-case `probe.json` is a **dispatch mapping**, not a correctness pass.
It deliberately keeps long-context sparse rows while sweeping the host bound.
Bounds below the actual context/sparse length are out-of-contract dispatch
experiments and must not be cited as correctness evidence. Valid long-context
padding comparisons and the preceding valid fresh-context diagnostic separately
establish the failure.

## Narrow correction

Keep the original eager DSA+BMM break for capture buckets whose minimum
possible live count is at most `SM_count // 2`. For supported 8/16/32 query
heads, greater live counts rule out all three launch-plan changes above.
The cutoff is derived from device SM count and actual configured bucket
boundaries, rather than assuming a fixed GPU or token threshold.

For the current B300 bucket list:

- **1–80 live tokens:** DSA+BMM stays eager; KDA remains captured (23 segments).
- **81–4096 live tokens:** DSA+BMM and KDA are captured (12 segments).
- **Above 4096:** existing prefill graph fallback remains in effect.

This is a conservative fallback, **not full DSA capture for every shape**.
Capturing the small shapes while retaining eager arithmetic requires launch-plan
variants selected from the existing CPU live-token/max-context metadata (Q tile,
KV split count, V split), or an explicit launch plan used consistently by eager
and captured calls. A fixed context bound alone is insufficient.

## Validation and runs

`guard-result.json`: **90/90 exact** valid layouts, including fresh 1–4096
tokens, prefixes 128/255/256/257/512/2048/8192/65536, and the 74/75/80/81 padding
boundaries. All finite; output padding zero. Small cases exercise the preserved
eager break; eligible large cases exercise actual captured attention. Do not
claim that all 90 cases ran attention inside a graph.

All repository pre-commit hooks and Python compilation passed for the three
modified source files.

| Run | App | Call |
| --- | --- | --- |
| Dispatch mapping | `ap-ScskqO9YcZSoC91hxObbXm` | `fc-01M211F2E3MBESY6CBGX491WC1` |
| Valid-layout guard probe | `ap-U0KF0s7ZsLgGMwG1xz5J7L` | `fc-01M211KQ0YFGMP0ZRACSSYSCNH` |
| Integrated matched smoke | `ap-BqufpvnK3UscFF1FvKoOCE` | `fc-01M211PDCRWEC2J55WGFB5GY2K` |

The complete probe/harness archive is in `glm53-kda-benchmarks` at
`dsa-numerics-handoff-20260908/harness.tar.gz`, with hashes in the adjacent
`archive-manifest.json`.

All runs use retries=0. Results are in `glm53-kda-benchmarks` under the run IDs
from the corresponding `launch-*.json` files. The integrated smoke is detached
and gated on the passed guard probe. It uses the unchanged serving recipe,
Triton KDA, TRTLLM DSA, DFlash2 epoch-2 block-8, and seed 479309393.

The GPU child call is `fc-01M211PJA27ZNY7QY5HH0VVHC7`; immutable image
`im-lqx8gZaE2jbRIROnN0R4To`. It cleared its dependency and entered published
startup.

Candidate and control run on the same four B300s. **Control keeps KDA captured
and DSA eager** to isolate the DSA change. Required numerical gate: all 535
greedy outputs equal, selected teacher-forced log probabilities within 0.05.
The original 510 cases are expanded with 25 targeted boundary/prefix cases.
The lane also runs the 33-case stress suite, multimodal replay, and all-rank
profiles. Integrated parity remains pending until its completed result is read.

Launch:

```sh
cd /home/ec2-user/artifacts/glm53-dsa-numerics-20260908/smoke
GLM53_CAPTURE_DSA=1 GLM53_MATCHED_CONTROL=1 MODAL_PROFILE=modal-labs \
  modal run --detach --env glm-bringup smoke_lane.py \
  --dependency-call fc-01M211KQ0YFGMP0ZRACSSYSCNH
```

## Capturing the small shapes: kernel-level alternative passed

A separate probe, without changing the guarded model smoke, captured **all
90/90 valid layouts bitwise exactly**, including the small 1–80-token cases.
It passes an exact-live-token query subview and uses
`min(2052, ceil(actual_max_seq_len / 128) * 128)` as the captured context
representative. Results: `variant-result.json`. All rows have
`captured_attention=true`; every graph was replayed three times.

Run `20260908T175038.897209Z-dsa-variant-probe`, app
`ap-ul6ap0v1wcVL1gkAAaOhdS`, call `fc-01M2129D9YQVH22DWKVDPDEFHE`.
The 90-case guarded probe above consists of 25 captured cases and 65 preserved
eager cases; this alternative actually captures attention for all 90.

This establishes a kernel-level route around the eager cutoff. It does **not**
establish integrated-model parity for graph variants. Integration must select
variants from CPU live-token/context metadata before replay, preserve fixed
subview sizes per graph, and copy/zero unused padded output rows. The current
prefill runner's `ShapeKey` already supports variant labels, but prefill capture
and selection currently only use them for chunked-prefix variants. The public
FlashInfer MLA wrapper has no explicit launch-plan override. Exact-live-count
variants require more graphs; 17 representative context bounds reduce that
dimension, and actual-plan deduplication could further limit captures.
