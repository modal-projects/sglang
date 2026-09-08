# GLM-5.3 Flash prefill capture experiment

Workspace/environment: **modal-labs / glm-bringup**. Do not change production.
Branch: `willhu/glm53-full-prefill`, based on release/glm53
`b418d1c0387a0f4deb83796c2c0f217feb0be103`.

The experiment retains the production recipe, Triton KDA, TRTLLM DSA, FP8 KV,
TP4/EP4, DeepGEMM MoE, and DFlash2 epoch-2 block8. Both new flags default off:
`SGLANG_KDA_PREFILL_CUDA_GRAPH` and `SGLANG_DSA_PREFILL_CUDA_GRAPH`.
The runner is still BCG; the generic `full` runner is not implemented here.

## Implemented

- `e21e303d9c`: capture KDA using stable ragged chunk maps, packed convolution
  chunks, masked state snapshots, and zeroed output padding. Existing Triton
  arithmetic is retained; fixed capacities can change its small-grid fusion
  choice, so exact equality is not claimed for every shape.
- `10f049bbc6`: independent no-RoPE MLA KV-write fix. Honor the reserved skip
  slot and DCP ownership, matching the existing RoPE scatter contract.
- `78220375f6`: capture no-RoPE FP8 TRTLLM DSA attention and its preceding BMM.
  The sparse-length bound replaces the original-context scalar for this
  attention call. The KPool indexer remains eager.
- `e44be6b92d`: preserve KDA's live-chunk fusion choice under graph replay.
  Both variants share output buffers, with a device-side guard selecting the
  original small-grid behavior. This is a candidate numerical fix, not yet
  validated on the integrated model.

## Results as of 2026-09-08 15:48 UTC

- 24 synthetic KDA replay cases passed on B300, reusing one 4096-token graph
  while changing sequence lengths, slot mappings, prefix flags and snapshots.
  Includes 32 requests requiring 95 chunks, logical padding, and slot reuse.
  Output/state exact in 18/24; maximum absolute differences 1.52587890625e-5
  and 6.103515625e-5. Convolution state exact in all 24. Padding zero.
- Six isolated DSA attention replay cases exact; 14 reserved-slot/DCP scatter
  tests exact. Captured max_seq_len=2052 matched the eager 1048576 scalar.
- Integrated KDA-only model started successfully. Profiles confirm **23 graph
  launches instead of 57**, on all four ranks for all 14 eligible prefills.
  Includes 2049/3073/4095/4096 tokens and batch sizes 2/4/8/16/32.
  4097-token prefill correctly launches zero graphs.
- Repository pre-commit hooks passed for all modified Python files.
- The first KDA model smoke completed 510 finite-output checks, the 33-case
  runtime stress suite (including long context and multimodal inputs), and
  all-rank multimodal replay profiles. **This does not establish parity.**
- Comparing its 510 numerical cases to the previous published-recipe smoke
  found **508/510 identical greedy outputs**, maximum selected teacher-forced
  log-probability difference **1.665709**. The 31- and 32-token cases differed.
  This comparison used different GPU allocations and startup seeds. The
  differences cluster around KDA's small-grid fusion choice, which padded
  capacities changed in the first implementation. The new revision restores
  selection using the live chunk count.

The original integrated KDA+DSA job was stopped before allocating GPUs after
this parity failure. Real-activation and same-GPU serving parity are pending.
No serving speedup or production-readiness claim. Do not benchmark or deploy
the first KDA implementation as a numerically validated optimization.

## Detached runs

All results below are in Modal volume `glm53-kda-benchmarks` in this environment.
Each run has a `run.json` or `probe.json` plus logs. Model runs persist every
30 seconds. Raw fixtures are synthetic prompts, not customer requests.

| Run | App | Call |
| --- | --- | --- |
| Baseline real-activation collection | `ap-9OG8IanUISrJ40OVjtAMRB` | `fc-01M20SHV5J6CC1KYRYWAEWHA08` |
| Synthetic KDA probe (passed) | `ap-7qc1VpjZaaMEXACXqWs2LK` | `fc-01M20SR0Q4S473CTBQCRM3Z9TJ` |
| KDA-v1 finite/stress smoke (passed; parity failed separately) | `ap-QetOu9UxD3z56uWjZWtHqS` | `fc-01M20SWQJD9AVCFC4C67NAHZZ5` |
| DSA probe (passed) | `ap-XIDN8Ou2bdWODPqqsIzE4p` | `fc-01M20T52K4NN2XJ2PG3Q5GDZ1F` |
| Real-activation KDA probe, waits for collection | `ap-dzoESBrOFEEiFVwHTqeHgX` | `fc-01M20TCYXY9TJ70PQCNTMYCTKX` |
| KDA+DSA-v1 model smoke (**stopped**) | `ap-YYW95owu35oVFwupeG0l8I` | `fc-01M20TFYYMWPMGJF7Q5TDF0FEX` |
| Revised KDA real-activation probe, waits for collection | `ap-yLT6SmGETd0sBJeH9yLeWX` | `fc-01M20V2C27S209031TEHKPY6QH` |
| Revised KDA same-GPU serving validation, waits for revised probe | `ap-du2uZvplfcXXMKfcoARLRB` | `fc-01M20V886H51K1YFV6E5WF5S38` |
| Revised KDA+DSA same-GPU validation, waits for matched KDA parity | `ap-lCBZ6bu5EektbkfeNtMoQp` | `fc-01M20VD1255WFG6E1AWF7A8NTF` |
| Throughput/interactivity pilot, waits for matched combined parity | `ap-t330ncV37nkzRjuIELN6WT` | `fc-01M20VE798XMGDKKFDXZN80YF2` |

Run directories, in the same order:

1. `20260908T151758.049033Z-kda-graph-fixtures`
2. `20260908T152120.347071Z-kda-graph-probe`
3. `20260908T152354.819242Z-captured-kda-smoke`
4. `20260908T152828.251011Z-dsa-graph-probe`
5. `20260908T153246.643723Z-kda-graph-probe`
6. `20260908T153424.971011Z-captured-kda-dsa-smoke`
7. `20260908T154428.221954Z-kda-graph-probe`
8. `20260908T154740.869584Z-captured-kda-smoke`
9. `20260908T155017.403236Z-captured-kda-dsa-smoke`
10. `20260908T155056.541792Z-captured-attention-curve`

The revised probe adds 31/32/64-token cases (30 synthetic replays total), then
replays saved real model activations three times each. The original real probe
remains as a control. The revised integrated lane runs candidate and baseline
sequentially on the same four GPU UUIDs with seed 479309393. It requires all
510 greedy sequences to match and selected teacher log probabilities to differ
by at most 0.05, and collects profiles of both arms. A finite-only smoke pass
must never be substituted for this parity gate.

The combined lane and curve have been queued as CPU driver jobs; they allocate
GPUs only if their correctness dependencies pass. The curve requires an
explicit matched-parity result, not just smoke status. It uses the same four
GPUs in ABBA order, 64 agentic-v2 prompts per point, concurrency 1/2/4/8/16/32,
two repetitions, and records profiles for both implementations. This is a
pilot, not a production-speed claim.

Profiles are in volume `glm53-torch-profiles`:
`20260908T152354.819242Z-captured-kda-smoke/text/`.
All-rank `inventory.json` and `release-smoke-breakable-TP-{rank}-EP-{rank}.trace.json.gz`
were committed to the volume after the profile check.

Harness archive: `glm53-kda-benchmarks/full-prefill-handoff-20260908/harness.tar.gz`.
The archive includes launch records, source hashes, scripts, recipe and overlays.
Its SHA-256 and per-file hashes are in the adjacent `manifest.json`.
Locally: `/home/ec2-user/artifacts/glm53-full-prefill-20260908`.

The baseline image is `im-O356Uyz2cuFL5NvpeEepjW`. KDA smoke image:
`im-NyC7fXG9lJM8p0EI0OfIIR`. Combined smoke image:
`im-wPLaSbgSSa04XyxoBfxSX0`.
Revised KDA same-GPU validation image: `im-ItRRmFe8dZQ0g2MAXnnzID`.
Revised combined validation/benchmark image: `im-EqRdVrWCHlamsDjtLz66pC`.

## Continue

1. Inspect the revised KDA model smoke and real-activation probe. Do not treat
   finite outputs alone as numerical equivalence. Compare saved outputs,
   selected SSM/conv states and untouched-slot behavior.
2. The queued combined model smoke should confirm **12 segments**, with both
   capture flags enabled and `GLM53_MATCHED_CONTROL=1`. Fix failures before
   benchmarking; all runs use retries=0 to avoid crash loops.
3. Inspect the queued serving curve after matched parity passes. It compares
   the unchanged BCG baseline on the same GPUs, with identical workload/seed
   and alternating order. No GPU benchmark points have completed yet.
4. KPool is the remaining 11 breaks. Its plan has capturable masked compression
   and tail kernels, but gathered history/logits need bounded scratch and a
   deliberate history-bucketing or paged/tiled design. Do not reserve the
   concatenation of every request's maximum history blindly. See KPOOL_NEXT.md
   in the harness archive for the concrete fields, bounds and validation needs.

Typical commands from the extracted harness:

```sh
MODAL_PROFILE=modal-labs modal run --detach --env glm-bringup collect_lane.py
MODAL_PROFILE=modal-labs modal run --detach --env glm-bringup probe_lane.py \
  --fixture-run RUN_ID --collection-call COLLECTION_CALL
GLM53_MATCHED_CONTROL=1 MODAL_PROFILE=modal-labs modal run --detach --env glm-bringup \
  smoke_lane.py --dependency-call REVISED_REAL_PROBE_CALL
GLM53_CAPTURE_DSA=1 GLM53_MATCHED_CONTROL=1 MODAL_PROFILE=modal-labs \
  modal run --detach --env glm-bringup smoke_lane.py --dependency-call MATCHED_KDA_SMOKE_CALL
```

`modal container exec` takes a profile and container ID, not `--env`; separate
the remote command with `--` before command options. Long first-bucket startup
was observed in Triton compilation/cache reads and DeepGEMM warmup; both model
lanes progressed. A Python stack was inspected to distinguish it from a stall.

Independent revised synthetic probe (does not wait for collection):
`ap-nVKvS8qYuuVstrRPKP2Y9c` / `fc-01M20VJ9QWQB975NNJ205C4CKQ`,
run `20260908T155310.127061Z-kda-graph-probe`; record
`launch-probe-synthetic-v2.json`.
