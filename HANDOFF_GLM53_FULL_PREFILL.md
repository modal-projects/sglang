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

## Verified as of 2026-09-08 15:38 UTC

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

Full numerical/stress validation, real-activation parity, and integrated DSA
capture are still pending. No serving speedup or production-readiness claim.

## Detached runs

All results below are in Modal volume `glm53-kda-benchmarks` in this environment.
Each run has a `run.json` or `probe.json` plus logs. Model runs persist every
30 seconds. Raw fixtures are synthetic prompts, not customer requests.

| Run | App | Call |
| --- | --- | --- |
| Baseline real-activation collection | `ap-9OG8IanUISrJ40OVjtAMRB` | `fc-01M20SHV5J6CC1KYRYWAEWHA08` |
| Synthetic KDA probe (passed) | `ap-7qc1VpjZaaMEXACXqWs2LK` | `fc-01M20SR0Q4S473CTBQCRM3Z9TJ` |
| KDA model smoke | `ap-QetOu9UxD3z56uWjZWtHqS` | `fc-01M20SWQJD9AVCFC4C67NAHZZ5` |
| DSA probe (passed) | `ap-XIDN8Ou2bdWODPqqsIzE4p` | `fc-01M20T52K4NN2XJ2PG3Q5GDZ1F` |
| Real-activation KDA probe, waits for collection | `ap-dzoESBrOFEEiFVwHTqeHgX` | `fc-01M20TCYXY9TJ70PQCNTMYCTKX` |
| KDA+DSA model smoke, waits for KDA smoke pass | `ap-YYW95owu35oVFwupeG0l8I` | `fc-01M20TFYYMWPMGJF7Q5TDF0FEX` |

Run directories, in the same order:

1. `20260908T151758.049033Z-kda-graph-fixtures`
2. `20260908T152120.347071Z-kda-graph-probe`
3. `20260908T152354.819242Z-captured-kda-smoke`
4. `20260908T152828.251011Z-dsa-graph-probe`
5. `20260908T153246.643723Z-kda-graph-probe`
6. `20260908T153424.971011Z-captured-kda-dsa-smoke`

Profiles are in volume `glm53-torch-profiles`:
`20260908T152354.819242Z-captured-kda-smoke/text/`.
All-rank `inventory.json` and `release-smoke-breakable-TP-{rank}-EP-{rank}.trace.json.gz`
were committed to the volume after the profile check.

Harness source/archive: `glm53-kda-benchmarks/full-prefill-handoff-20260908/harness.tar.gz`.
The archive includes launch records, source hashes, scripts, recipe and overlays.
Locally: `/home/ec2-user/artifacts/glm53-full-prefill-20260908`.

The baseline image is `im-O356Uyz2cuFL5NvpeEepjW`. KDA smoke image:
`im-NyC7fXG9lJM8p0EI0OfIIR`. Combined smoke image:
`im-wPLaSbgSSa04XyxoBfxSX0`.

## Continue

1. Inspect the running KDA model smoke and real-activation probe. Do not treat
   finite outputs alone as numerical equivalence. Compare saved outputs,
   selected SSM/conv states and untouched-slot behavior.
2. Combined model smoke should confirm **12 segments**. Its CPU driver only
   launches the 4-B300 job after the first model smoke passes. Fix failures
   before benchmarking; all runs use retries=0 to avoid crash loops.
3. Compare serving against the unchanged BCG baseline on the same GPUs, with
   identical workload/seed and alternating order. Benchmark is not launched yet.
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
GLM53_CAPTURE_DSA=1 MODAL_PROFILE=modal-labs modal run --detach --env glm-bringup \
  smoke_lane.py --dependency-call KDA_SMOKE_CALL
```

`modal container exec` takes a profile and container ID, not `--env`; separate
the remote command with `--` before command options. Long first-bucket startup
was observed in Triton compilation/cache reads and DeepGEMM warmup; both model
lanes progressed. A Python stack was inspected to distinguish it from a stall.
