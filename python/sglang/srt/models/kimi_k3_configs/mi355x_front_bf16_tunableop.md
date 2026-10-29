# MI355X front GEMM TunableOp table: findings

`mi355x_front_bf16_tunableop.csv` holds hipBLASLt/rocBLAS picks for the merged K3 MoE front GEMM ([shared gate_up | router | latent down], N=6016, K=7168) at the M values where they win. It loads lookup-only with `SGLANG_ROCM_K3_FRONT_GEMM_TABLE=1`. Other M values keep the default pick.

## Speed

Default vs table graphs were captured in the same process and replayed alternately, median of 15, ms per 92 layers on MI355X:

| M | 8 | 40 | 48 | 56 | 64 | 72 | 88 | 104 | 112 | 120 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 1.937 | 2.382 | 2.261 | 2.847 | 2.767 | 2.656 | 3.329 | 3.019 | 3.034 | 4.593 | 4.521 |
| table | 1.762 | 2.231 | 2.137 | 2.476 | 2.510 | 2.361 | 2.460 | 2.604 | 2.550 | 2.604 | 2.485 |

M=16, 24, 32, 80 and 96 were neutral or slower, so they are not in the table.

## Router check on real activations

The inputs were eager front inputs dumped from a serving rank 0: K3 Instinct, BF16 dense, long-context prefill, about 4096 rows per layer. They were run through the real merged front weight and `e_score_correction_bias`, comparing top-16 sigmoid+bias selection between the default kernel and the table. M=16/32 are controls; both paths use the same kernel there.

Each cell is set flips / rank-order flips (order flips include set flips), default vs table:

| M | L3 | L30 | L60 | L91 |
|---|---|---|---|---|
| 8 | 0 / 0 | 1 / 2 | 0 / 0 | 0 / 0 |
| 16 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 32 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 40 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |
| 48 | 0 / 1 | 1 / 2 | 0 / 0 | 0 / 0 |
| 56 | 0 / 1 | 1 / 2 | 0 / 1 | 0 / 0 |
| 64 | 0 / 1 | 1 / 2 | 0 / 1 | 0 / 0 |
| 72 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |
| 88 | 0 / 0 | 1 / 2 | 0 / 1 | 0 / 0 |
| 104 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |
| 112 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| 120 | 0 / 1 | 0 / 1 | 0 / 0 | 0 / 0 |
| 128 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |

Total: 5 set flips and 23 order flips (including those 5) over 211904 token rows. All 5 set flips are the same token in layer 30 (row 3942). Expert 122 sits on the 16th/17th boundary, 4e-4 above expert 375 in FP32 score. The two kernels round its logit to neighbouring bf16 values: -1.3047 vs -1.2969, against -1.3008 in FP32.

Top-16 set mismatches vs an FP32 reference for layer 30. The other layers were not measured against FP32 before the dump was discarded.

| M | default | table | rows |
|---|---|---|---|
| 8 | 859 | 858 | 4096 |
| 48 | 852 | 853 | 4080 |
| 64 | 858 | 859 | 4096 |
| 88 | 849 | 850 | 4048 |

With bf16 router logits, the default path already differs from FP32 on about 21% of tokens, and it changes kernel with M. The table moves that count by ±1 row in either direction, so the flip rate vs FP32 shows no systematic shift.

Decision: accepted at the bf16 rounding floor, gated end to end on GSM8K and accept.

## Numerics

- Max abs error vs FP32 matches the default at every M.
- Not bit-exact vs the default at tuned M, because the K split / reduction order differs.
- Graph replay matches eager.
