"""Per-kernel bench: fused gate top-k + logsigmoid renorm (production route).

Compares impls of the MoE gate sigmoid+bias path (select top-k by
sigmoid(routed logit)+gate_bias, then logsigmoid-renorm over top-k+shared on the
RAW logits):
  fused -> sigmoid_gate_topk_renorm (single Triton launch)
  fused_packed -> same kernel, return_packed_topk=True ((id<<16)|bf16 w, one store)
  jit_cuda -> inkling_gate_topk_renorm (shape-specialized CUDA JIT kernel)
  chain -> sigmoid+bias glue, gate_topk, renorm_topk_logits_scaled (today)
  torch -> sigmoid+bias + topk + renorm_torch

Logits are fp32 (linear_with_fp32_out -> sigmoid); gate bias fp32. topk=6,
256 routed + 2 shared. Each config prints a PASS/FAIL line on the (routed,
shared) weights vs torch before timing.

Run:  python benchmark/tml/fusion/bench_sigmoid_gate_topk_renorm.py


B200
[correctness] sigmoid_gate_topk_renorm t=1 [1,258] fused: PASS (abs_max=2.384e-07)
[correctness] sigmoid_gate_topk_renorm t=64 [64,258] fused: PASS (abs_max=4.768e-07)
[correctness] sigmoid_gate_topk_renorm t=8192 [8192,258] fused: PASS (abs_max=7.153e-07)
[correctness] sigmoid_gate_topk_renorm t=4096 [4096,258] fused: PASS (abs_max=5.960e-07)
[correctness] sigmoid_gate_topk_renorm t=16384 [16384,258] fused: PASS (abs_max=7.153e-07)
[correctness] sigmoid_gate_topk_renorm t=32768 [32768,258] fused: PASS (abs_max=7.153e-07)
[correctness] sigmoid_gate_topk_renorm t=1 [1,258] fused_packed: PASS (abs_max=3.310e-03)
[correctness] sigmoid_gate_topk_renorm t=64 [64,258] fused_packed: PASS (abs_max=5.612e-03)
[correctness] sigmoid_gate_topk_renorm t=8192 [8192,258] fused_packed: PASS (abs_max=7.690e-03)
[correctness] sigmoid_gate_topk_renorm t=4096 [4096,258] fused_packed: PASS (abs_max=7.801e-03)
[correctness] sigmoid_gate_topk_renorm t=16384 [16384,258] fused_packed: PASS (abs_max=7.777e-03)
[correctness] sigmoid_gate_topk_renorm t=32768 [32768,258] fused_packed: PASS (abs_max=7.775e-03)
[correctness] sigmoid_gate_topk_renorm t=1 [1,258] jit_cuda: PASS (abs_max=1.192e-07)
[correctness] sigmoid_gate_topk_renorm t=64 [64,258] jit_cuda: PASS (abs_max=4.768e-07)
[correctness] sigmoid_gate_topk_renorm t=8192 [8192,258] jit_cuda: PASS (abs_max=7.153e-07)
[correctness] sigmoid_gate_topk_renorm t=4096 [4096,258] jit_cuda: PASS (abs_max=5.960e-07)
[correctness] sigmoid_gate_topk_renorm t=16384 [16384,258] jit_cuda: PASS (abs_max=5.960e-07)
[correctness] sigmoid_gate_topk_renorm t=32768 [32768,258] jit_cuda: PASS (abs_max=7.153e-07)
[correctness] sigmoid_gate_topk_renorm t=1 [1,258] jit_cuda_packed: PASS (abs_max=3.028e-03)
[correctness] sigmoid_gate_topk_renorm t=64 [64,258] jit_cuda_packed: PASS (abs_max=3.904e-03)
[correctness] sigmoid_gate_topk_renorm t=8192 [8192,258] jit_cuda_packed: PASS (abs_max=7.557e-03)
[correctness] sigmoid_gate_topk_renorm t=4096 [4096,258] jit_cuda_packed: PASS (abs_max=6.545e-03)
[correctness] sigmoid_gate_topk_renorm t=16384 [16384,258] jit_cuda_packed: PASS (abs_max=7.770e-03)
[correctness] sigmoid_gate_topk_renorm t=32768 [32768,258] jit_cuda_packed: PASS (abs_max=7.804e-03)
[correctness] sigmoid_gate_topk_renorm t=1 [1,258] chain: PASS (abs_max=0.000e+00)
[correctness] sigmoid_gate_topk_renorm t=64 [64,258] chain: PASS (abs_max=3.576e-07)
[correctness] sigmoid_gate_topk_renorm t=8192 [8192,258] chain: PASS (abs_max=5.960e-07)
[correctness] sigmoid_gate_topk_renorm t=4096 [4096,258] chain: PASS (abs_max=5.960e-07)
[correctness] sigmoid_gate_topk_renorm t=16384 [16384,258] chain: PASS (abs_max=5.960e-07)
[correctness] sigmoid_gate_topk_renorm t=32768 [32768,258] chain: PASS (abs_max=5.960e-07)
[correctness] sigmoid_gate_topk_renorm t=1 [1,258] torch: PASS (abs_max=0.000e+00)
[correctness] sigmoid_gate_topk_renorm t=64 [64,258] torch: PASS (abs_max=0.000e+00)
[correctness] sigmoid_gate_topk_renorm t=8192 [8192,258] torch: PASS (abs_max=0.000e+00)
[correctness] sigmoid_gate_topk_renorm t=4096 [4096,258] torch: PASS (abs_max=0.000e+00)
[correctness] sigmoid_gate_topk_renorm t=16384 [16384,258] torch: PASS (abs_max=0.000e+00)
[correctness] sigmoid_gate_topk_renorm t=32768 [32768,258] torch: PASS (abs_max=0.000e+00)
============================================================================================================================================================================================================================
          t |       fused(us)  fused_packed(us)   jit_cuda(us)  jit_cuda_packed(us)      chain(us)      torch(us) |     fused(GB/s)  fused_packed(GB/s)  jit_cuda(GB/s)  jit_cuda_packed(GB/s)    chain(GB/s)    torch(GB/s)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0         1 |          3.7888            3.7901         3.8301               3.8275        18.7789        41.6563 |            0.26                0.26            0.26                   0.26           0.05           0.02
1        64 |          3.9731            3.9731         3.9328               3.9507        22.0890        47.9322 |           15.96               15.84           16.13                  15.93           2.87           1.32
2      8192 |         18.7712           18.7724        11.7760              11.7996        33.8550       154.4952 |          432.45              429.17          689.34                 682.79         239.78          52.54
3      4096 |         15.0730           15.1344         7.7213               7.7008        26.1526       103.8640 |          269.28              266.17          525.67                 523.10         155.20          39.08
4     16384 |         31.3522           31.4105        20.0885              20.0554        52.8812       261.2530 |          517.84              512.99          808.19                 803.44         307.02          62.14
5     32768 |         56.6418           56.7494        37.2541              37.2245        89.9365       464.9256 |          573.26              567.87          871.60                 865.73         361.04          69.84
============================================================================================================================================================================================================================
"""

import os
import sys

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.inkling_gate_topk_renorm import inkling_gate_topk_renorm
from sglang.srt.kernels.gate_topk import gate_topk
from sglang.srt.kernels.sigmoid_gate_topk_renorm import sigmoid_gate_topk_renorm
from sglang.srt.models.inkling_common.moe import renorm_topk_logits_scaled

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_refs import renorm_torch, report
from inkling_config import GATE_EXPERTS, N_SHARED, ROUTE_SCALE, TOKENS, TOPK

DEV = "cuda"
N_ROUTED = GATE_EXPERTS - N_SHARED  # 256


def _torch(logits, bias, gscale):
    sel = logits.sigmoid()[:, :N_ROUTED] + bias
    idx = sel.topk(TOPK, dim=-1).indices
    return renorm_torch(logits, idx, N_SHARED, ROUTE_SCALE, gscale, TOPK)


def _chain(logits, bias, gscale):
    sel = logits.sigmoid()[:, :N_ROUTED] + bias  # the add materializes contiguous
    _, idx = gate_topk(sel, TOPK)
    return renorm_topk_logits_scaled(logits, idx, N_SHARED, ROUTE_SCALE, gscale)


def _torch_ids_w(logits, bias, gscale):
    """Reference (expert_id fp32, routed_weight fp32) for the packed path."""
    sel = logits.sigmoid()[:, :N_ROUTED] + bias
    idx = sel.topk(TOPK, dim=-1).indices.to(torch.int32)
    routed_w, _ = renorm_torch(logits, idx, N_SHARED, ROUTE_SCALE, gscale, TOPK)
    return idx.float(), routed_w


def _unpack(packed):
    """quant.py-layout packed int32 -> (expert_id fp32, bf16-weight fp32). Decoding
    with this layout also asserts the kernel matches the quant.py pack -- a wrong
    layout yields garbage ids/weights that fail the compare."""
    ids = (packed >> 16).float()
    w = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
    return ids, w


@marker.parametrize("t", TOKENS + [4096, 16384, 32768])
@marker.benchmark(
    "impl", ["fused", "fused_packed", "jit_cuda", "jit_cuda_packed", "chain", "torch"]
)
def bench_sigmoid_gate_topk_renorm(t: int, impl: str):
    logits = torch.randn(t, GATE_EXPERTS, dtype=torch.float32, device=DEV).contiguous()
    bias = torch.randn(N_ROUTED, dtype=torch.float32, device=DEV)
    gscale = torch.ones(1, device=DEV, dtype=torch.float32)
    if impl == "fused":
        # returns (routed_w, indices, shared_w, packed); [0::2] -> (routed_w, shared_w)
        fn = lambda z: sigmoid_gate_topk_renorm(
            z, TOPK, N_SHARED, ROUTE_SCALE, gscale, bias
        )[0::2]
    elif impl == "fused_packed":
        fn = lambda z: sigmoid_gate_topk_renorm(
            z, TOPK, N_SHARED, ROUTE_SCALE, gscale, bias, return_packed_topk=True
        )[3]
    elif impl == "jit_cuda":
        fn = lambda z: inkling_gate_topk_renorm(z, bias, gscale, ROUTE_SCALE)[:2]
    elif impl == "jit_cuda_packed":
        fn = lambda z: inkling_gate_topk_renorm(
            z, bias, gscale, ROUTE_SCALE, return_packed=True
        )[0]
    elif impl == "chain":
        fn = lambda z: _chain(z, bias, gscale)
    else:
        fn = lambda z: _torch(z, bias, gscale)
    out = fn(logits)
    tag = f"sigmoid_gate_topk_renorm t={t} [{t},{GATE_EXPERTS}] {impl}"
    if impl in ("fused_packed", "jit_cuda_packed"):
        report(tag, _unpack(out), _torch_ids_w(logits, bias, gscale))
    else:
        report(tag, out, _torch(logits, bias, gscale))
    return marker.do_bench(fn, input_args=(logits,))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    bench_sigmoid_gate_topk_renorm.run()
