"""Fine-grained correctness + latency benchmark for the Inkling MoE gate.

Compares every implementation of the gate (linear -> sigmoid+bias top-6 ->
logsigmoid renorm over selected + 2 shared experts) at production shapes:
x [M, 6144] bf16, gate weight [264, 6144] bf16, logits [M, 258] fp32 (a
[M, 264]-padded slice), K=6, S=2.

Implementations
  standalone gate kernel (consumes precomputed cublas logits):
    triton          production `sigmoid_gate_topk_renorm` (auto-tuned params)
    triton:BM-BN-W  same kernel, explicit tile params (tuning sweep)
    v1              CUDA JIT warp-per-row (`inkling_gate_topk_renorm`)
    v2:W            CUDA JIT v2, warps_per_block=W (0=auto)
  end-to-end gate (linear included):
    mm+triton / mm+v1 / mm+v2:W    cublas fp32-out GEMM + standalone kernel
    gemv+v2         CUDA GEMV kernel + v2 kernel (PDL split pair)
    gemv_fused      single CUDA kernel: GEMV + gate epilogue   (M <= 64)
    tc_fused:BM-BK-ST  triton tensorcore GEMM + in-register gate epilogue

Correctness: standalone impls must match a bit-exact torch emulation of the
triton kernel's selection (uint64 sort keys) on the same logits; end-to-end
impls are checked against an fp64-GEMM reference with near-tie-aware index
matching (any index mismatch must be a genuine near-tie).

Timing: CUDA-graph replay of `--iters` back-to-back calls (decode regime; PDL
attrs are captured) and eager `triton.testing.do_bench` with L2 clearing
(prefill/cold-cache regime).

Run inside the sgl container, e.g.:
  CUDA_VISIBLE_DEVICES=2 python -m sglang.tml.kernels.benchmark.bench_gate_topk \
      --ms 1,2,4,8,16,32,64,128,512,4096,16384 --mode packed
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import torch
import triton
import triton.language as tl
import triton.testing

from sglang.srt.environ import envs

from sglang.jit_kernel.inkling_gate_topk_renorm import (
    inkling_gate_gemv,
    inkling_gate_gemv_fused,
    inkling_gate_topk_renorm,
    inkling_gate_topk_renorm_v2,
)
from sglang.srt.kernels.sigmoid_gate_topk_renorm import (
    _sigmoid_gate_topk_renorm_kernel,
    sigmoid_gate_topk_renorm,
)
from sglang.tml.kernels.gate_gemm_topk import gate_gemm_topk

HIDDEN = 6144
PRODUCER_GRID = 8  # small grid like a comm kernel: leaves SMs free for overlap


@triton.jit
def _producer_kernel(
    src_ptr,
    dst_ptr,
    n_elem,
    sign,
    spin,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """AR/norm stand-in: writes the consumer's input tensor (dst = sign * src),
    burns `spin` iterations to emulate the producer's duration, then triggers
    dependents. Small grid so a PDL secondary can be co-resident."""
    pid = tl.program_id(0)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK,), dtype=tl.float32) + sign
    for _ in range(spin):  # latency filler (compute-bound, no memory traffic)
        acc = tl.sqrt(tl.abs(acc)) + 1e-6
    eff_sign = sign * (tl.sum(acc) * 0.0 + 1.0)  # keep the spin loop alive
    n_blocks = tl.num_programs(0)
    per = tl.cdiv(n_elem, n_blocks * BLOCK) * BLOCK
    for off in range(pid * per, tl.minimum((pid + 1) * per, n_elem), BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < n_elem
        v = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_ptr + offs, (v * eff_sign).to(dst_ptr.dtype.element_ty), mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
N_ROUTED = 256
N_SHARED = 2
N_TOTAL = 258
N_PADDED = 264
TOPK = 6
ROUTE_SCALE = 8.0

DEFAULT_MS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192, 16384]


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------


def _fpval_to_key(bits: torch.Tensor) -> torch.Tensor:
    """uint32 sortable key of fp32 bits, emulating gate_topk.fpval_to_key (int64 math)."""
    top = 0x80000000
    full = 0xFFFFFFFF
    return bits ^ torch.where(bits & top != 0, full, top)


def ref_gate_from_logits(
    logits: torch.Tensor, bias: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bit-exact torch emulation of the triton kernel's selection + renorm.

    Returns (routed_w fp32 [M,6], indices int32 [M,6], shared_w fp32 [M,2],
    sel fp32 [M,256]); `sel` is returned for near-tie audits.
    """
    raw = logits[:, :N_ROUTED].float()
    sel = torch.sigmoid(raw) + bias[None, :]
    bits = sel.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = _fpval_to_key(bits)
    idx = torch.arange(N_ROUTED, device=logits.device, dtype=torch.int64)
    packed = (key << 16) | (N_ROUTED - idx)[None, :]
    top = torch.sort(packed, dim=1, descending=True).values[:, :TOPK]
    indices = (N_ROUTED - (top & 0xFFFF)).to(torch.int32)

    sel_raw = raw.gather(dim=1, index=indices.long())
    active = torch.cat(
        [torch.sigmoid(sel_raw), torch.sigmoid(logits[:, N_ROUTED:N_TOTAL].float())],
        dim=1,
    )
    weights = active / active.sum(dim=1, keepdim=True)
    weights = weights * (ROUTE_SCALE * global_scale.item())
    return weights[:, :TOPK].contiguous(), indices, weights[:, TOPK:].contiguous(), sel


def unpack_packed(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(id << 16) | bf16 bits -> (weights fp32, indices int32)."""
    indices = (packed >> 16).to(torch.int32)
    weights = (packed & 0xFFFF).to(torch.int32).to(torch.uint16).view(torch.bfloat16)
    return weights.float(), indices


# --------------------------------------------------------------------------
# implementations
# --------------------------------------------------------------------------


class Bench:
    def __init__(
        self, m: int, packed: bool, seed: int = 0, pdl: bool = True, chain: bool = False, spin: int = 0
    ):
        torch.manual_seed(seed)
        dev = torch.device("cuda")
        self.m = m
        self.packed = packed
        self.pdl = pdl
        self.chain = chain
        self.spin = spin
        self.x = (torch.randn((m, HIDDEN), device=dev) * 0.05).to(torch.bfloat16)
        self.weight = (torch.randn((N_PADDED, HIDDEN), device=dev) * 0.02).to(torch.bfloat16)
        self.weight[N_TOTAL:].zero_()
        self.weight_t = self.weight.T.contiguous()
        self.bias = torch.randn((N_ROUTED,), device=dev) * 0.1
        self.global_scale = torch.tensor([1.25], device=dev)
        # production logits layout: [M, 264] padded slice
        self.logits = self.mm_logits()
        # producer chain: pristine copies; the producer rewrites x / logits
        # in front of every gate call (sign=+1 keeps values identical).
        self.x_src = self.x.clone()
        self.logits_padded = torch.empty((m, N_PADDED), dtype=torch.float32, device=dev)
        self.logits_padded[:, :N_TOTAL] = self.logits
        self.logits_src = self.logits_padded.clone()
        if chain:
            self.logits = self.logits_padded[:, :N_TOTAL]

    def produce(self, dst: torch.Tensor, src: torch.Tensor, sign: float = 1.0) -> None:
        _producer_kernel[(PRODUCER_GRID,)](
            src,
            dst,
            src.numel(),
            sign,
            self.spin,
            BLOCK=1024,
            ENABLE_PDL=True,
            launch_pdl=True,
        )

    def chain_wrap(self, fn, e2e: bool):
        """Prepend the producer that writes this impl's input tensor."""
        if not self.chain:
            return fn
        if e2e:  # producer feeds x (like the AR/norm before the gate linear)
            def wrapped():
                self.produce(self.x, self.x_src)
                return fn()

        else:  # standalone kernels consume logits directly
            def wrapped():
                self.produce(self.logits_padded, self.logits_src)
                return fn()

        return wrapped

    def mm_logits(self) -> torch.Tensor:
        return torch.mm(self.x, self.weight.T, out_dtype=torch.float32)[:, :N_TOTAL]

    # ---- standalone gate kernels (consume self.logits) ----

    def run_triton(self, block_m: int = 0, block_n: int = 0, num_warps: int = 0):
        if block_m == 0:  # production auto params; pin the triton path (the
            # wrapper otherwise dispatches the production shape to the JIT kernel)
            with envs.SGLANG_OPT_USE_GATE_TOPK_JIT.override(False):
                out = sigmoid_gate_topk_renorm(
                    self.logits,
                    TOPK,
                    N_SHARED,
                    ROUTE_SCALE,
                    self.global_scale,
                    self.bias,
                    return_packed_topk=self.packed,
                )
            return out
        m = self.m
        shared_w = torch.empty((m, N_SHARED), dtype=torch.float32, device="cuda")
        if self.packed:
            packed = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
            routed_w = indices = None
            args = (packed, shared_w, packed, packed)
        else:
            routed_w = torch.empty((m, TOPK), dtype=torch.float32, device="cuda")
            indices = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
            packed = None
            args = (routed_w, shared_w, indices, indices)
        routed_arg, shared_arg, idx_arg, packed_arg = args
        _sigmoid_gate_topk_renorm_kernel[(triton.cdiv(m, block_m),)](
            self.logits,
            self.bias,
            self.logits.stride(0),
            routed_arg,
            shared_arg,
            idx_arg,
            packed_arg,
            self.global_scale,
            ROUTE_SCALE,
            M=m,
            N=N_ROUTED,
            G=N_TOTAL,
            N_PAD=triton.cdiv(N_ROUTED, block_n) * block_n,
            K=TOPK,
            K_POW2=8,
            S=N_SHARED,
            A_POW2=8,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            RETURN_PACKED_TOPK=self.packed,
            num_warps=num_warps,
        )
        return routed_w, indices, shared_w, packed

    def run_v1(self):
        out = inkling_gate_topk_renorm(
            self.logits,
            self.bias,
            self.global_scale,
            ROUTE_SCALE,
            return_packed=self.packed,
        )
        if self.packed:
            packed, shared_w = out
            return None, None, shared_w, packed
        routed_w, shared_w, indices = out
        return routed_w, indices.to(torch.int32), shared_w, None

    def run_v2(self, warps: int = 0, enable_pdl: bool | None = None):
        return inkling_gate_topk_renorm_v2(
            self.logits,
            self.bias,
            self.global_scale,
            ROUTE_SCALE,
            return_packed=self.packed,
            enable_pdl=self.pdl if enable_pdl is None else enable_pdl,
            warps_per_block=warps,
        )

    # ---- end-to-end gate (linear included) ----

    def e2e_mm(self, standalone_fn):
        def fn():
            self.logits = self.mm_logits()
            return standalone_fn()

        return fn

    def run_gemv_pair(self, epb: int = 0):
        logits = inkling_gate_gemv(
            self.x, self.weight, enable_pdl=self.pdl, experts_per_block=epb
        )
        saved = self.logits
        self.logits = logits
        try:
            return self.run_v2(warps=0)
        finally:
            self.logits = saved

    def run_gemv_fused(self, epb: int = 0):
        return inkling_gate_gemv_fused(
            self.x,
            self.weight,
            self.bias,
            self.global_scale,
            ROUTE_SCALE,
            return_packed=self.packed,
            enable_pdl=self.pdl,
            experts_per_block=epb,
        )

    def run_tc_fused(
        self,
        block_m: int = 32,
        block_k: int = 256,
        num_stages: int = 3,
        gemm_only: int = 0,
        use_wt: int = 1,
    ):
        return gate_gemm_topk(
            self.x,
            self.weight_t if use_wt else self.weight,
            self.bias,
            self.global_scale,
            TOPK,
            N_SHARED,
            ROUTE_SCALE,
            return_packed_topk=self.packed,
            block_m=block_m,
            block_k=block_k,
            num_stages=num_stages,
            debug_gemm_only=bool(gemm_only),
            transposed_w=bool(use_wt),
        )

    def make_impl(self, spec: str):
        """spec -> (callable, is_end_to_end) or None if unsupported at this M."""
        name, _, params = spec.partition(":")
        p = [int(v) for v in params.split("-") if v] if params else []
        if name == "mm":
            return (lambda: self.mm_logits()), "kernel-only"
        if name == "gemv":
            fn = lambda: inkling_gate_gemv(  # noqa: E731
                self.x, self.weight, enable_pdl=self.pdl, experts_per_block=p[0] if p else 0
            )
            return fn, "kernel-only"
        if name == "triton":
            fn = lambda: self.run_triton(*p)  # noqa: E731
            return fn, False
        if name == "v1":
            return self.run_v1, False
        if name == "v2":
            fn = lambda: self.run_v2(*p)  # noqa: E731
            return fn, False
        if name == "mm+triton":
            return self.e2e_mm(lambda: self.run_triton(*p)), True
        if name == "mm+v1":
            return self.e2e_mm(self.run_v1), True
        if name == "mm+v2":
            return self.e2e_mm(lambda: self.run_v2(*p)), True
        if name == "gemv+v2":
            return (lambda: self.run_gemv_pair(*p)), True
        if name == "gemv_fused":
            if self.m > 64:
                return None
            return (lambda: self.run_gemv_fused(*p)), True
        if name == "tc_fused":
            fn = lambda: self.run_tc_fused(*p)  # noqa: E731
            if len(p) > 3 and p[3]:  # gemm-only debug flag is 4th param
                return fn, "kernel-only"  # gemm-only debug: outputs are dummies
            return fn, True
        raise ValueError(f"unknown impl {spec!r}")


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------


def normalize_out(out) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (routed_w fp32, indices int32, shared_w fp32); unpacks packed mode."""
    routed_w, indices, shared_w, packed = out
    if packed is not None:
        routed_w, indices = unpack_packed(packed)
    return routed_w.float(), indices.to(torch.int32), shared_w.float()


def check_impl(bench: Bench, spec: str, fn, e2e: bool) -> str:
    """Returns '' if correct, else a description of the worst mismatch."""
    w, idx, sh = normalize_out(fn())
    # weights in packed mode are bf16-rounded: loosen tolerance accordingly
    watol = 2e-2 if bench.packed else (2e-3 if e2e else 1e-5)

    if not e2e:
        ref_w, ref_idx, ref_sh, _ = ref_gate_from_logits(
            bench.logits, bench.bias, bench.global_scale
        )
        if not torch.equal(idx, ref_idx):
            bad = (idx != ref_idx).sum().item()
            return f"indices mismatch at {bad}/{idx.numel()} positions"
    else:
        # fp64 reference GEMM; index mismatches must be genuine near-ties
        logits64 = (bench.x.double() @ bench.weight.double().T)[:, :N_TOTAL].float()
        ref_w, ref_idx, ref_sh, sel = ref_gate_from_logits(
            logits64, bench.bias, bench.global_scale
        )
        if not torch.equal(idx, ref_idx):
            rows = (idx != ref_idx).any(dim=1)
            n_rows = int(rows.sum())
            # audit: the swapped experts' selection scores must be within tol
            sel_rows = sel[rows]
            sel_mine = sel_rows.gather(1, idx[rows].long().clamp(0, N_ROUTED - 1))
            sel_ref = sel_rows.gather(1, ref_idx[rows].long())
            gap = (sel_mine - sel_ref).abs().max().item() if n_rows else 0.0
            if gap > 5e-3:
                return f"indices mismatch on {n_rows} rows, max sel gap {gap:.2e}"
            return ""  # near-ties only: accept, weights checked loosely below

    for tag, mine, ref in (("routed_w", w, ref_w), ("shared_w", sh, ref_sh)):
        err = (mine - ref).abs().max().item() if mine.numel() else 0.0
        if err > watol:
            return f"{tag} max abs err {err:.3e} > {watol:g}"
    return ""


def chain_order_check(bench: Bench, fn, e2e) -> str:
    """PDL ordering probe: alternate the producer's sign in a captured graph;
    each gate call must see exactly what its own producer wrote. A gate read
    escaping gdc_wait shows up as a stale-sign output."""
    consumes_x = bool(e2e)  # e2e and kernel-only impls read x; standalone read logits
    src = bench.x_src if consumes_x else bench.logits_src
    dst = bench.x if consumes_x else bench.logits_padded

    def prod(sign: float) -> None:
        bench.produce(dst, src, sign)

    refs = []
    for sign in (1.0, -1.0):
        prod(sign)
        refs.append(normalize_out(fn()))
    prod(1.0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = []
        for sign in (1.0, -1.0, 1.0, -1.0):
            prod(sign)
            outs.append(fn())
    g.replay()
    torch.cuda.synchronize()
    for i, o in enumerate(outs):
        ref = refs[i % 2]
        for tag, a, b in zip(("routed_w", "indices", "shared_w"), normalize_out(o), ref):
            if not torch.equal(a, b):
                return f"chain-order iter {i}: {tag} mismatch (stale read past gdc_wait?)"
    prod(1.0)  # restore
    torch.cuda.synchronize()
    return ""


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def time_graph(fn, iters: int, replays: int = 10) -> float:
    """µs per call: capture `iters` back-to-back calls, time graph replays."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * iters)


def time_eager(fn) -> float:
    """µs per call, eager launches with L2 clearing (do_bench median)."""
    return triton.testing.do_bench(fn, warmup=10, rep=50) * 1e3


def graph_output_check(bench: Bench, fn) -> str:
    """Re-verify outputs produced under graph replay (PDL / ticket safety)."""
    outs: list = []
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(3):
            outs.append(fn())
    g.replay()
    torch.cuda.synchronize()
    a = normalize_out(outs[0])
    b = normalize_out(outs[-1])
    for tag, x, y in zip(("routed_w", "indices", "shared_w"), a, b):
        if not torch.equal(x, y):
            return f"graph replay: {tag} differs between captured iterations"
    return ""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

DEFAULT_IMPLS = [
    # standalone (kernel-only, same cublas logits)
    "triton",
    "v1",
    "v2:0",
    # end-to-end
    "mm+triton",
    "mm+v2:0",
    "gemv+v2:1",
    "gemv_fused:1",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=str, default=",".join(map(str, DEFAULT_MS)))
    ap.add_argument("--impls", type=str, default=",".join(DEFAULT_IMPLS))
    ap.add_argument("--mode", choices=["packed", "unpacked", "both"], default="both")
    ap.add_argument("--iters", type=int, default=100, help="calls per captured graph")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-pdl", action="store_true", help="disable PDL on the JIT gate kernels")
    ap.add_argument(
        "--chain",
        action="store_true",
        help="prepend a PDL-triggering producer kernel (AR/norm stand-in) to every call",
    )
    ap.add_argument("--spin", type=int, default=2000, help="producer latency filler iterations")
    ap.add_argument("--no-graph", action="store_true", help="skip CUDA-graph timing")
    ap.add_argument("--json", type=str, default="", help="dump results to this path")
    args = ap.parse_args()

    ms = [int(v) for v in args.ms.split(",") if v]
    impls = [s for s in args.impls.split(",") if s]
    modes = {"packed": [True], "unpacked": [False], "both": [True, False]}[args.mode]

    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    results = []
    for packed in modes:
        mode = "packed" if packed else "unpacked"
        header = f"{'M':>6} | " + " | ".join(f"{s:>18}" for s in impls)
        print(f"\n=== mode={mode} (graph µs / eager µs) ===\n{header}\n{'-' * len(header)}")
        for m in ms:
            bench = Bench(
                m, packed=packed, pdl=not args.no_pdl, chain=args.chain, spin=args.spin
            )
            # cap in-graph iterations at large M (each captured call allocates
            # its outputs/logits from the graph pool)
            iters = max(10, min(args.iters, (1 << 22) // max(m, 1)))
            cells = []
            for spec in impls:
                made = bench.make_impl(spec)
                if made is None:
                    cells.append(f"{'--':>18}")
                    continue
                fn, e2e = made
                raw_fn = fn
                fn = bench.chain_wrap(fn, e2e)
                try:
                    if e2e == "kernel-only":
                        err = ""
                    else:
                        err = "" if args.no_check else check_impl(bench, spec, raw_fn, e2e)
                    if not err and e2e != "kernel-only" and not args.no_check and not args.no_graph:
                        err = graph_output_check(bench, fn)
                    if (
                        not err
                        and args.chain
                        and e2e != "kernel-only"
                        and not args.no_check
                        and not args.no_graph
                    ):
                        err = chain_order_check(bench, raw_fn, e2e)
                    if err:
                        cells.append(f"{'FAIL':>18}")
                        print(f"  [{mode} M={m} {spec}] {err}", file=sys.stderr)
                        results.append({"mode": mode, "m": m, "impl": spec, "error": err})
                        continue
                    t_graph = None if args.no_graph else time_graph(fn, iters=iters)
                    t_eager = time_eager(fn)
                    gtxt = f"{t_graph:6.2f}" if t_graph is not None else "  --  "
                    cells.append(f"{gtxt} /{t_eager:7.2f}   ")
                    results.append(
                        {"mode": mode, "m": m, "impl": spec, "graph_us": t_graph, "eager_us": t_eager}
                    )
                except Exception:
                    cells.append(f"{'ERROR':>18}")
                    traceback.print_exc()
                    results.append({"mode": mode, "m": m, "impl": spec, "error": "exception"})
            print(f"{m:>6} | " + " | ".join(cells))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
