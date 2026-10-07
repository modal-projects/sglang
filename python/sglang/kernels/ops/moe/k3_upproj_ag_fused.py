# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ROCm decode: column-shard up_proj GEMM + all-gather + add3 in ONE
launch per MoE layer. PROTOTYPE, NOT WIRED INTO THE MODEL: correct and
bit-exact, but slower than the hipBLASLt + allgather_lastdim_add pair at
M >= 8 on MI355X (M=8: 15.4-15.8 vs 12.7 us/layer; one cross-GPU push
barrier costs ~7 us vs ~5.8 us for the whole pull-based AG kernel). Results:
/mnt/scratch/k3/kt/agf/run*.log, test: /mnt/scratch/k3/kt/agf/test_agf.py.

Replaces the SGLANG_ROCM_K3_UPPROJ_AG pair
    y_r  = latent @ w_up[r*Ns:(r+1)*Ns]^T          (hipBLASLt, [M, Ns])
    out  = bf16(bf16(allgather_lastdim(y) + shared) + prefix)   (aiter AG kernel)
with a push design: each CTA computes one [M, BN] tile of its rank's column
shard, applies the same two bf16-rounded adds (shared/prefix are identical on
every rank after the MoE all-reduce, so the owner can finish its columns), and
stores the finished bf16 tile straight into all ranks' output buffers over
XGMI (IPC). One cross-rank barrier ends the kernel: each CTA bumps a
per-source arrival counter on every rank (release, system scope); every CTA
then waits until all sources' counters reach this call's target (acquire,
system scope). Counters only grow; the per-rank call base lives in device
memory and is advanced by CTA 0 after its wait, so the kernel is CUDA-graph
replayable with no host involvement. Two output buffers, chosen by the caller
per call site (alternate between consecutive calls), so a rank racing ahead
into the next call never overwrites data a peer may still be reading.

Numerics: the add part is bit-identical to aiter allgather_lastdim_add (fp32
add, bf16 round, twice). The GEMM is a Triton tl.dot over the full K (fp32
accumulate), not hipBLASLt, so y differs from the tgemm path by bf16-rounding
level (see /mnt/scratch/k3/kt/agf/test_agf.py).

Contract: every rank must call this the same number of times (like any
collective). Warm-up / capture skips must be rank-symmetric.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import torch
import triton
import triton.language as tl

_HIP = None


def _hip():
    global _HIP
    if _HIP is None:
        _HIP = ctypes.CDLL("libamdhip64.so")
    return _HIP


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: hipError {rc}")


@triton.jit
def _k3_upproj_ag_fused_kernel(
    x_ptr,  # [M, K] bf16 latent (already normed), row stride stride_x
    stride_x,
    w_ptr,  # [NS, K] bf16 this rank's up_proj rows (contiguous)
    sh_ptr,  # [M, N] bf16 shared-expert output (identical on all ranks)
    stride_sh,
    pf_ptr,  # [M, N] bf16 prefix (identical on all ranks), or sh_ptr if not HAS_PF
    stride_pf,
    out_tbl,  # int64 [2, WS]: output buffer base address per (parity, rank)
    cnt_tbl,  # int64 [WS]: per-rank arrival flags base (uncached i32 [WS], one per source rank)
    base_ptr,  # int32 [4] local: [completed-call epoch, sticky timeout flag, CTA ticket, -]
    M,
    rank,
    parity,
    N: tl.constexpr,
    NS: tl.constexpr,
    K: tl.constexpr,
    WS: tl.constexpr,
    HAS_PF: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPIN_LIMIT: tl.constexpr = 1 << 28,
    SYNC: tl.constexpr = True,
    MAX_CTA: tl.constexpr = 128,
    RELEASE_SCOPE: tl.constexpr = "sys",  # False: profiling only (no barrier; results invalid)
):
    pid = tl.program_id(0)
    n_cta = tl.num_programs(0)
    base = tl.load(base_ptr)
    offs_m = tl.arange(0, BM)
    offs_n = pid * BN + tl.arange(0, BN)
    mm = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_m[:, None] * stride_x + offs_k[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    y = acc.to(tl.bfloat16)
    col = rank * NS + offs_n  # global output columns
    m2 = mm[:, None]
    s = tl.load(sh_ptr + offs_m[:, None] * stride_sh + col[None, :], mask=m2, other=0.0)
    y = (y.to(tl.float32) + s.to(tl.float32)).to(tl.bfloat16)
    if HAS_PF:
        p = tl.load(pf_ptr + offs_m[:, None] * stride_pf + col[None, :], mask=m2, other=0.0)
        y = (y.to(tl.float32) + p.to(tl.float32)).to(tl.bfloat16)
    # push the finished tile to every rank (row stride N)
    for r in tl.static_range(WS):
        dst = tl.load(out_tbl + parity * WS + r).to(tl.pointer_type(tl.bfloat16))
        tl.store(dst + offs_m[:, None] * N + col[None, :], y, mask=m2)
    if SYNC:
        tl.debug_barrier()
        # 1) local last-arriver ticket: the release (agent scope) waits for
        #    this CTA's remote pushes to be acknowledged before counting it
        epoch = base + 1
        ticket = tl.atomic_add(base_ptr + 2, 1, sem="release", scope="gpu")
        if ticket == n_cta - 1:
            tl.atomic_xchg(base_ptr + 2, 0, sem="relaxed", scope="gpu")  # re-arm
            # 2) one arrival per rank: flag[src = rank] = epoch on every rank
            #    (system-scope release: all of this rank's pushes are visible)
            for r in tl.static_range(WS):
                f = tl.load(cnt_tbl + r).to(tl.pointer_type(tl.int32)) + rank
                if r == 0:
                    tl.atomic_xchg(f, epoch, sem="release", scope=RELEASE_SCOPE)
                else:
                    tl.atomic_xchg(f, epoch, sem="relaxed", scope="sys")
            # 3) this CTA holds the kernel open until every rank's flag shows
            #    this epoch: all WS flags in one vector atomic per poll round.
            # NB: the bounded compound loop condition is load-bearing: a plain
            # `while v < target` around a scalar atomic miscompiles on gfx950;
            # plain volatile loads of the remotely written flags never
            # observed the update.
            me = tl.load(cnt_tbl + rank).to(tl.pointer_type(tl.int32))
            offs_s = tl.arange(0, WS)
            zero = tl.zeros((WS,), dtype=tl.int32)
            v = tl.min(tl.atomic_add(me + offs_s, zero, sem="relaxed", scope="sys"), axis=0)
            it = 0
            while (v < epoch) & (it < SPIN_LIMIT):
                v = tl.min(tl.atomic_add(me + offs_s, zero, sem="relaxed", scope="sys"), axis=0)
                it += 1
            if it >= SPIN_LIMIT:
                tl.store(base_ptr + 1, 1)  # timed out: sticky error flag
            tl.atomic_add(me, 0, sem="acquire", scope="sys")
            tl.store(base_ptr, epoch)


class UpProjAGFused:
    """Owns the IPC output buffers and arrival counters of one TP group.

    Construct once per process (before graph capture) on every rank, passing a
    CPU (gloo) process group for the handle exchange."""

    def __init__(self, rank: int, world_size: int, cpu_group, n: int, max_m: int,
                 device: torch.device):
        import torch.distributed as dist

        hip = _hip()
        self.rank, self.ws, self.n, self.max_m = rank, world_size, n, max_m
        self.device = device
        nbytes = 2 * max_m * n * 2  # two parities of [max_m, n] bf16
        # data: plain device memory (written remotely, read locally after an
        # acquire); counters: uncached so remote atomics are coherent
        data = ctypes.c_void_p()
        _check(hip.hipMalloc(ctypes.byref(data), ctypes.c_size_t(nbytes)), "hipMalloc")
        cnt = ctypes.c_void_p()
        _check(hip.hipExtMallocWithFlags(ctypes.byref(cnt), ctypes.c_size_t(4096), ctypes.c_uint(3)),
               "hipExtMallocWithFlags")
        _check(hip.hipMemset(cnt, 0, ctypes.c_size_t(4096)), "hipMemset")
        _check(hip.hipDeviceSynchronize(), "sync")
        hd, hc = _IpcHandle(), _IpcHandle()
        _check(hip.hipIpcGetMemHandle(ctypes.byref(hd), data), "hipIpcGetMemHandle")
        _check(hip.hipIpcGetMemHandle(ctypes.byref(hc), cnt), "hipIpcGetMemHandle")
        mine = (ctypes.string_at(ctypes.addressof(hd), 64), ctypes.string_at(ctypes.addressof(hc), 64))
        allh = [None] * world_size
        dist.all_gather_object(allh, mine, group=cpu_group)
        data_addrs, cnt_addrs = [], []
        self._opened = []
        for r, (bd, bc) in enumerate(allh):
            if r == rank:
                data_addrs.append(data.value)
                cnt_addrs.append(cnt.value)
                continue
            for b, lst in ((bd, data_addrs), (bc, cnt_addrs)):
                h = _IpcHandle()
                ctypes.memmove(ctypes.addressof(h), b, 64)
                p = ctypes.c_void_p()
                _check(hip.hipIpcOpenMemHandle(ctypes.byref(p), h, ctypes.c_uint(1)), "hipIpcOpenMemHandle")
                self._opened.append(p)
                lst.append(p.value)
        half = max_m * n * 2
        tbl = [a for a in data_addrs] + [a + half for a in data_addrs]  # [parity 0 | parity 1]
        self.out_tbl = torch.tensor(tbl, dtype=torch.int64, device=device)
        self.cnt_tbl = torch.tensor(cnt_addrs, dtype=torch.int64, device=device)
        # [call base, sticky barrier-timeout flag]
        self.base = torch.zeros(4, dtype=torch.int32, device=device)
        self.spin_limit = 1 << 28
        self.release_scope = "sys"  # "gpu" measured no faster (kt/agf/run7.log)
        self._data, self._cnt = data, cnt
        # local views of the two parity buffers, for returning outputs
        self._local = [self._wrap(data.value + i * half, (max_m, n)) for i in range(2)]
        dist.barrier(group=cpu_group)

    def _wrap(self, addr, shape):
        # a torch view of raw device memory owned by this object
        numel = shape[0] * shape[1]

        class _Holder:
            __cuda_array_interface__ = {
                "shape": (numel,), "typestr": "<i2", "data": (addr, False), "version": 3,
            }

        t = torch.as_tensor(_Holder(), device=self.device)
        return t.view(torch.bfloat16).view(shape)

    def cfg(self, m: int):
        # (BM, BN, BK, num_warps, num_stages)
        # single-GPU sweep (kt/agf/tune_gemm.py), MI355X, N=896 K=3584
        if m <= 16:
            return 16, 16, 512, 4, 3
        if m <= 32:
            return 32, 16, 256, 8, 3
        return 64, 16, 256, 4, 3

    def __call__(self, latent, w_shard, shared, prefix: Optional[torch.Tensor], parity: int,
                 cfg=None, _sync: bool = True) -> torch.Tensor:
        """parity (0/1): which output buffer this call writes and returns.
        Consecutive calls must alternate (e.g. MoE layer index % 2): a rank
        that runs ahead into the next call then writes the other buffer."""
        m, k = latent.shape
        ns = w_shard.shape[0]
        assert m <= self.max_m and ns * self.ws == self.n and w_shard.is_contiguous()
        bm, bn, bk, nw, nst = cfg or self.cfg(m)
        assert ns % bn == 0 and k % bk == 0 and ns // bn <= 128
        has_pf = prefix is not None
        _k3_upproj_ag_fused_kernel[(ns // bn,)](
            latent, latent.stride(0), w_shard, shared, shared.stride(0),
            prefix if has_pf else shared, prefix.stride(0) if has_pf else 0,
            self.out_tbl, self.cnt_tbl, self.base, m, self.rank, parity,
            N=self.n, NS=ns, K=k, WS=self.ws, HAS_PF=has_pf, BM=bm, BN=bn, BK=bk,
            SYNC=_sync, SPIN_LIMIT=self.spin_limit, RELEASE_SCOPE=self.release_scope, num_warps=nw, num_stages=nst,
        )
        return self._local[parity][:m]
