"""
GPU unit test for DCP q-replication (SGLANG_DCP_REPLICATE_Q).

4-GPU torchrun test (skipped when fewer GPUs are available), mirroring the
bootstrap of test_dcp_merge_ar_unit.py. Builds a toy head-sharded q_b_proj /
w_kc across a 4-rank group and checks that the replicated-q path (all-gather
the WEIGHT shards once, then compute every head's absorbed q locally --
what ``build_dcp_qrep_weights`` + the ``dcp_qrep`` branch of
``forward_absorb_prepare`` do) matches the per-step gather path
(``all_gather_q_for_mla_decode``):

(a) load-time weight gather reproduces the full weights bitwise (head-order
    equality of the dcp-group rank-order concat vs the unsharded weight);
(b) replicated-path q_nope_out / q_pe match the gathered-path outputs to
    GEMM-reduction-order tolerance on random weights, decode- and
    verify-shaped token counts, with rope applied;
(c) head-stamp ordering: weights constructed so head h outputs exactly h+1
    -- both paths must yield the stamp bitwise at every head index, proving
    the replicated head order equals the gather's head order (and hence the
    order assumed by the cp_lse_ag_out_mla head slice).

Usage:
    python -m pytest test_dcp_qrep_unit.py -v

This file doubles as the torchrun worker script (same pattern as
test_dcp_merge_ar_unit.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.layers.cp.dcp.comm import all_gather_q_for_mla_decode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

NPROC = 4

# Toy MLA geometry (H must be divisible by NPROC).
H = 8
QK_NOPE = 32
QK_ROPE = 16
KV_LORA = 64
Q_LORA = 96
QK_HEAD = QK_NOPE + QK_ROPE

# decode-like (bs tokens) and verify-like (bs * draft tokens)
TOKEN_COUNTS = [6, 48]


def _apply_rope(q_pe: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Simple neox-style rope, per-position and head-independent -- the two
    properties the production path relies on when roping the full-head q
    locally instead of roping the local shard before the gather."""
    tokens, num_heads, dim = q_pe.shape
    half = dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(half, device=q_pe.device, dtype=torch.float32) / half)
    )
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]  # [T, half]
    cos = freqs.cos()[:, None, :]  # [T, 1, half] -- broadcast over heads
    sin = freqs.sin()[:, None, :]
    x1, x2 = q_pe[..., :half], q_pe[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _run_both_paths(
    coord: ps.GroupCoordinator,
    rank: int,
    q_b_full: torch.Tensor,  # [H * QK_HEAD, Q_LORA]
    w_kc_full: torch.Tensor,  # [H, QK_NOPE, KV_LORA]
    q_a: torch.Tensor,  # [tokens, Q_LORA] -- replicated input (post q_a_norm)
    positions: torch.Tensor,  # [tokens]
):
    """Returns ((ref_nope, ref_pe), (rep_nope, rep_pe), (q_b_dcp, w_kc_dcp))."""
    world = coord.world_size
    local_h = H // world
    row0 = rank * local_h * QK_HEAD
    q_b_shard = q_b_full[row0 : row0 + local_h * QK_HEAD]
    w_kc_shard = w_kc_full[rank * local_h : (rank + 1) * local_h]

    # --- Gather-based reference path (today's production flow): local-head
    # q_b -> absorb bmm -> rope -> all_gather_q_for_mla_decode.
    q_local = F.linear(q_a, q_b_shard).view(-1, local_h, QK_HEAD)
    q_nope_l, q_pe_l = q_local.split([QK_NOPE, QK_ROPE], dim=-1)
    q_nope_out_l = torch.bmm(q_nope_l.transpose(0, 1), w_kc_shard).transpose(0, 1)
    q_pe_l = _apply_rope(q_pe_l, positions)
    ref_nope, ref_pe = all_gather_q_for_mla_decode(
        q_nope_out=q_nope_out_l.contiguous(),
        q_pe=q_pe_l.contiguous(),
    )

    # --- Replicated path: load-time weight gather (exactly what
    # build_dcp_qrep_weights does: rank-order concat along the head/output
    # dim over the dcp group), then full-head local compute.
    q_b_dcp = coord.all_gather(q_b_shard.contiguous(), dim=0)
    w_kc_dcp = coord.all_gather(w_kc_shard.contiguous(), dim=0)
    q_full = F.linear(q_a, q_b_dcp).view(-1, H, QK_HEAD)
    q_nope_f, q_pe_f = q_full.split([QK_NOPE, QK_ROPE], dim=-1)
    rep_nope = torch.bmm(q_nope_f.transpose(0, 1), w_kc_dcp).transpose(0, 1)
    rep_pe = _apply_rope(q_pe_f, positions)

    return (ref_nope, ref_pe), (rep_nope, rep_pe), (q_b_dcp, w_kc_dcp)


@torch.inference_mode()
def worker_case_random(
    rank: int, device: torch.device, coord: ps.GroupCoordinator, tokens: int
) -> List[str]:
    errors: List[str] = []
    # SAME seed on every rank: the full toy weights and the (replicated)
    # q_a input are identical everywhere, like the real model.
    torch.manual_seed(20260701 + tokens)
    q_b_full = torch.randn(H * QK_HEAD, Q_LORA, device=device)
    w_kc_full = torch.randn(H, QK_NOPE, KV_LORA, device=device) / KV_LORA**0.5
    q_a = torch.randn(tokens, Q_LORA, device=device)
    positions = torch.arange(100, 100 + tokens, device=device)

    (ref_nope, ref_pe), (rep_nope, rep_pe), (q_b_dcp, w_kc_dcp) = _run_both_paths(
        coord, rank, q_b_full, w_kc_full, q_a, positions
    )

    # (a) load-time gather == unsharded weight, bitwise (head order).
    if not torch.equal(q_b_dcp, q_b_full):
        errors.append(f"q_b_proj dcp-gather != full weight (tokens={tokens})")
    if not torch.equal(w_kc_dcp, w_kc_full):
        errors.append(f"w_kc dcp-gather != full weight (tokens={tokens})")

    # (b) path parity. Same weights/inputs; only GEMM reduction grouping
    # differs (full-K vs sharded-K matmul tiling), so fp32-tight tolerance.
    if ref_nope.shape != rep_nope.shape or ref_pe.shape != rep_pe.shape:
        errors.append(
            f"shape mismatch: gather {tuple(ref_nope.shape)}/{tuple(ref_pe.shape)}"
            f" vs replicated {tuple(rep_nope.shape)}/{tuple(rep_pe.shape)}"
        )
        return errors
    if not torch.allclose(ref_nope, rep_nope, rtol=1e-4, atol=1e-4):
        max_err = (ref_nope - rep_nope).abs().max().item()
        errors.append(f"q_nope_out mismatch (tokens={tokens}) max_abs={max_err}")
    if not torch.allclose(ref_pe, rep_pe, rtol=1e-4, atol=1e-4):
        max_err = (ref_pe - rep_pe).abs().max().item()
        errors.append(f"q_pe mismatch (tokens={tokens}) max_abs={max_err}")
    return errors


@torch.inference_mode()
def worker_case_head_stamp(
    rank: int, device: torch.device, coord: ps.GroupCoordinator
) -> List[str]:
    """Head-order proof: head h's q_b rows read only q_a[0] with weight h+1,
    w_kc[h] passes component 0 through -- so q_nope_out[:, h, 0] == h+1
    EXACTLY in both paths iff the head ordering matches everywhere."""
    errors: List[str] = []
    tokens = 5
    q_b_full = torch.zeros(H * QK_HEAD, Q_LORA, device=device)
    for h in range(H):
        q_b_full[h * QK_HEAD : (h + 1) * QK_HEAD, 0] = float(h + 1)
    w_kc_full = torch.zeros(H, QK_NOPE, KV_LORA, device=device)
    w_kc_full[:, 0, 0] = 1.0
    q_a = torch.zeros(tokens, Q_LORA, device=device)
    q_a[:, 0] = 1.0
    positions = torch.zeros(tokens, dtype=torch.int64, device=device)

    (ref_nope, ref_pe), (rep_nope, rep_pe), _ = _run_both_paths(
        coord, rank, q_b_full, w_kc_full, q_a, positions
    )
    stamp = torch.arange(1, H + 1, device=device, dtype=torch.float32)
    stamp = stamp[None, :].expand(tokens, H)
    if not torch.equal(ref_nope[:, :, 0], stamp):
        errors.append(f"gather path head stamp wrong: {ref_nope[0, :, 0].tolist()}")
    if not torch.equal(rep_nope[:, :, 0], stamp):
        errors.append(f"replicated path head stamp wrong: {rep_nope[0, :, 0].tolist()}")
    # rope with position 0 is identity on the stamped q_pe component too
    if not torch.equal(ref_pe, rep_pe):
        errors.append("q_pe head stamp mismatch between paths")
    return errors


# ---------------------------------------------------------------------------
# Multiprocess launch plumbing (mirrors test_dcp_merge_ar_unit.py)
# ---------------------------------------------------------------------------


def multiprocess_test(file: str, nproc: int, timeout: int = 240) -> None:
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        file,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"torchrun (nproc={nproc}) timed out after {timeout}s\n{e.stdout}"
        ) from e

    assert result.returncode == 0, (
        f"torchrun (nproc={nproc}) failed with rc={result.returncode}\n"
        f"{result.stdout}"
    )


def test_dcp_qrep_matches_gather_4gpu() -> None:
    device_count = torch.cuda.device_count()
    if device_count < NPROC:
        pytest.skip(
            f"Requires at least {NPROC} GPUs, but only {device_count} available"
        )
    multiprocess_test(__file__, NPROC)


# ---------------------------------------------------------------------------
# Worker logic (executed by each torchrun process)
# ---------------------------------------------------------------------------


def init_distributed():
    """Initialize distributed groups via torchrun env vars."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = local_rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    # all_gather_q_for_mla_decode reaches the group via get_dcp_group().
    ps._DCP = coord
    return rank, device, coord


def worker_main() -> None:
    rank, device, coord = init_distributed()

    errors: List[str] = []
    for tokens in TOKEN_COUNTS:
        errors.extend(worker_case_random(rank, device, coord, tokens))
    errors.extend(worker_case_head_stamp(rank, device, coord))

    # Synchronize failure state across ranks.
    result = torch.tensor([len(errors)], device="cpu")
    dist.all_reduce(result, group=coord.cpu_group)
    if result.item() > 0 and not errors:
        errors.append("failure on another rank")

    dist.barrier(group=coord.cpu_group)
    if errors:
        print(f"[rank {rank}] FAILURES:\n" + "\n".join(errors), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        worker_main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
