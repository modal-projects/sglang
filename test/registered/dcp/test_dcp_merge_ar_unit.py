"""
GPU unit tests for the DCP all-reduce LSE merge (``cp_lse_ag_out_ar_mla``).

Two parts:

(a) Single GPU: the (single, stride-driven) Triton correction kernel invoked
    with ``new_output_layout="BHD"`` and a bf16 output buffer — global LSE +
    ``out * factor`` — is compared against a torch fp32 reference, including
    the sentinel edge cases (+inf/NaN LSE rows -> -inf, factor == 0 forces the
    output to exactly 0 even when the local partial output is NaN).

(b) 4 GPUs (torchrun launch, skipped when fewer GPUs are available):
    ``cp_lse_ag_out_ar_mla`` (bf16 all-reduce + head slice) must match
    ``cp_lse_ag_out_rs_mla`` (fp32 reduce-scatter) to bf16 tolerance on
    random decode-shaped (tokens = bs) and verify-shaped (tokens = bs * draft)
    inputs, including a rank contributing an empty KV shard (+inf LSE / NaN
    output sentinel).

Usage:
    python -m pytest test_dcp_merge_ar_unit.py -v

This file doubles as the torchrun worker script (same pattern as
test_reduce_scatter_along_dim.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Tuple

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.layers.cp.dcp.comm import (
    cp_lse_ag_out_ar_mla,
    cp_lse_ag_out_rs_mla,
)
from sglang.srt.layers.cp.dcp.kernels import CPTritonContext, correct_attn_out
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

NPROC = 4

# (tokens, H, D): decode-like (q_len=1, bs tokens), verify-like (bs * draft
# tokens), and a small odd shape. H must be divisible by NPROC.
MERGE_SHAPES = [
    (48, 64, 512),  # decode, production-like (B200 dcp=4 profile)
    (128, 64, 512),  # target-verify, bs=16 * draft=8
    (7, 16, 128),  # small/odd token count
]


# ---------------------------------------------------------------------------
# Torch reference of the Triton correction kernel semantics (base-2 domain)
# ---------------------------------------------------------------------------


def _correct_ref(
    out: torch.Tensor,  # [B, H, D] any float dtype
    lses: torch.Tensor,  # [N, B, H] fp32
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (out_scaled fp32 [B, H, D], global_lse fp32 [B, H])."""
    neg_inf = float("-inf")
    lses = lses.to(torch.float32)
    # Sentinel handling: NaN / +inf -> -inf.
    lses = torch.where(
        torch.isnan(lses) | (lses == float("inf")),
        torch.full_like(lses, neg_inf),
        lses,
    )
    lse_max = lses.amax(dim=0)
    lse_max_safe = torch.where(
        lse_max == neg_inf, torch.zeros_like(lse_max), lse_max
    )
    lse_exp = torch.exp2(lses - lse_max_safe)
    global_lse = torch.log2(lse_exp.sum(dim=0)) + lse_max_safe

    diff = lses[rank] - global_lse
    diff = torch.where(
        torch.isnan(diff) | (diff == float("inf")),
        torch.full_like(diff, neg_inf),
        diff,
    )
    factor = torch.exp2(diff)  # [B, H], <= 1
    scaled = out.to(torch.float32) * factor.unsqueeze(-1)
    # factor == 0 guard: force exactly 0 (do not trust NaN partial outputs).
    scaled = torch.where(
        factor.unsqueeze(-1) == 0.0, torch.zeros_like(scaled), scaled
    )
    return scaled, global_lse


# ---------------------------------------------------------------------------
# Part (a): single-GPU kernel test (BHD/bf16 output layout)
# ---------------------------------------------------------------------------


def _run_kernel_case(
    tokens: int, num_heads: int, head_dim: int, world_size: int, sentinel: str
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(tokens * 31 + num_heads)
    rank = 1 % world_size

    out = torch.randn(tokens, num_heads, head_dim, device=device).to(
        torch.bfloat16
    )
    lses = (
        torch.randn(world_size, tokens, num_heads, device=device) * 4.0
    ).to(torch.float32)

    if sentinel == "local_empty":
        # This rank's KV shard is empty for the first half of the tokens:
        # LSE sentinel +inf (rows 0::2) / NaN (rows 1::2), NaN partial output.
        lses[rank, : tokens // 2 : 2] = float("inf")
        lses[rank, 1 : tokens // 2 : 2] = float("nan")
        out[: tokens // 2] = float("nan")
    elif sentinel == "remote_empty":
        other = (rank + 1) % world_size
        lses[other, :] = float("inf")
    elif sentinel == "all_empty":
        # Whole token row empty on every rank -> factor 0 -> output 0.
        lses[:, 0] = float("inf")
        out[0] = float("nan")

    out_scaled = out.new_empty(out.shape)  # bf16, [B, H, D]
    _, got_lse = correct_attn_out(
        out,
        lses,
        rank,
        CPTritonContext(),
        out_scaled,
        new_output_layout="BHD",
    )

    ref_scaled, ref_lse = _correct_ref(out, lses, rank)

    torch.testing.assert_close(
        out_scaled.to(torch.float32),
        ref_scaled.to(torch.bfloat16).to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    # Zero-weight rows must be exactly 0 (no NaN leakage into the allreduce).
    assert not torch.isnan(out_scaled).any(), sentinel
    torch.testing.assert_close(
        got_lse, ref_lse, rtol=1e-5, atol=1e-5, equal_nan=True
    )


@pytest.mark.parametrize("sentinel", ["none", "local_empty", "remote_empty", "all_empty"])
@pytest.mark.parametrize("shape", MERGE_SHAPES)
def test_correct_attn_out_bhd_bf16(shape, sentinel) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    tokens, num_heads, head_dim = shape
    _run_kernel_case(tokens, num_heads, head_dim, world_size=4, sentinel=sentinel)


def test_correct_attn_out_bad_layout_rejected() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    out = torch.randn(4, 8, 16, device="cuda", dtype=torch.bfloat16)
    lses = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        correct_attn_out(
            out, lses, 0, CPTritonContext(), out.new_empty(out.shape),
            new_output_layout="DBH",
        )


# ---------------------------------------------------------------------------
# Helpers for multiprocess launch (shared between test and worker)
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


# ---------------------------------------------------------------------------
# Part (b): 4-GPU ar-vs-rs parity test (runs via pytest, launches torchrun)
# ---------------------------------------------------------------------------


def test_dcp_merge_ar_matches_rs_4gpu() -> None:
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
    return rank, device, coord


@torch.inference_mode()
def worker_case(
    rank: int,
    device: torch.device,
    coord: ps.GroupCoordinator,
    shape: Tuple[int, int, int],
    empty_rank: int,
) -> List[str]:
    """One (shape, empty_rank) config: ar vs rs parity on random inputs.

    ``empty_rank`` < 0 disables the empty-shard case; otherwise that rank
    contributes the +inf-LSE / NaN-output sentinel for all its tokens.
    """
    errors: List[str] = []
    tokens, num_heads, head_dim = shape
    # Different data per rank (realistic partial outputs), same across the
    # ar/rs calls below.
    torch.manual_seed(1234 + rank + tokens)
    out = torch.randn(
        tokens, num_heads, head_dim, device=device
    ).to(torch.bfloat16)
    lse = (torch.randn(tokens, num_heads, device=device) * 4.0).to(
        torch.float32
    )
    if rank == empty_rank:
        lse[:] = float("inf")
        out[:] = float("nan")

    rs = cp_lse_ag_out_rs_mla(out.clone(), lse.clone(), coord)
    ar = cp_lse_ag_out_ar_mla(out.clone(), lse.clone(), coord)

    if ar.shape != rs.shape:
        errors.append(f"shape mismatch {tuple(ar.shape)} vs {tuple(rs.shape)}")
        return errors
    if ar.dtype != out.dtype:
        errors.append(f"dtype mismatch {ar.dtype} vs {out.dtype}")
    if not ar.is_contiguous():
        errors.append("ar output is not contiguous")
    if torch.isnan(ar).any() or torch.isnan(rs).any():
        errors.append(f"NaN in merged output (shape={shape}, empty={empty_rank})")
    # rs sums in fp32 then casts to bf16; ar sums bf16 terms — allow bf16-level
    # accumulation-order slack.
    if not torch.allclose(
        ar.to(torch.float32), rs.to(torch.float32), rtol=3e-2, atol=3e-2
    ):
        max_err = (ar.to(torch.float32) - rs.to(torch.float32)).abs().max().item()
        errors.append(
            f"ar/rs mismatch: shape={shape}, empty_rank={empty_rank}, "
            f"max_abs_err={max_err}"
        )
    return errors


def worker_main() -> None:
    """Entry point for each torchrun worker process."""
    rank, device, coord = init_distributed()

    errors: List[str] = []
    for shape in MERGE_SHAPES:
        for empty_rank in (-1, coord.world_size - 1):
            errors.extend(worker_case(rank, device, coord, shape, empty_rank))

            # Synchronize across ranks – if any rank fails, all fail.
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
