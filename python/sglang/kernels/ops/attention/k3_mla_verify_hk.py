"""HipKittens stage 1 and split merge of k3_mla_verify_auto (gfx950).

Stage 1 (jit/csrc/kimi_k3/k3_mla_verify_hk.cuh): one CTA per (request, KV split)
covers all 8 q_pos x 12 heads, so the KV is read once. Its inner loop is
hand-scheduled asm generated at build time by gen_k3_mla_verify_hk_asm.py.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

_LOG2E = 1.4426950408889634

_CSRC = os.path.join(os.path.dirname(__file__), "..", "..", "jit", "csrc", "kimi_k3")
_GEN = "gen_k3_mla_verify_hk_asm.py"


def _src_hash() -> str:
    h = hashlib.sha1()
    for name in ("k3_mla_verify_hk.cuh", _GEN):
        with open(os.path.join(_CSRC, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def _asm_include_dir(tag: str) -> str:
    d = os.path.join(tempfile.gettempdir(), f"sglang_k3_mla_verify_hk_{tag}")
    out = os.path.join(d, "k3_mla_verify_hk_asm.inc")
    if not os.path.exists(out):
        os.makedirs(d, exist_ok=True)
        tmp = f"{out}.{os.getpid()}"
        subprocess.run([sys.executable, os.path.join(_CSRC, _GEN), tmp], check=True)
        os.replace(tmp, out)
    return d


@cache_once
def _module():
    tag = _src_hash()
    return load_jit(
        "kimi_k3_mla_verify_hk",
        tag,
        cuda_files=["kimi_k3/k3_mla_verify_hk.cuh"],
        cuda_wrappers=[
            ("run", "K3MlaVerifyHK::run"),
            ("reduce", "K3MlaVerifyHK::reduce"),
        ],
        extra_cuda_cflags=["-O3", "-mllvm", "-amdgpu-spill-vgpr-to-agpr=0"],
        extra_include_paths=[_asm_include_dir(tag)],
    )


# Requests with at least this many active splits store fp8 partials (O / kv_scale), read
# only by hk_reduce.
PART8_MIN_SPLITS = 96


def hk_stage1(
    q,
    kv,
    kv_indptr,
    kv_indices,
    kv_scale,
    o_part,
    lse_part,
    nsplit: int,
    min_chunk: int,
    sm_scale: float,
    part8_min: int,
) -> None:
    _module().run(
        q,
        kv.view(torch.uint8),
        kv_indptr,
        kv_indices,
        kv_scale,
        o_part,
        lse_part,
        nsplit,
        min_chunk,
        float(sm_scale) * _LOG2E,
        part8_min,
    )


def hk_reduce(
    o_part,
    lse_part,
    kv_indptr,
    out,
    nsplit: int,
    min_chunk: int,
    kv_scale,
    part8_min: int,
    regime,
    g_logits,
    g_lse,
    g_ns: int,
    g_block_n: int,
) -> None:
    """Merge into out the HK splits (base-2 LSE), or the Gluon splits
    (natural-log LSE) when regime[0] == 0."""
    _module().reduce(
        o_part,
        lse_part,
        kv_indptr,
        out,
        regime,
        g_logits,
        g_lse,
        kv_scale,
        nsplit,
        min_chunk,
        g_ns,
        g_block_n,
        part8_min,
    )
