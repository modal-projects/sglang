"""Opt 1 (fair latency) — one (T, D, B) shape per fresh process.

The OLD Helion kernel keys its specialization on (D, dtype, W) and is fragile under
shape reuse: the first call in a process locks a specialization that yields WRONG
results for later differently-shaped calls (see opt1 correctness: OLD passes only at
T=1). To measure OLD latency fairly we must specialize it on the timed shape, i.e. run
one shape per fresh process so the first (and only) call is the shape under test.

Each invocation prints a single JSON line (parsed by the runner):
  {"T","D","B", "old_helion_contig","new_triton_contig","new_triton_noncontig",
   "old_correct","new_correct","old_max_diff","new_max_diff"}
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sconv_baseline as OLD
from _harness import DEVICE, DTYPE, bench, rand, set_seed
from opt1_causal_conv1d import _inputs, reference_causal_conv1d

from sglang.tml.kernels import sconv as NEW


def main() -> None:
    T, D, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    set_seed(0)
    k_c, k_nc, w, cache, om, nm = _inputs(T, D, W, B, "all")
    ref = reference_causal_conv1d(k_c, w, cache, **nm, activation="silu", use_residual=True)

    out = {"T": T, "D": D, "B": B}
    # NEW first (always correct); OLD specialized on this exact shape in this process.
    new_nc = NEW.causal_conv1d(k_nc, w, cache, **nm, activation="silu", use_residual=True)
    out["new_max_diff"] = float((ref.float() - new_nc.float()).abs().max())
    out["new_correct"] = bool(torch.allclose(ref.float(), new_nc.float(), atol=2e-2, rtol=2e-2))
    try:
        old = OLD.causal_conv1d(k_c, w, cache, **om, activation="silu", use_residual=True)
        d = (ref.float() - old.float()).abs()
        out["old_max_diff"] = float(d.max())
        out["old_correct"] = bool(torch.allclose(ref.float(), old.float(), atol=2e-2, rtol=2e-2))
        out["old_helion_contig"] = bench(
            lambda: OLD.causal_conv1d(k_c, w, cache, **om, activation="silu", use_residual=True)
        )
    except Exception as e:
        out["old_correct"] = False
        out["old_max_diff"] = None
        out["old_helion_contig"] = None
        out["old_error"] = f"{type(e).__name__}: {e}"
    out["new_triton_contig"] = bench(
        lambda: NEW.causal_conv1d(k_c, w, cache, **nm, activation="silu", use_residual=True)
    )
    out["new_triton_noncontig"] = bench(
        lambda: NEW.causal_conv1d(k_nc, w, cache, **nm, activation="silu", use_residual=True)
    )
    print("OPT1B_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
