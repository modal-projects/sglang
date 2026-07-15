"""Shared harness for Inkling kernel-fusion old-vs-new microbenchmarks.

Each opt script imports this, runs correctness + latency sweeps, and at the end
calls ``Collector.emit()`` which prints a delimited JSON block to stdout. Results
are collected from stdout by the runner (the container->host bind mount is synced
one-way with ``--delete``, so files written in the container do not survive).

Timing uses CUDA events (median over ``reps`` independent launches).
"""

from __future__ import annotations

import json
from typing import Callable, Optional

import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16

_JSON_BEGIN = "===FUSION-JSON-BEGIN==="
_JSON_END = "===FUSION-JSON-END==="


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


def rand(*shape, dtype=DTYPE, device=DEVICE) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device=device)


def bench(fn: Callable[[], object], *, warmup: int = 25, reps: int = 100) -> float:
    """Median wall time of ``fn`` in microseconds, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    return times_us[len(times_us) // 2]


class Collector:
    """Accumulates correctness + latency rows and emits a JSON summary."""

    def __init__(self, opt: str, title: str) -> None:
        self.opt = opt
        self.title = title
        self.correctness: list[dict] = []
        self.latency: list[dict] = []
        self._n_fail = 0

    # -- correctness -------------------------------------------------------- #
    def check(
        self,
        label: str,
        ref: torch.Tensor,
        opt: torch.Tensor,
        *,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        extra_ok: bool = True,
        extra_msg: str = "",
    ) -> bool:
        ref_f = ref.float()
        opt_f = opt.float()
        diff = (ref_f - opt_f).abs()
        max_diff = float(diff.max()) if diff.numel() else 0.0
        mean_diff = float(diff.mean()) if diff.numel() else 0.0
        ok_val = torch.allclose(ref_f, opt_f, atol=atol, rtol=rtol)
        ok = bool(ok_val and extra_ok)
        if not ok:
            self._n_fail += 1
        tag = "PASS" if ok else "FAIL"
        msg = f"  [{tag}] {label}  max_diff={max_diff:.3e} mean_diff={mean_diff:.3e}"
        if extra_msg:
            msg += f"  {extra_msg}"
        print(msg)
        self.correctness.append(
            {
                "label": label,
                "ok": ok,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "atol": atol,
                "rtol": rtol,
                "note": extra_msg,
            }
        )
        return ok

    def record_ok(self, label: str, ok: bool, note: str = "") -> None:
        if not ok:
            self._n_fail += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}  {note}")
        self.correctness.append(
            {"label": label, "ok": bool(ok), "max_diff": None, "mean_diff": None, "note": note}
        )

    # -- latency ------------------------------------------------------------ #
    def latency_row(self, shape: dict, variants: dict[str, float], baseline_key: str) -> None:
        """Record one latency row. ``variants`` maps variant name -> microseconds.

        Speedups are computed relative to ``baseline_key`` (old impl)."""
        base = variants[baseline_key]
        speedups = {
            k: (base / v if v > 0 else float("inf"))
            for k, v in variants.items()
            if k != baseline_key
        }
        self.latency.append({"shape": shape, "us": variants, "speedup_vs_old": speedups})

    # -- output ------------------------------------------------------------- #
    def emit(self) -> None:
        total = len(self.correctness)
        n_pass = total - self._n_fail
        print(f"\n[{self.opt}] correctness: {n_pass}/{total} passed, {self._n_fail} failed")
        payload = {
            "opt": self.opt,
            "title": self.title,
            "n_correct_pass": n_pass,
            "n_correct_total": total,
            "correctness": self.correctness,
            "latency": self.latency,
        }
        print(_JSON_BEGIN)
        print(json.dumps(payload))
        print(_JSON_END)


def print_latency_table(
    rows: list[dict],
    shape_keys: list[str],
    variant_keys: list[str],
    baseline_key: str,
    title: str,
) -> None:
    """Print an aligned latency table from Collector.latency rows."""
    headers = list(shape_keys) + [f"{v} (us)" for v in variant_keys]
    # add speedup columns for non-baseline variants (base/variant, relative to baseline_key)
    speed_variants = [v for v in variant_keys if v != baseline_key]
    headers += [f"{v}/{baseline_key}" for v in speed_variants]

    table: list[list[str]] = []
    for r in rows:
        line = [str(r["shape"].get(k, "")) for k in shape_keys]
        for v in variant_keys:
            val = r["us"].get(v)
            line.append(f"{val:.1f}" if val is not None else "-")
        for v in speed_variants:
            sp = r["speedup_vs_old"].get(v)
            line.append(f"{sp:.2f}x" if sp is not None else "-")
        table.append(line)

    widths = [
        max(len(headers[c]), max((len(table[r][c]) for r in range(len(table))), default=0)) + 2
        for c in range(len(headers))
    ]
    sep = "+" + "+".join("-" * w for w in widths) + "+"
    print(f"\n{title}")
    print(sep)
    print("|" + "|".join(h.center(w) for h, w in zip(headers, widths)) + "|")
    print(sep)
    for line in table:
        print("|" + "|".join(c.center(w) for c, w in zip(line, widths)) + "|")
    print(sep)
