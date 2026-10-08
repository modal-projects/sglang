"""SGLANG_DEBUG_K3_STEP_TIMING: per-rank host + GPU step timeline, no rocprof.

One record per scheduler loop iteration. Host marks are ``time.perf_counter_ns``
(CLOCK_MONOTONIC: comparable across the TP rank processes of one host), so the
merge script can line ranks up by iteration index (``forward_ct`` is identical
on every TP rank) and measure cross-rank skew at every mark. GPU marks are
timing events on the current stream; their offsets are read lazily, once the
record's last event has completed (never a host sync).

Record line format (one per iteration, ``<DIR>/rank<r>.log``)::

    it=<forward_ct> t0=<abs ns> mode=<decode|extend|idle|none> bs=<n> \
        h:<mark>=<us since t0> ... g:<mark>=<us since first gpu mark> ... \
        gp=<us between this and the previous record's first gpu mark>

Every ``SGLANG_DEBUG_K3_STEP_TIMING`` decode iterations each rank also logs the
median of every consecutive host / GPU segment.
"""

from __future__ import annotations

import collections
import logging
import os
import statistics
import time
from typing import Deque, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class _StepTimer:
    def __init__(self) -> None:
        try:
            from sglang.srt.environ import envs

            self.every = int(envs.SGLANG_DEBUG_K3_STEP_TIMING.get())
            self.out_dir = envs.SGLANG_DEBUG_K3_STEP_TIMING_DIR.get()
            self.use_gpu = bool(envs.SGLANG_DEBUG_K3_STEP_TIMING_GPU.get())
        except Exception:  # pragma: no cover - env module unavailable
            self.every = 0
            self.out_dir = "/tmp/k3_step_timing"
            self.use_gpu = False
        self.enabled = self.every > 0
        self._rank: Optional[int] = None
        self._fh = None
        self._rec: Optional[dict] = None
        self._pending: Deque[dict] = collections.deque()
        self._prev_first_ev = None
        self._n_decode = 0
        self._hist: Dict[str, List[float]] = collections.defaultdict(list)

    # ------------------------------------------------------------------ marks
    def iter_begin(self) -> None:
        """Loop top: close the previous iteration's record, open a new one."""
        if self._rec is not None:
            self._close(self._rec)
        self._rec = {
            "t0": time.perf_counter_ns(),
            "h": [],
            "g": [],
            "it": -1,
            "mode": "none",
            "bs": 0,
        }

    def mark(self, name: str) -> None:
        rec = self._rec
        if rec is not None:
            rec["h"].append((name, time.perf_counter_ns()))

    def gpu(self, name: str) -> None:
        """Host mark + GPU timing event on the current stream."""
        rec = self._rec
        if rec is None:
            return
        rec["h"].append((name, time.perf_counter_ns()))
        if self.use_gpu:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            rec["g"].append((name, ev))

    def meta(self, *, it: Optional[int] = None, mode=None, bs=None) -> None:
        rec = self._rec
        if rec is None:
            return
        if it is not None:
            rec["it"] = int(it)
        if mode is not None:
            rec["mode"] = str(mode)
        if bs is not None:
            rec["bs"] = int(bs)

    # --------------------------------------------------------------- internals
    def _get_rank(self) -> int:
        if self._rank is None:
            r = 0
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    r = torch.distributed.get_rank()
            except Exception:
                r = 0
            self._rank = int(r)
        return self._rank

    def _file(self):
        if self._fh is None:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"rank{self._get_rank()}.log")
            self._fh = open(path, "a", buffering=1 << 16)
            self._fh.write(f"# k3 step timing pid={os.getpid()} start\n")
        return self._fh

    def _close(self, rec: dict) -> None:
        if not rec["h"] and not rec["g"]:
            return
        self._pending.append(rec)
        self._drain(final=False)

    def _drain(self, final: bool) -> None:
        while self._pending:
            rec = self._pending[0]
            g = rec["g"]
            if g and not final and not g[-1][1].query():
                return
            self._pending.popleft()
            self._emit(rec)

    def _emit(self, rec: dict) -> None:
        t0 = rec["t0"]
        parts = [
            f"it={rec['it']}",
            f"t0={t0}",
            f"mode={rec['mode']}",
            f"bs={rec['bs']}",
        ]
        host = [(n, (t - t0) / 1e3) for n, t in rec["h"]]
        parts += [f"h:{n}={v:.1f}" for n, v in host]
        gpu: List[Tuple[str, float]] = []
        g = rec["g"]
        if g:
            first = g[0][1]
            try:
                gpu = [(n, first.elapsed_time(ev) * 1e3) for n, ev in g]
                parts += [f"g:{n}={v:.1f}" for n, v in gpu]
                if self._prev_first_ev is not None:
                    parts.append(
                        f"gp={self._prev_first_ev.elapsed_time(first) * 1e3:.1f}"
                    )
            except Exception:
                gpu = []
            self._prev_first_ev = first
        self._file().write(" ".join(parts) + "\n")

        if rec["mode"] != "decode":
            return
        hist = self._hist
        for (a, ta), (b, tb) in zip(host, host[1:]):
            hist[f"h {a}->{b}"].append(tb - ta)
        if host:
            hist["h iter_total"].append(host[-1][1])
        for (a, ta), (b, tb) in zip(gpu, gpu[1:]):
            hist[f"g {a}->{b}"].append(tb - ta)
        self._n_decode += 1
        if self._n_decode % self.every == 0:
            self._summary()

    def _summary(self) -> None:
        r = self._get_rank()
        lines = []
        for k, v in self._hist.items():
            if v:
                lines.append(f"{k}={statistics.median(v):.1f}")
        logger.info(
            "[k3-step-timing rank %d] median us over %d decode iters: %s",
            r,
            self.every,
            " ".join(lines),
        )
        self._hist.clear()
        if self._fh is not None:
            self._fh.flush()


STEP_TIMER = _StepTimer()
