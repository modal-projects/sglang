"""Per-phase GPU timing of DFlash decode steps (debug; SGLANG_DFLASH_STEP_TIMING=N).

Records device events at phase boundaries of each verify step and, every N steps,
logs the mean ms per phase and the start-to-start step period per batch size on
TP rank 0. Off by default; when on it adds one host sync per N steps.
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class DFlashStepTimer:
    def __init__(self, interval: int, rank: int):
        self.interval = int(interval)
        self.rank = int(rank)
        self._steps: List[Tuple[str, int, List[Tuple[str, torch.cuda.Event]]]] = []
        self._cur: Optional[List[Tuple[str, torch.cuda.Event]]] = None
        self._cur_kind = ""
        self._cur_bs = 0
        self._sums: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._counts: Dict[Tuple[str, int], int] = defaultdict(int)

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def _event(self) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def begin(self, kind: str, bs: int) -> None:
        if not self.enabled:
            return
        self._cur_kind, self._cur_bs = kind, int(bs)
        self._cur = [("start", self._event())]

    def mark(self, name: str) -> None:
        if self._cur is not None:
            self._cur.append((name, self._event()))

    def end(self) -> None:
        if self._cur is None:
            return
        self._steps.append((self._cur_kind, self._cur_bs, self._cur))
        self._cur = None
        if len(self._steps) >= self.interval:
            self._flush()

    def _flush(self) -> None:
        steps, self._steps = self._steps, []
        steps[-1][2][-1][1].synchronize()
        for i, (kind, bs, events) in enumerate(steps):
            key = (kind, bs)
            sums = self._sums[key]
            for (_, a), (name, b) in zip(events, events[1:]):
                sums[name] += a.elapsed_time(b)
            sums["total"] += events[0][1].elapsed_time(events[-1][1])
            if i + 1 < len(steps):
                sums["period"] += events[0][1].elapsed_time(steps[i + 1][2][0][1])
                sums["period_n"] += 1
            self._counts[key] += 1
        if self.rank != 0:
            self._sums.clear()
            self._counts.clear()
            return
        rows = []
        for (kind, bs), sums in sorted(self._sums.items()):
            n = self._counts[(kind, bs)]
            period_n = sums.pop("period_n", 0.0)
            period = sums.pop("period", 0.0)
            row = {"kind": kind, "bs": bs, "n": n}
            row.update({k: round(v / n, 4) for k, v in sums.items()})
            if period_n:
                row["period"] = round(period / period_n, 4)
            rows.append(row)
        logger.info("DFLASH_STEP_TIMING %s", json.dumps(rows))
        self._sums.clear()
        self._counts.clear()
