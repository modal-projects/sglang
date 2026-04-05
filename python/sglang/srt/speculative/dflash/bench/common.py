from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class TimingStats:
    warmup: int
    iters: int
    mean_us: float
    median_us: float
    p95_us: float
    min_us: float
    max_us: float
    stdev_us: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile for an empty sample.")
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    index = (len(values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def time_callable(
    fn: Callable[[], object],
    *,
    warmup: int = 25,
    iters: int = 100,
    device: torch.device | str | None = None,
) -> TimingStats:
    if warmup < 0 or iters <= 0:
        raise ValueError(
            f"Expected warmup >= 0 and iters > 0, got warmup={warmup}, iters={iters}."
        )

    device_obj = torch.device(device) if device is not None else None
    use_cuda_events = bool(device_obj is not None and device_obj.type == "cuda")

    for _ in range(warmup):
        fn()
        if use_cuda_events:
            torch.cuda.synchronize(device_obj)

    samples_us: list[float] = []
    if use_cuda_events:
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples_us.append(float(start.elapsed_time(end) * 1000.0))
    else:
        for _ in range(iters):
            start_ns = time.perf_counter_ns()
            fn()
            end_ns = time.perf_counter_ns()
            samples_us.append((end_ns - start_ns) / 1000.0)

    return TimingStats(
        warmup=warmup,
        iters=iters,
        mean_us=statistics.mean(samples_us),
        median_us=statistics.median(samples_us),
        p95_us=_percentile(samples_us, 0.95),
        min_us=min(samples_us),
        max_us=max(samples_us),
        stdev_us=statistics.pstdev(samples_us),
    )


def format_timing_stats(name: str, stats: TimingStats) -> str:
    return (
        f"{name:<18} median={stats.median_us:10.2f} us  "
        f"p95={stats.p95_us:10.2f} us  "
        f"mean={stats.mean_us:10.2f} us"
    )
