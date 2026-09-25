"""Sampled eager preconversion Q rows for TRT-LLM MLA ragged prefill.

SGLANG_DEBUG_FP8_RANGE_EVERY=N observes every Nth nonempty causal call per
backend and layer; zero (the default) disables observer allocation and observation.
Prefix-only passes, CUDA capture/replay, decode, and other backend paths are
not covered. Sampling runs tensor reductions; draining synchronizes with their
producer streams and copies the bounded counters to the host.

The gt448 and gt464 kinds count fully finite rows with maximum absolute value
strictly above the threshold. Rows containing NaN or infinity count only as
nonfinite. These overlapping finite thresholds are observations, not clamp
counts or evidence identifying the original source of invalid values.
With metrics enabled, counters retain scheduler rank labels; warnings are
limited to one per layer per minute on the stats logging rank.
"""

import torch

FP8_RANGE_KINDS = ("gt448", "gt464", "nonfinite")


class Fp8RangeObserver:
    def __init__(self, *, num_layers: int, device, every: int):
        if every <= 0:
            raise ValueError("FP8 range sampling interval must be positive")
        self.every = every
        self._calls = [0] * num_layers
        self._counts = torch.zeros(
            (num_layers, len(FP8_RANGE_KINDS)), dtype=torch.int64, device=device
        )
        self._dirty = False
        self._event = None
        if self._counts.is_cuda:
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(self._counts.device))

    def observe(self, *, layer_id: int, q: torch.Tensor) -> None:
        # Python sampling must never become a fixed decision baked into replay.
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        if q.numel() == 0:
            return
        self._calls[layer_id] += 1
        if self._calls[layer_id] % self.every:
            return
        if q.is_cuda:
            stream = torch.cuda.current_stream(q.device)
            if self._event is None:
                self._event = torch.cuda.Event()
            else:
                stream.wait_event(self._event)
        row_max = q.detach().flatten(start_dim=1).abs().amax(dim=1)
        finite = torch.isfinite(row_max)
        self._counts[layer_id].add_(
            torch.stack(
                (
                    (finite & (row_max > 448)).sum(),
                    (finite & (row_max > 464)).sum(),
                    (~finite).sum(),
                )
            )
        )
        if self._event is not None:
            self._event.record(stream)
        self._dirty = True

    def drain(self) -> list[tuple[int, str, int]]:
        if not self._dirty:
            return []
        if self._event is not None:
            stream = torch.cuda.current_stream(self._counts.device)
            stream.wait_event(self._event)
        counts = self._counts.tolist()
        self._counts.zero_()
        if self._event is not None:
            self._event.record(stream)
        self._dirty = False
        return [
            (layer_id, kind, count)
            for layer_id, row in enumerate(counts)
            for kind, count in zip(FP8_RANGE_KINDS, row)
            if count
        ]
