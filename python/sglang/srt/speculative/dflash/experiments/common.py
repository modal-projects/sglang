from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch

from sglang.srt.speculative.dflash.bench.common import TimingStats, format_timing_stats


def assert_dataclass_tensors_equal(
    expected: object,
    actual: object,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    if not (is_dataclass(expected) and is_dataclass(actual)):
        raise TypeError("assert_dataclass_tensors_equal expects dataclass instances.")
    if type(expected) is not type(actual):
        raise TypeError(
            "Dataclass type mismatch. "
            f"Expected {type(expected).__name__}, got {type(actual).__name__}."
        )

    for field in fields(expected):
        lhs = getattr(expected, field.name)
        rhs = getattr(actual, field.name)
        if is_dataclass(lhs):
            assert_dataclass_tensors_equal(lhs, rhs, atol=atol, rtol=rtol)
        elif isinstance(lhs, torch.Tensor):
            if not isinstance(rhs, torch.Tensor):
                raise AssertionError(
                    f"Field '{field.name}' expected a tensor, got {type(rhs).__name__}."
                )
            if atol == 0.0 and rtol == 0.0:
                if not torch.equal(lhs, rhs):
                    raise AssertionError(
                        f"Field '{field.name}' does not match exactly."
                    )
            elif not torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"Field '{field.name}' does not match within atol={atol}, rtol={rtol}."
                )
        else:
            if lhs != rhs:
                raise AssertionError(
                    f"Field '{field.name}' does not match: {lhs!r} != {rhs!r}."
                )


def print_stats_block(title: str, stats_by_variant: dict[str, TimingStats]) -> None:
    print(f"\n{title}")
    for variant, stats in stats_by_variant.items():
        print(format_timing_stats(variant, stats))
