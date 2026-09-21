"""Breakable prefill bridge for the request-dependent pooled-key indexer."""

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)


def _kpool_indexer_prefill_with_output(
    indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    # Metadata, write counts and cache destinations change between requests.
    # Resolve the live batch inside the eager break, never from capture args.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    n = forward_batch.extend_num_tokens
    if n is None or not 0 <= n <= x.shape[0]:
        raise ValueError(f"Invalid pooled-indexer prefill token count: {n}")
    if n > q_lora.shape[0] or n > positions.shape[0]:
        raise ValueError("Pooled-indexer prefill inputs have inconsistent rows")
    real_n = forward_batch.global_num_token_non_padded_cpu
    real_n = n if real_n is None else real_n
    if not 0 <= real_n <= n:
        raise ValueError(
            "Invalid pooled-indexer non-padded token count: "
            f"real={real_n}, padded={n}"
        )
    return_indices = output.shape[0] != 0
    result = indexer._forward_cuda_impl(
        x=x[:real_n],
        q_lora=q_lora[:real_n],
        positions=positions[:real_n],
        forward_batch=forward_batch,
        layer_id=layer_id,
        return_indices=return_indices,
    )
    if not return_indices:
        return
    # Most eager indexer paths return only the live rows. The short-sequence
    # fast path derives its row count from graph-padded attention metadata and
    # therefore returns the captured bucket instead. Both layouts describe the
    # same live prefix; only that prefix may flow into the following captured
    # attention segment.
    expected_rows = {real_n, n, output.shape[0]}
    if (
        result is None
        or result.ndim != 2
        or result.shape[0] not in expected_rows
        or result.shape[1] != output.shape[1]
    ):
        result_shape = None if result is None else tuple(result.shape)
        raise ValueError(
            "Pooled-indexer prefill returned an unexpected top-k shape: "
            f"got {result_shape}, expected rows in {sorted(expected_rows)} "
            f"with width {output.shape[1]}"
        )
    # The following captured attention segment reads this stable padded buffer.
    output[:real_n].copy_(result[:real_n])
    output[real_n:].fill_(-1)


def _kpool_indexer_prefill_capture_stub(
    indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    output.fill_(-1)


bcg_kpool_indexer_prefill_with_output = eager_on_graph(
    True, capture_stub=_kpool_indexer_prefill_capture_stub
)(_kpool_indexer_prefill_with_output)
