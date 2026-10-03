"""Apply the configured top-k/top-p order with backend renormalization ops."""

from functools import partial
from typing import Callable, Optional

import torch


def renorm_top_p(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    top_p_renorm: Callable,
) -> torch.Tensor:
    """Keep disabled top-p rows unchanged in heterogeneous sampling batches.

    A normalized float32 row can sum slightly above one. Some backend top-p
    renormalizers then discard its tail at p=1, even though top-p is disabled
    for that request. Preserve the input without a device synchronization.
    """
    filtered = top_p_renorm(probs, top_ps)
    return torch.where(top_ps.reshape(-1, 1) >= 1.0, probs, filtered)


def renorm_top_k_top_p(
    probs: torch.Tensor,
    top_ks: Optional[torch.Tensor],
    top_ps: Optional[torch.Tensor],
    filter_apply_order: str,
    *,
    top_k_renorm: Callable,
    top_p_renorm: Callable,
) -> torch.Tensor:
    filters = (
        (top_k_renorm, top_ks),
        (partial(renorm_top_p, top_p_renorm=top_p_renorm), top_ps),
    )
    if filter_apply_order == "joint":
        # Top-p must measure mass on the full distribution before top-k.
        filters = filters[::-1]
    for renorm, threshold in filters:
        if threshold is not None:
            probs = renorm(probs, threshold)
    return probs
