"""Reference for raw feature capture used by CPU ownership tests."""

import torch


def capture_dflash_features_reference(hidden, starts, commit_lens, steps, indices):
    ends = torch.full_like(starts, -1)
    if steps is not None:
        valid = (steps >= 0) & (indices >= 0)
        ends[valid] = starts[valid] + steps[valid] + 1
    return hidden.detach().clone(), torch.stack(
        (starts, commit_lens, ends), dim=1
    ).long()
