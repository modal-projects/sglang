"""Token-wise reference for the DFlash window page gather."""

import torch


def dflash_window_pages_reference(
    req_to_token,
    window_indices,
    prefix_lens,
    bounds,
    *,
    window_size,
    block_size,
    page_size,
    max_block_size,
):
    pages = [torch.zeros(1, dtype=torch.int64)]
    for req, batch_index in window_indices.tolist():
        if batch_index >= 0:
            prefix = int(prefix_lens[batch_index])
            bounds[req] = torch.tensor(
                [max(0, prefix - window_size), prefix + block_size]
            )
        start, end = bounds[req].tolist()
        pages.append((req_to_token[req, start:end].long() // page_size).clamp_min(0))
    return torch.cat(pages)


def dflash_window_page_changes_reference(
    pages, previous, virtual_to_physical, page_epochs=None, epoch=None
):
    wanted = sorted(set(pages.tolist()) - {0})
    missing = [p for p in wanted if virtual_to_physical[p] <= 0]
    dead = [
        i
        for i, p in enumerate(previous.tolist())
        if p not in wanted and virtual_to_physical[p] > 0
    ]
    return (
        torch.tensor(wanted, dtype=torch.int64),
        torch.tensor(missing, dtype=torch.int64),
        torch.tensor(dead, dtype=torch.int64),
        torch.tensor([len(wanted), len(missing), len(dead)], dtype=torch.int32),
    )


def dflash_recycle_window_pages_reference(
    held, missing, retired, counts, virtual_to_physical, physical_to_virtual, num_new=0
):
    for i in range(int(counts[1]) - num_new):
        slot = int(retired[i])
        old, new = int(held[slot]), int(missing[num_new + i])
        physical = int(virtual_to_physical[old])
        virtual_to_physical[old] = -1
        virtual_to_physical[new] = physical
        physical_to_virtual[physical] = new
        held[slot] = new


def dflash_update_window_pages_reference(pages, held, mapping, inverse, epochs, epoch):
    _, missing, retired, counts = dflash_window_page_changes_reference(
        pages, held, mapping
    )
    dflash_recycle_window_pages_reference(
        held, missing, retired, counts, mapping, inverse
    )


def dflash_update_window_reference(
    table, indices, prefix, bounds, held, mapping, inverse, epochs, epoch, **kwargs
):
    pages = dflash_window_pages_reference(table, indices, prefix, bounds, **kwargs)
    dflash_update_window_pages_reference(pages, held, mapping, inverse, epochs, epoch)


def dflash_unbound_full_pages_reference(full_map, draft_map, num_pages):
    return torch.where((full_map > 0) & (draft_map <= 0))[0][:num_pages]
