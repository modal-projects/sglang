"""Page-table gathering for the bounded DFlash context cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_window_pages_kernel(
    ReqToToken,
    WindowIndices,
    PrefixLens,
    Bounds,
    Pages,
    TABLE_STRIDE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    window = tl.program_id(0)
    req = tl.load(WindowIndices + window * 2)
    batch_index = tl.load(WindowIndices + window * 2 + 1)
    if batch_index >= 0:
        prefix = tl.load(PrefixLens + batch_index).to(tl.int64)
        start = tl.maximum(0, prefix - WINDOW_SIZE)
        end = prefix + BLOCK_SIZE
        tl.store(Bounds + req * 2, start)
        tl.store(Bounds + req * 2 + 1, end)
    else:
        start = tl.load(Bounds + req * 2)
        end = tl.load(Bounds + req * 2 + 1)

    offsets = tl.arange(0, BLOCK)
    # Logical pages map to one virtual page, including partially filled ends.
    positions = tl.maximum(start, (start // PAGE_SIZE + offsets) * PAGE_SIZE)
    tokens = tl.load(
        ReqToToken + req * TABLE_STRIDE + positions,
        mask=(offsets < NUM_PAGES) & (positions < end),
        other=0,
    )
    tl.store(
        Pages + 1 + window * NUM_PAGES + offsets,
        tl.maximum(tokens // PAGE_SIZE, 0),
        mask=offsets < NUM_PAGES,
    )
    if window == 0:
        tl.store(Pages, 0)


def dflash_window_pages(
    req_to_token: torch.Tensor,
    window_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    bounds: torch.Tensor,
    *,
    window_size: int,
    block_size: int,
    page_size: int,
    max_block_size: int,
) -> torch.Tensor:
    """Update window bounds and gather virtual pages in one launch.

    ``window_indices`` is nonempty contiguous [windows, 2] integer metadata with
    unique request rows and their batch indices (-1 for requests absent from the
    current batch). ``bounds`` is caller-owned [request rows, 2] int64 storage. Every
    retained row must have been initialized by an earlier call on this stream.
    The current batch keeps [max(0, prefix-W), prefix+block); other windows keep
    their bounds but reread the request table after radix canonicalization.

    ``prefix_lens``, ``bounds`` and request-table rows must be contiguous; logical
    pages must map to single virtual pages. ``max_block_size`` covers all windows.
    The flat output includes padding and one guaranteed zero, so sorted unique
    pages can discard their first entry without a device-dependent mask.
    """
    num_pages = triton.cdiv(window_size + max_block_size + page_size - 1, page_size)
    pages = torch.empty(
        1 + window_indices.shape[0] * num_pages,
        dtype=torch.int64,
        device=req_to_token.device,
    )
    _dflash_window_pages_kernel[(window_indices.shape[0],)](
        req_to_token,
        window_indices,
        prefix_lens,
        bounds,
        pages,
        req_to_token.stride(0),
        window_size,
        block_size,
        page_size,
        num_pages,
        triton.next_power_of_2(num_pages),
    )
    return pages


@triton.jit(do_not_specialize=["NUM_VIRTUAL", "NUM_PAGES"])
def _dflash_unbound_full_pages_kernel(
    FullMap,
    DraftMap,
    Pages,
    Count,
    NUM_VIRTUAL,
    NUM_PAGES,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    full = tl.load(FullMap + offsets, offsets < NUM_VIRTUAL, other=-1)
    draft = tl.load(DraftMap + offsets, offsets < NUM_VIRTUAL, other=-1)
    unbound = (offsets > 0) & (full > 0) & (draft <= 0)
    base = tl.atomic_add(Count, tl.sum(unbound.to(tl.int32)), sem="relaxed")
    indices = base + tl.cumsum(unbound.to(tl.int32)) - 1
    tl.store(Pages + indices, offsets.to(tl.int64), unbound & (indices < NUM_PAGES))


def dflash_unbound_full_pages(
    full_map: torch.Tensor, draft_map: torch.Tensor, num_pages: int
) -> torch.Tensor:
    """Select a known count of Full pages without draft bindings.

    The maps share a virtual ID domain and remain stable during selection.
    The caller proves that at least ``num_pages`` Full-owned, draft-unbound
    pages exist from allocator counts. Returned IDs are unique; order is unspecified.
    """
    pages = torch.empty(num_pages, dtype=torch.int64, device=full_map.device)
    count = torch.zeros((), dtype=torch.int32, device=full_map.device)
    if num_pages:
        _dflash_unbound_full_pages_kernel[(triton.cdiv(full_map.numel(), 256),)](
            full_map, draft_map, pages, count, full_map.numel(), num_pages, 256
        )
    return pages


@triton.jit(do_not_specialize=["NUM_HELD"])
def _dflash_compact_window_pages_kernel(
    Held,
    VirtualToPhysical,
    Live,
    Count,
    NUM_HELD,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pages = tl.load(Held + offsets, offsets < NUM_HELD, other=0)
    physical = tl.load(VirtualToPhysical + pages, offsets < NUM_HELD, other=-1)
    keep = (offsets < NUM_HELD) & (physical > 0)
    base = tl.atomic_add(Count, tl.sum(keep.to(tl.int32)), sem="relaxed")
    indices = base + tl.cumsum(keep.to(tl.int32)) - 1
    tl.store(Live + indices, pages, keep)


def dflash_compact_window_pages(
    held: torch.Tensor, virtual_to_physical: torch.Tensor, num_live: int
) -> torch.Tensor:
    """Discard freed bindings using the allocator's known live count.

    ``held`` contains every allocated binding exactly once, plus possibly freed
    IDs. The caller supplies their exact live count; no count travels to the CPU.
    The output's order is unspecified.
    """
    live = torch.empty(num_live, dtype=held.dtype, device=held.device)
    count = torch.zeros((), dtype=torch.int32, device=held.device)
    if held.numel():
        _dflash_compact_window_pages_kernel[(triton.cdiv(held.numel(), 256),)](
            held, virtual_to_physical, live, count, held.numel(), 256
        )
    return live


@triton.jit(do_not_specialize=["Epoch", "NUM_PAGES"])
def _dflash_mark_window_pages_kernel(
    Pages,
    VirtualToPhysical,
    Epochs,
    Wanted,
    Missing,
    Counts,
    Epoch,
    NUM_PAGES,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pages = tl.load(Pages + offsets, offsets < NUM_PAGES, other=0)
    live = pages > 0
    previous_epoch = tl.atomic_xchg(Epochs + pages, Epoch, live, sem="relaxed")
    wanted = live & (previous_epoch != Epoch)
    physical = tl.load(VirtualToPhysical + pages, live, other=-1)
    missing = wanted & (physical <= 0)
    num_wanted = tl.sum(wanted.to(tl.int32))
    num_missing = tl.sum(missing.to(tl.int32))
    wanted_base = tl.atomic_add(Counts, num_wanted, sem="relaxed")
    missing_base = tl.atomic_add(Counts + 1, num_missing, sem="relaxed")
    wanted_index = wanted_base + tl.cumsum(wanted.to(tl.int32)) - 1
    missing_index = missing_base + tl.cumsum(missing.to(tl.int32)) - 1
    tl.store(Wanted + wanted_index, pages, wanted)
    tl.store(Missing + missing_index, pages, missing)


@triton.jit(do_not_specialize=["Epoch", "NUM_PREVIOUS"])
def _dflash_retired_window_pages_kernel(
    Previous,
    VirtualToPhysical,
    Epochs,
    Dead,
    Counts,
    Epoch,
    NUM_PREVIOUS,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offsets < NUM_PREVIOUS
    pages = tl.load(Previous + offsets, live, other=0)
    epoch = tl.load(Epochs + pages, live, other=0)
    physical = tl.load(VirtualToPhysical + pages, live, other=-1)
    dead = live & (epoch != Epoch) & (physical > 0)
    num_dead = tl.sum(dead.to(tl.int32))
    base = tl.atomic_add(Counts + 2, num_dead, sem="relaxed")
    indices = base + tl.cumsum(dead.to(tl.int32)) - 1
    tl.store(Dead + indices, offsets, dead)


def dflash_window_page_changes(
    pages: torch.Tensor,
    previous: torch.Tensor,
    virtual_to_physical: torch.Tensor,
    page_epochs: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan window ownership without separate dynamic-shape set operations.

    Inputs are contiguous int64 virtual page IDs; ``pages`` may contain zero
    padding and duplicates, whereas ``previous`` is unique and positive. Page
    IDs index ``virtual_to_physical`` and the caller-owned int64 ``page_epochs``.
    Epochs start at zero; the caller supplies a new positive epoch each call.

    Return wanted/missing page IDs, retired indices into ``previous``, and their
    three device counts. Retired pages are still bound, so they can be freed
    without another live filter or deduplication. Planning and allocation must
    be serialized on the same stream. Output order is unspecified.
    """
    wanted = torch.empty_like(pages)
    missing = torch.empty_like(pages)
    dead = torch.empty_like(previous)
    counts = torch.zeros(3, dtype=torch.int32, device=pages.device)
    _dflash_mark_window_pages_kernel[(triton.cdiv(pages.numel(), 256),)](
        pages,
        virtual_to_physical,
        page_epochs,
        wanted,
        missing,
        counts,
        epoch,
        pages.numel(),
        256,
    )
    if previous.numel():
        _dflash_retired_window_pages_kernel[(triton.cdiv(previous.numel(), 256),)](
            previous,
            virtual_to_physical,
            page_epochs,
            dead,
            counts,
            epoch,
            previous.numel(),
            256,
        )
    return wanted, missing, dead, counts


@triton.jit(do_not_specialize=["NUM_NEW", "NUM_HELD"])
def _dflash_recycle_window_pages_kernel(
    Held,
    Missing,
    Retired,
    Counts,
    VirtualToPhysical,
    PhysicalToVirtual,
    NUM_NEW,
    NUM_HELD,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    num_recycle = tl.load(Counts + 1) - NUM_NEW
    live = (offsets < num_recycle) & (offsets < NUM_HELD)
    slots = tl.load(Retired + offsets, live, other=0)
    old = tl.load(Held + slots, live, other=0)
    new = tl.load(Missing + NUM_NEW + offsets, live, other=0)
    physical = tl.load(VirtualToPhysical + old, live, other=0)
    tl.store(VirtualToPhysical + old, -1, live)
    tl.store(VirtualToPhysical + new, physical, live)
    tl.store(PhysicalToVirtual + physical, new, live)
    tl.store(Held + slots, new, live)


def dflash_recycle_window_pages(
    held: torch.Tensor,
    missing: torch.Tensor,
    retired: torch.Tensor,
    counts: torch.Tensor,
    virtual_to_physical: torch.Tensor,
    physical_to_virtual: torch.Tensor,
    num_new: int = 0,
) -> None:
    """Rebind retired physical pages; leave allocator ownership unchanged.

    ``held`` contains unique, bound pages. ``missing``, ``retired`` and ``counts``
    come from ``dflash_window_page_changes`` on the same stream. The first
    ``num_new`` missing pages are allocated separately; the remainder must fit
    in the retired slots. Update both maps and those slots of ``held`` in place.
    No KV copy is needed: retired contents are outside every active window.
    """
    if held.numel():
        _dflash_recycle_window_pages_kernel[(triton.cdiv(held.numel(), 256),)](
            held,
            missing,
            retired,
            counts,
            virtual_to_physical,
            physical_to_virtual,
            num_new,
            held.numel(),
            256,
        )


@triton.jit
def _dflash_rebind_pages(
    pages,
    Held,
    VirtualToPhysical,
    PhysicalToVirtual,
    Epochs,
    Missing,
    Epoch,
    NUM_HELD,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = pages > 0
    old_epoch = tl.atomic_xchg(Epochs + pages, Epoch, valid, sem="relaxed")
    physical = tl.load(VirtualToPhysical + pages, valid, other=-1)
    missing = valid & (old_epoch != Epoch) & (physical <= 0)
    missing_indices = tl.cumsum(missing.to(tl.int32)) - 1
    tl.store(Missing + missing_indices, pages, missing)
    num_missing = tl.sum(missing.to(tl.int32))
    # A single CTA owns the plan: finish marking and packing before examining
    # held pages, including shared pages marked by a different warp.
    tl.debug_barrier()
    old = tl.load(Held + offsets, offsets < NUM_HELD, other=0)
    old_epoch = tl.load(Epochs + old, offsets < NUM_HELD, other=Epoch)
    retired = (offsets < NUM_HELD) & (old_epoch != Epoch)
    retired_indices = tl.cumsum(retired.to(tl.int32)) - 1
    reuse = retired & (retired_indices < num_missing)
    new = tl.load(Missing + retired_indices, reuse, other=0)
    physical = tl.load(VirtualToPhysical + old, reuse, other=0)
    tl.store(VirtualToPhysical + old, -1, reuse)
    tl.store(VirtualToPhysical + new, physical, reuse)
    tl.store(PhysicalToVirtual + physical, new, reuse)
    tl.store(Held + offsets, new, reuse)


# Counts and the epoch change during serving; alignment specialization would
# compile new variants in the decode loop without improving this gather.
@triton.jit(do_not_specialize=["Epoch", "NUM_WINDOWS", "NUM_HELD"])
def _dflash_update_window_kernel(
    ReqToToken,
    WindowIndices,
    PrefixLens,
    Bounds,
    Held,
    VirtualToPhysical,
    PhysicalToVirtual,
    Epochs,
    Missing,
    Epoch,
    NUM_WINDOWS,
    NUM_HELD,
    TABLE_STRIDE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    window = offsets // NUM_PAGES
    page = offsets % NUM_PAGES
    valid = window < NUM_WINDOWS
    req = tl.load(WindowIndices + window * 2, valid, other=0)
    batch = tl.load(WindowIndices + window * 2 + 1, valid, other=-1)
    current = valid & (batch >= 0)
    prefix = tl.load(PrefixLens + batch, current, other=0).to(tl.int64)
    previous_start = tl.load(Bounds + req * 2, valid & ~current, other=0)
    previous_end = tl.load(Bounds + req * 2 + 1, valid & ~current, other=0)
    start = tl.where(current, tl.maximum(0, prefix - WINDOW_SIZE), previous_start)
    end = tl.where(current, prefix + BLOCK_SIZE, previous_end)
    tl.store(Bounds + req * 2, start, current & (page == 0))
    tl.store(Bounds + req * 2 + 1, end, current & (page == 0))
    position = tl.maximum(start, (start // PAGE_SIZE + page) * PAGE_SIZE)
    tokens = tl.load(
        ReqToToken + req * TABLE_STRIDE + position,
        valid & (position < end),
        other=0,
    )
    pages = tl.maximum(tokens // PAGE_SIZE, 0)
    _dflash_rebind_pages(
        pages,
        Held,
        VirtualToPhysical,
        PhysicalToVirtual,
        Epochs,
        Missing,
        Epoch,
        NUM_HELD,
        BLOCK,
    )


def dflash_update_window(
    req_to_token: torch.Tensor,
    window_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    bounds: torch.Tensor,
    held: torch.Tensor,
    virtual_to_physical: torch.Tensor,
    physical_to_virtual: torch.Tensor,
    page_epochs: torch.Tensor,
    epoch: int,
    *,
    window_size: int,
    block_size: int,
    page_size: int,
    max_block_size: int,
) -> None:
    """Gather and rebind a provisioned window in one launch for small geometries.

    Inputs obey ``dflash_window_pages`` and ``dflash_update_window_pages``.
    Held bindings must cover the maximum union size; no physical allocation or
    count readback occurs here. Large geometries use the same multi-CTA plan.
    """
    num_pages = triton.cdiv(window_size + max_block_size + page_size - 1, page_size)
    count = window_indices.shape[0] * num_pages
    block = triton.next_power_of_2(max(count, held.numel()))
    if block > 4096:
        pages = dflash_window_pages(
            req_to_token,
            window_indices,
            prefix_lens,
            bounds,
            window_size=window_size,
            block_size=block_size,
            page_size=page_size,
            max_block_size=max_block_size,
        )
        dflash_update_window_pages(
            pages, held, virtual_to_physical, physical_to_virtual, page_epochs, epoch
        )
        return
    missing = torch.empty(count, dtype=torch.int64, device=held.device)
    _dflash_update_window_kernel[(1,)](
        req_to_token,
        window_indices,
        prefix_lens,
        bounds,
        held,
        virtual_to_physical,
        physical_to_virtual,
        page_epochs,
        missing,
        epoch,
        window_indices.shape[0],
        held.numel(),
        req_to_token.stride(0),
        window_size,
        block_size,
        page_size,
        num_pages,
        block,
    )


def dflash_update_window_pages(
    pages: torch.Tensor,
    held: torch.Tensor,
    virtual_to_physical: torch.Tensor,
    physical_to_virtual: torch.Tensor,
    page_epochs: torch.Tensor,
    epoch: int,
) -> None:
    """Update a fully provisioned working set without host counts or allocation.

    Every held page must be bound and unique. Their count must cover the maximum
    possible unique nonzero pages in ``pages``. Epochs obey the page-planner
    contract. Only bindings change; physical ownership and contents stay put.
    """
    _, missing, retired, counts = dflash_window_page_changes(
        pages, held, virtual_to_physical, page_epochs, epoch
    )
    dflash_recycle_window_pages(
        held, missing, retired, counts, virtual_to_physical, physical_to_virtual
    )
