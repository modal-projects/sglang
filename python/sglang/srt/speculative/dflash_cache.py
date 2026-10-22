"""Sizing and residency of the optional, paged DFlash context cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from sglang.kernels.ops.attention.dflash_window import (
    dflash_compact_window_pages,
    dflash_unbound_full_pages,
    dflash_update_window,
)
from sglang.srt.speculative.dflash_feature_tail import (
    CONTEXT_CHUNK_SIZE,
    DFlashContextLayout,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
    from sglang.srt.mem_cache.allocator.dflash_swa import (
        DFlashSWATokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.dflash_feature_tail import (
        DFlashContextCapture,
        DFlashContextState,
    )


@dataclass(frozen=True)
class DFlashCacheSizing:
    window_size: int
    block_size: int
    max_requests: int
    feature_size: int
    feature_dtype: torch.dtype
    kv_heads: int
    head_dim: int
    num_layers: int
    kv_dtype: torch.dtype
    page_size: int
    prefill_tokens: int
    overlap: bool
    tap_cache: bool = False

    @property
    def context_layout(self) -> DFlashContextLayout:
        return DFlashContextLayout(
            self.num_layers,
            self.kv_heads,
            self.head_dim,
            self.kv_dtype,
            self.feature_size if self.tap_cache else 0,
            self.feature_dtype,
        )

    @property
    def checkpoint_window_bytes(self) -> int:
        return self.window_size * self.context_layout.row_bytes

    @property
    def draft_tokens(self) -> int:
        # Both ends of a window can cut a page. The block includes the anchor.
        return self.max_requests * ceil_align(
            self.window_size + self.block_size + self.page_size - 1, self.page_size
        )

    @property
    def draft_bytes(self) -> int:
        return (
            self.draft_tokens
            * self.num_layers
            * self.kv_heads
            * self.head_dim
            * 2
            * self.kv_dtype.itemsize
        )

    @property
    def raw_window_bytes(self) -> int:
        return self.window_size * self.feature_size * self.feature_dtype.itemsize

    @property
    def max_pending_transfers(self) -> int:
        return 2 * self.max_requests

    @property
    def context_gpu_reserve_bytes(self) -> int:
        # Each request owns one live window and at most two checkpoints. These
        # share chunks, but disjoint checkpoint windows are valid too. A tail
        # can retain unused rows in its two boundary chunks. The transfer queue
        # independently caps leases after request/checkpoint ownership ends.
        chunk = min(self.window_size, CONTEXT_CHUNK_SIZE)
        owned_rows = (3 * self.max_requests + self.max_pending_transfers) * (
            self.window_size + 2 * (chunk - 1)
        )
        # A prefill capture contains only new rows from its forward, capped by
        # the checkpoint + final windows. Remapping reuses this same capture.
        prefill_rows = min(
            self.prefill_tokens or 2 * self.max_requests * self.window_size,
            2 * self.max_requests * self.window_size,
        )
        # Overlap can hold two queued captures plus one retired source whose
        # result-stream copy has not finished. The next forward waits for that
        # stream, so result completion drains it before another can retire.
        staged_rows = (3 if self.overlap else 1) * max(
            prefill_rows, self.max_requests * self.block_size
        )
        # Revision recovery can hold joined, projected and rechunked windows.
        # Prefill projects both checkpoint and final windows in one batch.
        projection_rows = max(
            3 * self.window_size,
            prefill_rows,
        )
        return (
            (owned_rows + staged_rows + projection_rows) * self.context_layout.row_bytes
            # Selected target rows are temporary projection inputs, including
            # when retaining raw taps is disabled.
            + prefill_rows * self.feature_size * self.feature_dtype.itemsize
        )


@dataclass
class _DraftWindow:
    kv: ReqKvInfo
    req_pool_idx: int
    block_size: int
    feature_state: DFlashContextState | None = None


class DFlashCacheResidency:
    """Keep active windows in a bounded, reusable set of SWA page bindings.

    The request table remains authoritative, including canonicalization by the
    radix tree. Cached checkpoints own immutable context windows separately.
    Retired bindings, including those whose target page was freed, can supply
    the next window without changing physical ownership: a binding never
    implies materialized content.
    """

    def __init__(
        self,
        allocator: DFlashSWATokenToKVPoolAllocator,
        req_to_token_pool: ReqToTokenPool,
        window_size: int,
    ):
        self.allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.window_size = window_size
        self._windows: dict[int, _DraftWindow] = {}
        self._pages = torch.empty(0, dtype=torch.int64, device=allocator.device)
        self._page_epochs = torch.zeros_like(
            allocator.swa_attn_allocator.virtual_to_physical
        )
        self._epoch = 0
        self._indices_key = ()
        self._window_indices = None
        self._bounds = torch.empty(
            (req_to_token_pool.req_to_token.shape[0], 2),
            dtype=torch.int64,
            device=allocator.device,
        )

    def clear(self) -> None:
        self._windows.clear()
        self._pages = self._pages[:0]
        self._indices_key = ()
        self._window_indices = None

    def prepare(
        self,
        batch: ScheduleBatch,
        *,
        prefix_lens: torch.Tensor | None = None,
        block_size: int = 0,
        restore: Callable,
    ) -> None:
        self._windows = {
            key: w
            for key, w in self._windows.items()
            if w.kv.holds_kv
            and w.kv.req_pool_idx == w.req_pool_idx
            and (
                w.feature_state is None or w.kv.dflash_context_state is w.feature_state
            )
        }
        batch_indices = {id(req.kv): i for i, req in enumerate(batch.reqs)}
        if prefix_lens is None:
            prefix_lens = batch.seq_lens
        for req in batch.reqs:
            old = self._windows.get(id(req.kv))
            if old is None:
                self._windows[id(req.kv)] = _DraftWindow(
                    req.kv, req.kv.req_pool_idx, block_size, req.kv.dflash_context_state
                )
            else:
                old.block_size = block_size

        table = self.req_to_token_pool.req_to_token
        ps = self.allocator.page_size
        indices_key = tuple(
            (w.req_pool_idx, batch_indices.get(key, -1))
            for key, w in self._windows.items()
        )
        if indices_key != self._indices_key:
            self._window_indices = torch.tensor(
                indices_key, dtype=torch.int64, pin_memory=table.is_cuda
            ).to(table.device, non_blocking=True)
            self._indices_key = indices_key
        max_block_size = max(w.block_size for w in self._windows.values())
        swa = self.allocator.swa_attn_allocator
        # Only this worker allocates SWA pages. Target frees can remove held
        # bindings; compaction preserves their identities and live count.
        num_live = swa.allocated_count() // ps
        if self._pages.numel() != num_live:
            self._pages = dflash_compact_window_pages(
                self._pages, swa.virtual_to_physical, num_live
            )
        self._epoch += 1
        max_pages = 0
        for key, w in self._windows.items():
            bound = ceil_align(self.window_size + w.block_size + ps - 1, ps) // ps
            if block_size and key in batch_indices:
                # Host commit metadata can lag one overlap iteration; the
                # allocated end already covers its outcome and the next block.
                # These bounds often avoid waiting for a never-needed spare page
                # while a newly admitted window has not crossed its first page.
                start = max(0, w.kv.kv_committed_len - self.window_size)
                bound = min(
                    bound, ceil_align(w.kv.kv_allocated_len, ps) // ps - start // ps
                )
            max_pages += bound
        full = self.allocator.full_attn_allocator
        # Every wanted virtual page is Full-owned. Provision enough bindings for
        # the window bound (or every Full page, if fewer exist), using the fixed
        # draft reservation. Spare bindings avoid GPU count readbacks on growth.
        capacity = min(max_pages, full.allocated_count() // ps)
        num_new = max(0, capacity - self._pages.numel())
        if num_new:
            if not self.allocator.ensure_capacity(0, num_new * ps):
                raise RuntimeError(
                    "DFlash active windows exceeded the reserved SWA budget"
                )
            new_pages = dflash_unbound_full_pages(
                full.virtual_to_physical, swa.virtual_to_physical, num_new
            )
            swa.alloc_with_virtual(new_pages)
            self._pages = torch.cat((self._pages, new_pages))
        dflash_update_window(
            table,
            self._window_indices,
            prefix_lens,
            self._bounds,
            self._pages,
            swa.virtual_to_physical,
            swa.physical_to_virtual,
            self._page_epochs,
            self._epoch,
            window_size=self.window_size,
            block_size=block_size,
            page_size=ps,
            max_block_size=max_block_size,
        )
        swa.clear_inverse_history()

        # Canonical target pages may have no draft contents. Restore saved KV;
        # a draft revision change instead rebuilds the live window from its taps.
        revision = self.allocator.dflash_checkpoint_pool.revision
        for w in self._windows.values():
            state = w.kv.dflash_context_state
            remap = state.take_canonical_remap() if state is not None else None
            stale = state is not None and state.revision != revision
            tail = state.snapshot() if remap is not None or stale else None
            if tail is None:
                continue
            start = tail.start_pos if stale else max(remap[0], tail.start_pos)
            end = tail.end_pos if stale else min(remap[1], tail.end_pos)
            if start >= end:
                continue
            positions = torch.arange(start, end, device=table.device)
            virtual = table[w.req_pool_idx, start:end]
            keep = swa.virtual_to_physical[virtual.long() // ps] > 0
            if stale or keep.any():
                updated = restore(
                    tail=tail,
                    cache_loc=self.allocator.translate_loc_from_full_to_swa(
                        virtual[keep]
                    ),
                    positions=positions[keep],
                )
                if stale:
                    state.replace_live_window(updated)

    def retain_prefill(
        self, batch: ScheduleBatch, capture: DFlashContextCapture
    ) -> None:
        # Tie each window to the feature state its prefill created.
        for req, entry in zip(batch.reqs, capture.entries, strict=True):
            self._windows[id(req.kv)].feature_state = (
                entry.owner if entry is not None else None
            )
