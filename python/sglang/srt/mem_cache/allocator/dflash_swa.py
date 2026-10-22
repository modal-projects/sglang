"""Bounded draft SWA beside a Full + recurrent target in the unified arena."""

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.utils.common import get_num_new_pages


class DFlashSWATokenToKVPoolAllocator(UnifiedMambaSWATokenToKVPoolAllocator):
    """Target allocation creates Full pages; the worker binds live draft windows.

    Fixed admission caps reserve the peak draft working set and recurrent state
    independently. They share the existing arena, page maps and compaction, but
    unused reservations are not lent to the target in this initial policy.
    """

    @property
    def size(self):
        # Mamba accounting pairs this total with available_size(). A live SWA
        # region can split free space without consuming the fixed Full budget.
        return self.current_full_capacity

    @size.setter
    def size(self, value):
        pass

    def available_size(self):
        return self.full_available_size()

    def ensure_capacity(self, full_tokens, swa_tokens):
        if (
            full_tokens > self._conserve_full_available_size()
            or swa_tokens > self._conserve_swa_available_size()
        ):
            return False
        return super().ensure_capacity(full_tokens, swa_tokens)

    def alloc(self, need_size):
        if not self.ensure_capacity(need_size, 0):
            return None
        return self.full_attn_allocator.alloc(need_size)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
    ):
        result = self._extend_in_virtual_space(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_swa_pages=0,
        )
        return None if result is None else result[0]

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        need = (
            get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            * self.page_size
        )
        if not self.ensure_capacity(need, 0):
            return None
        return self.full_attn_allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    def _release_page_reps(self, both, full_only):
        # Draft bindings outlive freed target pages as spares in the fixed
        # reservation; finding the bound reps would wait for the in-flight forward.
        pieces = (*both, *full_only)
        if pieces:
            reps = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            self.full_attn_allocator.free(reps, _pages=reps // self.page_size)
            self.full_attn_allocator.clear_inverse_history()

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        return BaseTokenToKVPoolAllocator.create_prefill_budget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def evict_to_free_tokens(self, tree_cache, num_tokens):
        if self.free_group is not None:
            self.flush_deferred_full_frees()
        BaseTokenToKVPoolAllocator.evict_to_free_tokens(self, tree_cache, num_tokens)
        if self.free_group is not None:
            self.flush_deferred_full_frees()
        return self.ensure_capacity(num_tokens, 0)

    def mamba_slot_full_token_cost(self):
        # Recurrent slots already have a separate, bounded arena reservation.
        return 0

    def mamba_full_cache_donor(self):
        return None
