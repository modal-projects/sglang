import unittest

import torch

from sglang.kernels.ops.attention.dflash_window import (
    dflash_compact_window_pages,
    dflash_recycle_window_pages,
    dflash_unbound_full_pages,
    dflash_update_window,
    dflash_update_window_pages,
    dflash_window_page_changes,
    dflash_window_pages,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.dflash_window import (
    dflash_window_page_changes_reference,
    dflash_window_pages_reference,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestDFlashWindowPages(CustomTestCase):
    def test_spare_selection_uses_only_full_owned_unbound_pages(self):
        for length in (1, 1089, 65537):
            full = torch.arange(length, device="cuda")
            full[::7] = -1
            draft = torch.full_like(full, -1)
            draft[1::3] = 42
            candidates = torch.where((full > 0) & (draft <= 0))[0]
            for count in sorted({0, min(1, candidates.numel()), candidates.numel()}):
                with self.subTest(length=length, count=count):
                    pages = dflash_unbound_full_pages(full, draft, count)
                    self.assertEqual(pages.unique().numel(), count)
                    self.assertTrue((full[pages] > 0).all().item())
                    self.assertTrue((draft[pages] <= 0).all().item())

    def test_fused_update_handles_shared_and_retained_windows(self):
        for rows, window, page_size in ((2, 8, 4), (32, 128, 1)):
            with self.subTest(rows=rows, window=window, page_size=page_size):
                per_window = (window + 2 + 2 * page_size - 2) // page_size
                held = torch.arange(60000, 60000 + rows * per_window, device="cuda")
                mapping = torch.full((65536,), -1, device="cuda", dtype=torch.int64)
                inverse = torch.full_like(mapping, -1)
                physical = torch.arange(10000, 10000 + held.numel(), device="cuda")
                mapping[held] = physical
                inverse[physical] = held
                epochs = torch.zeros_like(mapping)
                positions = torch.arange(512)
                table = (
                    torch.arange(rows)[:, None] * 512 + positions // page_size + 1
                ) * page_size + positions % page_size
                table[1, :256] = table[0, :256]
                bounds = torch.zeros((rows, 2), dtype=torch.int64)
                bounds_gpu = bounds.cuda()
                for epoch in range(1, 5):
                    prefix = torch.full((rows,), 160 + epoch * 7, dtype=torch.int64)
                    indices = [
                        (i, i if epoch == 1 or i == 0 else -1) for i in range(rows)
                    ]
                    indices = torch.tensor(indices, dtype=torch.int64)
                    if epoch == 3:
                        # Canonicalization also affects an unscheduled window.
                        table[1] += 20000 * page_size
                    kwargs = dict(
                        window_size=window,
                        block_size=2,
                        page_size=page_size,
                        max_block_size=2,
                    )
                    wanted = dflash_window_pages_reference(
                        table, indices, prefix, bounds, **kwargs
                    ).unique()
                    wanted = wanted[wanted > 0].cuda()
                    before = mapping.clone()
                    retained = wanted[before[wanted] > 0]
                    dflash_update_window(
                        table.cuda(),
                        indices.cuda(),
                        prefix.cuda(),
                        bounds_gpu,
                        held,
                        mapping,
                        inverse,
                        epochs,
                        epoch,
                        **kwargs,
                    )
                    self.assertTrue((mapping[wanted] > 0).all().item())
                    torch.testing.assert_close(mapping[retained], before[retained])
                    torch.testing.assert_close(mapping[held].sort().values, physical)
                    torch.testing.assert_close(inverse[mapping[held]], held)
                    torch.testing.assert_close(bounds_gpu.cpu(), bounds)

    def test_compaction_uses_live_bindings_after_target_frees(self):
        for length in (0, 1, 1089, 65537):
            for stride in (1, 3, 7):
                with self.subTest(length=length, stride=stride):
                    held = torch.arange(1, length + 1, device="cuda")
                    mapping = torch.arange(length + 2, device="cuda")
                    mapping[::stride] = -1
                    expected = held[mapping[held] > 0]
                    actual = dflash_compact_window_pages(
                        held, mapping, expected.numel()
                    )
                    torch.testing.assert_close(actual.sort().values, expected)

    def test_page_changes_preserve_sharing_and_skip_freed_mappings(self):
        for length in (1, 35, 1089, 65537):
            for old_length in (0, 33, 1030):
                with self.subTest(length=length, old_length=old_length):
                    # Duplicates model shared prefixes; old mappings may already
                    # have been freed or tombstoned by target canonicalization.
                    generator = torch.Generator().manual_seed(42)
                    pages = torch.randint(0, 2000, (length,), generator=generator)
                    previous = (
                        torch.randperm(1999, generator=generator)[:old_length] + 1
                    )
                    mapping = torch.arange(2000)
                    mapping[::5] = -1
                    mapping[1::7] = 0
                    expected = dflash_window_page_changes_reference(
                        pages, previous, mapping
                    )
                    *buffers, counts = dflash_window_page_changes(
                        pages.cuda(),
                        previous.cuda(),
                        mapping.cuda(),
                        torch.zeros_like(mapping, device="cuda"),
                        1,
                    )
                    torch.testing.assert_close(counts.cpu(), expected[-1])
                    for buffer, count, reference in zip(
                        buffers, counts.tolist(), expected[:3], strict=True
                    ):
                        torch.testing.assert_close(
                            buffer[:count].cpu().sort().values, reference
                        )

    def test_recycling_preserves_active_pages_and_physical_ownership(self):
        held = torch.arange(1, 35, device="cuda")
        mapping = torch.full((512,), -1, dtype=torch.int64, device="cuda")
        inverse = torch.full_like(mapping, -1)
        physical = torch.arange(100, 134, device="cuda")
        mapping[held] = physical
        inverse[physical] = held
        epochs = torch.zeros_like(mapping)
        for epoch in range(1, 40):
            # Alternate partial-page alignments, retaining spare retired pages.
            # Repeated IDs also exercise cross-block sharing in the mark kernel.
            wanted = torch.arange(epoch * 3, epoch * 3 + 33 + epoch % 2, device="cuda")
            pages = wanted.repeat(257 if epoch % 3 == 0 else 19)
            before = mapping.clone()
            previous = held.clone()
            if epoch % 2:
                dflash_update_window_pages(pages, held, mapping, inverse, epochs, epoch)
            else:
                _, missing, retired, counts = dflash_window_page_changes(
                    pages, held, mapping, epochs, epoch
                )
                dflash_recycle_window_pages(
                    held, missing, retired, counts, mapping, inverse
                )
            self.assertTrue((mapping[wanted] > 0).all().item())
            self.assertEqual(held.unique().numel(), 34)
            torch.testing.assert_close(mapping[held].sort().values, physical)
            torch.testing.assert_close(inverse[mapping[held]], held)
            retained = previous[torch.isin(previous, wanted)]
            torch.testing.assert_close(mapping[retained], before[retained])
            # Unified compaction may move physical pages between forwards.
            # Rebinding must follow the current map, not cached physical IDs.
            if epoch % 5 == 0:
                mapping[held] = mapping[held].roll(1)
                inverse[mapping[held]] = held

    def test_page_boundaries_and_retained_window_remapping(self):
        # A page-wise gather must match every token in the interval, including
        # short prefixes, partial end pages, shared pages and inactive windows.
        for page_size in (1, 4, 64):
            for window_size in (1, 8, 2048):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(page=page_size, window=window_size, dtype=dtype):
                        self._check_windows(page_size, window_size, dtype)

    def _check_windows(self, page_size, window_size, dtype):
        rows, length = 37, 4352
        positions = torch.arange(length + 23)
        # Permuted virtual pages with sharing across requests. The int64 case
        # also catches accidental narrowing of virtual token/page IDs.
        page_ids = (
            positions // page_size * 17 + torch.arange(rows)[:, None] * 31
        ) % 997 + 1
        if dtype == torch.int64:
            page_ids += 2**33
        table = (page_ids * page_size + positions % page_size).to(dtype)[:, :length]
        table_gpu = torch.empty((rows, length + 23), dtype=dtype, device="cuda")
        table_gpu = table_gpu[:, :length]
        table_gpu.copy_(table)
        bounds = torch.full((rows, 2), -1, dtype=torch.int64)
        bounds_gpu = bounds.cuda()
        request_rows = list(range(1, 33))
        boundaries = [
            0,
            1,
            page_size - 1,
            page_size,
            page_size + 1,
            window_size - 1,
            window_size,
            window_size + 1,
            4095,
        ]
        prefix = torch.tensor(
            [boundaries[i % len(boundaries)] for i in range(32)], dtype=dtype
        )

        def check(indices, prefix, block_size, max_block_size):
            indices = torch.tensor(indices, dtype=torch.int64)
            kwargs = dict(
                window_size=window_size,
                block_size=block_size,
                page_size=page_size,
                max_block_size=max_block_size,
            )
            expected = dflash_window_pages_reference(
                table, indices, prefix, bounds, **kwargs
            )
            actual = dflash_window_pages(
                table_gpu, indices.cuda(), prefix.cuda(), bounds_gpu, **kwargs
            )
            torch.testing.assert_close(
                torch.unique(actual).cpu(), torch.unique(expected)
            )
            torch.testing.assert_close(bounds_gpu.cpu(), bounds)
            self.assertEqual(actual[0].item(), 0)

        check(list(zip(request_rows, range(32))), prefix, 16, 16)
        # Repoint a retained window to canonical pages without scheduling it.
        table[2, :4096] = table[3, :4096]
        table_gpu.copy_(table)
        check(
            [(row, 0 if row == 1 else -1) for row in request_rows],
            torch.tensor([4096], dtype=dtype),
            0,
            16,
        )
        # Recycled request rows must replace their old bounds, not inherit them.
        check([(2, 0)], torch.tensor([1], dtype=dtype), 0, 0)


if __name__ == "__main__":
    unittest.main()
