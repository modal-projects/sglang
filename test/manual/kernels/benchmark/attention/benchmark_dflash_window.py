"""Compare wall time for token-wise and fused DFlash window page gathering."""

import argparse
import json
import time
from pathlib import Path

import torch

from sglang.kernels.ops.attention.dflash_window import dflash_window_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    window, block, page, length = 2048, 16, 64, 8192
    results = []
    for batch_size in (1, 8, 32):
        table = torch.arange(
            page, batch_size * length + page, dtype=torch.int64, device="cuda"
        ).view(batch_size, length)
        prefixes = 4096 + torch.arange(batch_size, device="cuda") * 7
        bounds = torch.empty((batch_size, 2), dtype=torch.int64, device="cuda")

        def tokenwise():
            positions = [
                (prefixes[i] + torch.arange(-window, block, device="cuda")).clamp_min_(
                    0
                )
                for i in range(batch_size)
            ]
            pages = torch.unique(
                torch.cat([table[i, pos] // page for i, pos in enumerate(positions)])
            )
            return pages[pages > 0]

        def fused():
            indices = torch.tensor(
                [(i, i) for i in range(batch_size)],
                dtype=torch.int64,
                pin_memory=True,
            ).to("cuda", non_blocking=True)
            pages = dflash_window_pages(
                table,
                indices,
                prefixes,
                bounds,
                window_size=window,
                block_size=block,
                page_size=page,
                max_block_size=block,
            )
            return torch.unique(pages)[1:]

        torch.testing.assert_close(tokenwise(), fused())
        row = {"batch_size": batch_size}
        for name, fn in (("tokenwise", tokenwise), ("fused", fused)):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                fn()
            torch.cuda.synchronize()
            row[name + "_wall_us"] = (
                (time.perf_counter() - start) * 1e6 / args.iterations
            )
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
