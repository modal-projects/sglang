from bisect import bisect_left


def context_bound(max_seq_len):
    return min(2052, ((max_seq_len + 127) // 128) * 128)


class DSAPrefillGraphVariants:
    def __init__(self, specification, capture_sizes, sm_count, num_heads):
        self.capture_sizes = sorted(capture_sizes)
        self.max_tokens = self.capture_sizes[
            min(
                len(self.capture_sizes) - 1,
                bisect_left(self.capture_sizes, sm_count // 2),
            )
        ]
        import flashinfer

        if (sm_count, num_heads, flashinfer.__version__) != (148, 16, "0.6.17"):
            raise ValueError(
                "DSA prefill graph variants require 148 SMs, 16 query heads, "
                "and FlashInfer 0.6.17"
            )
        if specification == "all":
            from sglang.srt.layers.attention.dsa.trtllm_prefill_graph_plans import (
                CONTEXT_BOUNDS,
                REPRESENTATIVES,
            )

            if self.max_tokens > len(REPRESENTATIVES):
                raise ValueError("DSA variant map covers at most 80 live tokens")
            if len(REPRESENTATIVES) != 80 or any(
                len(row) != len(CONTEXT_BOUNDS)
                or any(
                    representative not in CONTEXT_BOUNDS
                    or representative < bound
                    or row[CONTEXT_BOUNDS.index(representative)] != representative
                    for bound, representative in zip(CONTEXT_BOUNDS, row)
                )
                for row in REPRESENTATIVES
            ):
                raise ValueError("Invalid validated DSA variant map")
            self.lookup = {
                (tokens, bound): (tokens, representative)
                for tokens, row in enumerate(
                    REPRESENTATIVES[: self.max_tokens], start=1
                )
                for bound, representative in zip(CONTEXT_BOUNDS, row)
            }
            self.variants = sorted(set(self.lookup.values()))
            return
        self.variants = set()
        for item in specification.split(","):
            tokens, bound = map(int, item.split(":"))
            if not 1 <= tokens <= self.max_tokens or bound not in (
                *range(128, 2049, 128),
                2052,
            ):
                raise ValueError(f"Invalid DSA prefill graph variant: {item}")
            self.variants.add((tokens, bound))
        self.variants = sorted(self.variants)
        self.lookup = {variant: variant for variant in self.variants}

    def bucket(self, tokens):
        return self.capture_sizes[bisect_left(self.capture_sizes, tokens)]

    def resolve(self, tokens, max_seq_len):
        return self.lookup.get((tokens, context_bound(max_seq_len)))

    @staticmethod
    def label(variant):
        return f"prefill-{variant[0]}-{variant[1]}" if variant is not None else None
