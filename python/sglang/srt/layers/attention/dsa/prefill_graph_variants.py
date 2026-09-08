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
        if num_heads != 16:
            raise ValueError(
                "DSA prefill graph variants currently require 16 query heads"
            )
        if specification == "all":
            raise ValueError(
                "Full DSA variant capture requires a validated launch-plan map"
            )
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
