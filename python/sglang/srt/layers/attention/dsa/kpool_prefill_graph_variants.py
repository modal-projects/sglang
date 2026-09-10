from bisect import bisect_left


class KPoolPrefillGraphVariants:
    def __init__(self, specification, capture_sizes, dsa_variants, max_requests=32):
        self.capture_sizes = sorted(capture_sizes)
        self.dsa_variants = dsa_variants
        self.max_requests = max_requests
        self.plans = {}
        variants = set()
        for item in specification.split(","):
            tokens, context, history = map(int, item.split(":"))
            if not (
                1 <= tokens <= self.capture_sizes[-1]
                and 1 <= context <= 1048576
                and history > 0
                and history <= max_requests * 262144
                and history % 64 == 0
                and history >= ((context // 4 + 63) // 64) * 64
            ):
                raise ValueError(f"Invalid KPool prefill capture variant: {item}")
            if tokens <= dsa_variants.max_tokens and self.dsa_variant((tokens, context, history)) is None:
                raise ValueError(f"KPool capture requires a matching DSA variant: {item}")
            variants.add((tokens, context, history))
        self.variants = sorted(variants)
        if len(self.variants) > 64 or sum(
            history * 132 + max_requests * context * 4
            for _, context, history in self.variants
        ) > 2 * 1024**3:
            raise ValueError("KPool variants exceed 64 graphs or 2 GiB of persistent buffers")

    def bucket(self, variant):
        return self.capture_sizes[bisect_left(self.capture_sizes, variant[0])]

    def dsa_variant(self, variant):
        return self.dsa_variants.resolve(variant[0], variant[1])

    @staticmethod
    def label(variant):
        return "kpool-" + "-".join(map(str, variant))

    def resolve(self, forward_batch):
        if (
            not forward_batch.forward_mode.is_extend_without_speculative()
            or forward_batch.seq_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
            or not 0 < forward_batch.batch_size <= self.max_requests
        ):
            return None
        lengths = forward_batch.seq_lens_cpu.tolist()
        extends = forward_batch.extend_seq_lens_cpu
        if len(lengths) != len(extends) or any(n <= 0 or n > s for n, s in zip(extends, lengths)):
            return None
        tokens = sum(extends)
        if not 0 < tokens <= len(forward_batch.input_ids):
            return None
        bucket_index = bisect_left(self.capture_sizes, tokens)
        if bucket_index == len(self.capture_sizes):
            return None
        bucket = self.capture_sizes[bucket_index]
        context = max(lengths)
        history = sum(((s // 4 + 63) // 64) * 64 for s in lengths)
        dsa = self.dsa_variants.resolve(tokens, context)
        for variant in self.variants:
            n, max_context, max_history = variant
            if (
                (tokens == n or self.dsa_variants.max_tokens < tokens <= n)
                and self.bucket(variant) == bucket
                and context <= max_context
                and history <= max_history
                and (context <= 2048) == (max_context <= 2048)
                and dsa == self.dsa_variant(variant)
            ):
                return variant
        return None

    def prepare(self, variant, forward_batch, metadata, device, mapping_mode):
        from sglang.srt.layers.attention.dsa.kpool_prefill_graph_plan import KPoolPrefillGraphPlan

        if variant not in self.variants:
            raise ValueError("Unknown KPool capture variant")
        if variant not in self.plans:
            self.plans[variant] = KPoolPrefillGraphPlan(
                variant[0], self.max_requests, variant[2], device,
                max_context=variant[1], mapping_mode=mapping_mode,
            )
        plan = self.plans[variant]
        if plan.mapping_mode != mapping_mode:
            raise ValueError("KPool capture mapping changed")
        if forward_batch.kpool_prefill_graph_capture:
            plan.clear()
        else:
            if self.resolve(forward_batch) != variant:
                raise ValueError("Live KPool batch does not match selected graph")
            plan.update(
                metadata.kpool_extend_plan, metadata.topk_indices_offset,
                req_pool_indices=forward_batch.req_pool_indices,
                extend_seq_lens=forward_batch.extend_seq_lens_cpu,
                max_seq_len=int(forward_batch.seq_lens_cpu.max()),
            )
        return plan
