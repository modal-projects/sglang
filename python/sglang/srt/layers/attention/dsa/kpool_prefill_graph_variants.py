from bisect import bisect_left


def _all_variants(capture_sizes, dsa_variants, max_requests):
    from sglang.srt.layers.attention.dsa.trtllm_prefill_graph_plans import (
        CONTEXT_BOUNDS,
        REPRESENTATIVES,
    )

    if capture_sizes[-1] > 4096 or not 0 < dsa_variants.max_tokens <= 80:
        raise ValueError("Full KPool coverage requires capture sizes at most 4096")
    long_contexts = (8192, 32768, 131072, 1048576)
    variants = set()

    def add(tokens, context):
        history = min(tokens, max_requests) * ((context // 4 + 63) // 64) * 64
        variants.add((tokens, context, history))

    for tokens, row in enumerate(REPRESENTATIVES[: dsa_variants.max_tokens], start=1):
        short_classes = {}
        for context, representative in zip(CONTEXT_BOUNDS, row):
            if dsa_variants.resolve(tokens, context) != (tokens, representative):
                raise ValueError("Full KPool coverage requires all validated DSA plans")
            if context <= 2048:
                short_classes[representative] = max(
                    context, short_classes.get(representative, 0)
                )
        for context in (*short_classes.values(), *long_contexts):
            add(tokens, context)
    for tokens in capture_sizes:
        if tokens > dsa_variants.max_tokens:
            for context in (2048, *long_contexts):
                add(tokens, context)
    return variants


class KPoolPrefillGraphVariants:
    def __init__(self, specification, capture_sizes, dsa_variants, max_requests=32):
        self.capture_sizes = sorted(capture_sizes)
        self.dsa_variants = dsa_variants
        self.max_requests = max_requests
        self.plans = {}
        self._storage_plan = None
        automatic = specification == "all"
        if automatic:
            items = _all_variants(self.capture_sizes, dsa_variants, max_requests)
        else:
            items = [
                tuple(map(int, item.split(":"))) for item in specification.split(",")
            ]
        variants = set()
        for tokens, context, history in items:
            if not (
                1 <= tokens <= self.capture_sizes[-1]
                and 1 <= context <= 1048576
                and history > 0
                and history <= max_requests * 262144
                and history % 64 == 0
                and history >= ((context // 4 + 63) // 64) * 64
            ):
                raise ValueError(
                    f"Invalid KPool prefill capture variant: {tokens}:{context}:{history}"
                )
            if (
                tokens <= dsa_variants.max_tokens
                and self.dsa_variant((tokens, context, history)) is None
            ):
                raise ValueError(
                    f"KPool capture requires a matching DSA variant: {tokens}:{context}:{history}"
                )
            variants.add((tokens, context, history))
        self.variants = sorted(variants)
        self._variants_by_tokens = {}
        for variant in self.variants:
            tokens = (
                variant[0]
                if variant[0] <= dsa_variants.max_tokens
                else self.bucket(variant)
            )
            self._variants_by_tokens.setdefault(tokens, []).append(variant)
        self.max_tokens = max(v[0] for v in self.variants)
        self.max_context = max(v[1] for v in self.variants)
        self.max_history = max(v[2] for v in self.variants)
        max_writes = (self.max_tokens + 3 * max_requests) // 4
        self.storage_bytes = (
            self.max_history * 132
            + max_requests * self.max_context * 4
            + max_writes * 41
            + max_requests * 24
            + self.max_tokens * 24
            + self.max_history // 64 * 4
            + 8
        )
        graph_limit = 789 if automatic else 64
        if len(self.variants) > graph_limit or self.storage_bytes > 2 * 1024**3:
            raise ValueError(
                f"KPool variants exceed {graph_limit} graphs or 2 GiB of persistent buffers"
            )

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
        if len(lengths) != len(extends) or any(
            n <= 0 or n > s for n, s in zip(extends, lengths)
        ):
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
        token_key = tokens if tokens <= self.dsa_variants.max_tokens else bucket
        for variant in self._variants_by_tokens.get(token_key, ()):
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
        from sglang.srt.layers.attention.dsa.kpool_prefill_graph_plan import (
            KPoolPrefillGraphPlan,
        )

        if variant not in self.variants:
            raise ValueError("Unknown KPool capture variant")
        if self._storage_plan is None:
            self._storage_plan = KPoolPrefillGraphPlan(
                self.max_tokens,
                self.max_requests,
                self.max_history,
                device,
                max_context=self.max_context,
                mapping_mode="paged",
            )
        if variant not in self.plans:
            self.plans[variant] = KPoolPrefillGraphPlan(
                variant[0],
                self.max_requests,
                variant[2],
                device,
                max_context=variant[1],
                mapping_mode=mapping_mode,
                storage=self._storage_plan._storage,
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
                metadata.kpool_extend_plan,
                metadata.topk_indices_offset,
                req_pool_indices=forward_batch.req_pool_indices,
                extend_seq_lens=forward_batch.extend_seq_lens_cpu,
                max_seq_len=int(forward_batch.seq_lens_cpu.max()),
            )
        return plan
