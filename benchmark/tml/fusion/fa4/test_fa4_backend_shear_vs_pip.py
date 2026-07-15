"""Compare SGLang's FA4 backend sheared-bias path against pip score_mod.

This is the middle layer between the raw FA4 kernel tests and the full HTTP
server A/B probe.  It drives ``FlashAttentionBackend.forward_extend`` and
``forward_decode`` with ``rel_bias=...``, then rebuilds the equivalent pip FA4
call from the backend's own metadata and KV-pool buffers:

    backend.forward_*(..., rel_bias=bias)
      vs
    pip flash_attn_varlen_func(..., score_mod=..., aux_tensors=[bias])

That specifically covers SGLang FA4 backend metadata/page-table/KV-cache
plumbing while still using pip as the score_mod reference.

Run:  python benchmark/tml/fusion/test_fa4_backend_shear_vs_pip.py
"""

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    set_forward_context,
)
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

from flash_attn.cute import flash_attn_varlen_func as pip_fn

from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

DEV = "cuda"
DTYPE = torch.bfloat16
PAGE_SIZE = 128
NH, NHK, D = 64, 8, 128
REL_EXTENT = 1024
FAILS = []


@lru_cache(maxsize=None)
def score_mod(rel_extent):
    return get_inkling_relative_attention_score_mod(rel_extent)


@dataclass(frozen=True)
class Case:
    name: str
    mode: ForwardMode
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    window_size: int | None = None
    nh: int = 64
    nhk: int = 8
    seed: int = 0


class MockReqToTokenPool:
    def __init__(self, batch_size, context_len):
        self.size = batch_size
        self.req_to_token = torch.zeros(
            batch_size, context_len, dtype=torch.int32, device=DEV
        )


class MockModelConfig:
    def __init__(self, context_len):
        self.context_len = context_len
        self.is_multimodal = False
        self.attention_arch = AttentionArch.MHA
        self.is_encoder_decoder = False
        self.is_local_attention_model = False
        self.head_dim = D
        self.hf_text_config = SimpleNamespace(
            num_attention_heads=NH,
            attn_logit_softcapping=None,
        )

    @staticmethod
    def get_num_kv_heads(tp_size):
        return NHK


class MockModelRunner:
    def __init__(self, batch_size, context_len):
        self.device = DEV
        self.dtype = DTYPE
        self.kv_cache_dtype = DTYPE
        self.is_hybrid_swa = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.prefill_aware_swa = False
        self.attn_cp_size = 1
        self.tp_size = 1
        self.page_size = PAGE_SIZE
        self.model_config = MockModelConfig(context_len)
        self.server_args = SimpleNamespace(
            kv_cache_dtype="auto",
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=0,
            enable_multi_layer_eagle=False,
            enable_deterministic_inference=False,
            is_embedding=False,
            chunked_prefill_size=0,
            disable_radix_cache=False,
            enable_dp_attention=False,
        )
        self.req_to_token_pool = MockReqToTokenPool(batch_size, context_len)
        # +PAGE_SIZE leaves room for one extra page, matching production pool
        # conventions for padded/dummy writes.
        self.token_to_kv_pool = MHATokenToKVPool(
            size=batch_size * context_len + PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
            head_num=NHK,
            head_dim=D,
            layer_num=1,
            device=DEV,
            enable_memory_saver=False,
        )
        self.hisparse_coordinator = None


def report(tag, ok, detail=""):
    print(f"  {tag:56s} {'OK' if ok else 'FAIL ' + detail}")
    if not ok:
        FAILS.append(tag)


def make_layer(window_size=None):
    layer = RadixAttention(
        num_heads=NH,
        head_dim=D,
        scaling=1.0 / D,
        num_kv_heads=NHK,
        layer_id=0,
        sliding_window_size=window_size,
    )
    layer.k_scale = None
    layer.v_scale = None
    return layer


def fill_req_to_token(req_to_token, kv_lens, max_pages_per_seq):
    """Populate production-style token-slot locations.

    FlashAttentionBackend converts these rows to a strided page table via
    ``row[:, ::page_size] // page_size`` when ``page_size > 1``.
    """
    for req_idx, kv_len in enumerate(kv_lens):
        base_page = req_idx * max_pages_per_seq
        locs = base_page * PAGE_SIZE + torch.arange(
            kv_len, dtype=torch.int32, device=DEV
        )
        req_to_token[req_idx, :kv_len] = locs


def locs_for_sequence(req_idx, token_start, token_count, max_pages_per_seq):
    base_page = req_idx * max_pages_per_seq
    token_ids = torch.arange(
        token_start, token_start + token_count, dtype=torch.int64, device=DEV
    )
    return base_page * PAGE_SIZE + token_ids


def make_forward_batch(case):
    batch_size = len(case.q_lens)
    max_kv_len = max(case.kv_lens)
    max_pages_per_seq = (max_kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    context_len = max_pages_per_seq * PAGE_SIZE
    runner = MockModelRunner(batch_size, context_len)
    fill_req_to_token(
        runner.req_to_token_pool.req_to_token, case.kv_lens, max_pages_per_seq
    )

    if case.mode == ForwardMode.EXTEND:
        q_lens = torch.tensor(case.q_lens, dtype=torch.int32, device=DEV)
        kv_lens = torch.tensor(case.kv_lens, dtype=torch.int32, device=DEV)
        prefix_lens = kv_lens - q_lens
        locs = [
            locs_for_sequence(
                i, int(prefix_lens[i].item()), int(q_lens[i].item()), max_pages_per_seq
            )
            for i in range(batch_size)
        ]
        out_cache_loc = torch.cat(locs, dim=0)
        input_ids = torch.zeros((sum(case.q_lens),), dtype=torch.int32, device=DEV)
        fb = ForwardBatch(
            forward_mode=case.mode,
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=torch.arange(batch_size, dtype=torch.int64, device=DEV),
            seq_lens=kv_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=sum(case.kv_lens),
            seq_lens_cpu=torch.tensor(case.kv_lens, dtype=torch.int32, device="cpu"),
            extend_prefix_lens=prefix_lens,
            extend_prefix_lens_cpu=[kv - q for q, kv in zip(case.q_lens, case.kv_lens)],
            extend_seq_lens=q_lens,
            extend_seq_lens_cpu=list(case.q_lens),
        )
    else:
        assert all(q == 1 for q in case.q_lens), "decode cases must use q_len=1"
        kv_lens = torch.tensor(case.kv_lens, dtype=torch.int32, device=DEV)
        locs = [
            locs_for_sequence(i, int(kv_lens[i].item() - 1), 1, max_pages_per_seq)
            for i in range(batch_size)
        ]
        out_cache_loc = torch.cat(locs, dim=0)
        fb = ForwardBatch(
            forward_mode=case.mode,
            batch_size=batch_size,
            input_ids=torch.zeros((batch_size,), dtype=torch.int32, device=DEV),
            req_pool_indices=torch.arange(batch_size, dtype=torch.int64, device=DEV),
            seq_lens=kv_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=sum(case.kv_lens),
            seq_lens_cpu=torch.tensor(case.kv_lens, dtype=torch.int32, device="cpu"),
        )

    return runner, fb, max_pages_per_seq


def populate_prefix_cache(runner, layer, case, max_pages_per_seq):
    if case.mode == ForwardMode.EXTEND:
        prefix_lens = [kv - q for q, kv in zip(case.q_lens, case.kv_lens)]
    else:
        prefix_lens = [kv - 1 for kv in case.kv_lens]

    locs = []
    for i, prefix_len in enumerate(prefix_lens):
        if prefix_len:
            locs.append(locs_for_sequence(i, 0, prefix_len, max_pages_per_seq))
    if not locs:
        return

    loc = torch.cat(locs, dim=0)
    k = torch.randn(loc.numel(), NHK, D, dtype=DTYPE, device=DEV)
    v = torch.randn(loc.numel(), NHK, D, dtype=DTYPE, device=DEV)
    runner.token_to_kv_pool.set_kv_buffer(layer, loc, k, v)


def backend_vs_pip(case):
    # ponytail: per-case head config via module globals; thread through params
    # if this ever grows beyond a test script.
    global NH, NHK
    NH, NHK = case.nh, case.nhk
    assert len(case.q_lens) == len(case.kv_lens)
    assert all(kv >= q for q, kv in zip(case.q_lens, case.kv_lens))
    torch.manual_seed(case.seed)

    runner, forward_batch, max_pages_per_seq = make_forward_batch(case)
    backend = FlashAttentionBackend(runner, fa_impl_ver=4)
    set_forward_context(ForwardContext(attn_backend=backend))
    layer = make_layer(window_size=case.window_size)
    populate_prefix_cache(runner, layer, case, max_pages_per_seq)

    total_q = sum(case.q_lens)
    q = torch.randn(total_q, NH, D, dtype=DTYPE, device=DEV)
    k = torch.randn(total_q, NHK, D, dtype=DTYPE, device=DEV)
    v = torch.randn(total_q, NHK, D, dtype=DTYPE, device=DEV)
    bias = torch.randn(total_q, NH, REL_EXTENT, dtype=DTYPE, device=DEV)

    backend.init_forward_metadata(forward_batch)
    if case.mode == ForwardMode.EXTEND:
        sheared = backend.forward_extend(q, k, v, layer, forward_batch, rel_bias=bias)
    else:
        sheared = backend.forward_decode(q, k, v, layer, forward_batch, rel_bias=bias)

    metadata = backend.forward_metadata
    key_cache, value_cache = runner.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    key_cache = key_cache.view(-1, PAGE_SIZE, NHK, D)
    value_cache = value_cache.view(-1, PAGE_SIZE, NHK, D)

    window_size = (
        (case.window_size, 0)
        if case.window_size is not None and case.window_size > -1
        else (-1, -1)
    )
    pip_out, _ = pip_fn(
        q,
        key_cache,
        value_cache,
        cu_seqlens_q=metadata.cu_seqlens_q,
        seqused_k=metadata.cache_seqlens_int32,
        max_seqlen_q=metadata.max_seq_len_q,
        page_table=metadata.page_table,
        softmax_scale=layer.scaling,
        causal=True,
        window_size=window_size,
        num_splits=backend.num_splits,
        score_mod=score_mod(REL_EXTENT),
        aux_tensors=[bias],
        return_lse=True,
    )
    pip_out = pip_out.view_as(sheared)

    ndiff = int((sheared != pip_out).sum().item())
    max_err = (sheared.float() - pip_out.float()).abs().max().item()
    report(case.name, ndiff == 0, f"ndiff={ndiff} max={max_err:.3e}")
    torch.cuda.empty_cache()


def main():
    cases = [
        Case("backend decode bs=1 kv=129", ForwardMode.DECODE, (1,), (129,), seed=1),
        Case(
            "backend decode ragged long",
            ForwardMode.DECODE,
            (1, 1, 1, 1),
            (129, 1024, 4097, 65537),
            seed=2,
        ),
        Case(
            "backend extend mixed q",
            ForwardMode.EXTEND,
            (7, 33, 256),
            (517, 4097, 8261),
            seed=3,
        ),
        Case(
            "backend extend local window",
            ForwardMode.EXTEND,
            (1, 7, 64),
            (517, 1024, 2048),
            window_size=REL_EXTENT - 1,
            seed=4,
        ),
        Case(
            "backend decode tp8 heads (8q/1kv)",
            ForwardMode.DECODE,
            (1, 1, 1),
            (129, 4097, 65537),
            nh=8,
            nhk=1,
            seed=5,
        ),
        Case(
            "backend extend tp8 heads (8q/1kv)",
            ForwardMode.EXTEND,
            (7, 33, 256),
            (517, 4097, 8261),
            nh=8,
            nhk=1,
            seed=6,
        ),
        Case(
            "backend extend 16k chunked prefill",
            ForwardMode.EXTEND,
            (16384,),
            (49152,),
            seed=7,
        ),
    ]

    for case in cases:
        backend_vs_pip(case)

    print("ALL OK" if not FAILS else f"FAILURES: {FAILS}")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
