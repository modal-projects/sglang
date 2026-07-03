"""L1 correctness test for extend-as-virtual-verify graph translation.

Runs a small (truncated, dummy-weight) MLA model WITHOUT speculative decoding,
so the CUDA graphs are DECODE-mode (num_tokens_per_bs=1) and the translator
decomposes an extend of M tokens into M single-token virtual decode requests
sharing the request's page table with staggered seq_lens.

For each case, runs the SAME prefix-extend twice on fresh requests with
identical token ids:
  leg E: translator disabled  -> fully-eager extend (MHA ragged path)
  leg T: translator enabled   -> monolithic decode-graph replay (absorbed MLA)
and compares next-token logits (argmax match + cosine similarity), then runs a
few graphed decode steps on both legs and compares greedy continuations.

Example (Kimi-K2.6 truncated to 4 layers, tokenspeed backend, TP=4):

python -m sglang.l1_extend_verify --model-path nvidia/Kimi-K2.6-NVFP4 --tp 4 \
    --load-format dummy --json-model-override-args '{"num_hidden_layers": 4}' \
    --trust-remote-code --dtype bfloat16 --quantization modelopt_fp4 \
    --kv-cache-dtype fp8_e4m3 --attention-backend tokenspeed_mla \
    --moe-runner-backend flashinfer_trtllm \
    --enforce-disable-flashinfer-allreduce-fusion \
    --mem-fraction-static 0.60 --page-size 64 \
    --cuda-graph-bs 1 2 3 4 6 8 12 16
"""

import argparse
import logging
import multiprocessing
import os
from array import array

import numpy as np
import torch
import torch.distributed as dist

from sglang.bench_one_batch import (
    TreeCacheNamespace,
    _maybe_prepare_mlp_sync_batch,
    _set_envs_and_config,
    load_model,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id

# (prefix_len, extend_len) per request in the batch; each case runs both legs.
CASES = [
    [(128, 12)],  # single req, the hot-extend shape (M small, prefix cached)
    [(128, 16)],  # exact captured-bs boundary
    [(64, 1)],  # single-token extend
    [(128, 5), (64, 9)],  # multi-req, mixed M (v_total=14 <= 16)
    # Extra extend-only buckets (SGLANG_EXTEND_GRAPH_BS beyond the native
    # list; skipped automatically if the env is not set):
    [(128, 17)],  # pads into extra bucket 24
    [(128, 24)],  # exact extra bucket
]
DECODE_STEPS = 4


def _make_reqs(rid_base, token_ids_list):
    sampling_params = SamplingParams(temperature=0, max_new_tokens=8)
    reqs = []
    for i, ids in enumerate(token_ids_list):
        req = Req(
            rid=rid_base + i,
            origin_input_text="",
            origin_input_ids=ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)
    return reqs


@torch.no_grad()
def _run_extend(reqs, mr):
    tree_cache = TreeCacheNamespace(
        page_size=mr.server_args.page_size,
        device=mr.device,
        token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=mr.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, mr)
    forward_batch = ForwardBatch.init_new(batch, mr)
    out = mr.forward(forward_batch)
    logits = out.logits_output.next_token_logits
    next_ids = mr.sample(out.logits_output, forward_batch)
    return next_ids, logits, batch, out.can_run_graph


@torch.no_grad()
def _run_decode(next_ids, batch, mr):
    # Mirrors sglang.bench_one_batch.decode().
    batch.input_ids = next_ids.to(torch.int64)
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, mr)
    forward_batch = ForwardBatch.init_new(batch, mr)
    out = mr.forward(forward_batch)
    next_ids = mr.sample(out.logits_output, forward_batch)
    return next_ids, out.logits_output.next_token_logits


def _run_leg(mr, case, rng_ids, rid_base, use_translator):
    """Prefill (eager, prefix=0), then prefix-extend (leg-dependent path),
    then a few decode steps. Returns (extend_logits, decode_token_lists)."""
    from sglang.srt.model_executor import model_runner as mr_mod

    if use_translator:
        os.environ["SGLANG_EXTEND_VERIFY_GRAPH"] = "1"
        mr._extend_verify_translator = mr_mod._UNSET_TRANSLATOR
        translator = mr._get_extend_verify_translator()
        assert translator is not None, "translator failed to enable"
        hits_before = translator.num_hits
    else:
        mr._extend_verify_translator = None

    # 1) prefill the prefix (M = prefix_len > max_bs => always eager).
    # Req.__init__ stores ids as array("q", ...) (buffer for np.frombuffer in
    # prepare_for_extend) — never assign plain lists to fill_ids.
    reqs = _make_reqs(rid_base, [ids[:p] for (p, _m), ids in zip(case, rng_ids)])
    _next, _logits, batch, _g = _run_extend(reqs, mr)

    # 2) prefix-extend by M tokens
    for i, (p_len, m) in enumerate(case):
        req = reqs[i]
        req.fill_ids = array("q", rng_ids[i][: p_len + m])
        req.prefix_indices = mr.req_to_token_pool.req_to_token[
            req.req_pool_idx, :p_len
        ].to(torch.int64)
        req.set_extend_input_len(m)
    next_ids, ext_logits, batch, ran_graph = _run_extend(reqs, mr)

    if use_translator:
        hits = translator.num_hits - hits_before
        assert hits == 1, f"expected exactly 1 translator hit, got {hits}"
        assert ran_graph, "translated extend did not report can_run_graph"
    else:
        assert not ran_graph, "eager leg unexpectedly ran a graph"

    # 3) greedy decode continuation
    decode_tokens = [next_ids.tolist()]
    for _ in range(DECODE_STEPS):
        next_ids, _ = _run_decode(next_ids, batch, mr)
        decode_tokens.append(next_ids.tolist())
    return ext_logits.float(), decode_tokens


def run_test(server_args, port_args, gpu_id, tp_rank):
    rank_print = print if tp_rank == 0 else (lambda *a, **k: None)
    mr_wrapped, _tok = load_model(server_args, port_args, gpu_id, tp_rank)
    mr = mr_wrapped.torch_runner

    runner = mr.graph_runner
    assert runner is not None, "CUDA graph runner required for this test"
    rank_print(
        f"[l1] graphs: mode={runner.capture_forward_mode} "
        f"num_tokens_per_bs={runner.num_tokens_per_bs} bs={runner.capture_bs}"
    )

    # Coverage ceiling of the translator (block=1 in decode mode => M tokens
    # need M virtual reqs). Cases beyond it are skipped (env-dependent).
    max_virtual = max(runner.capture_bs) * runner.num_tokens_per_bs

    rng = np.random.RandomState(42)
    failures = 0
    for ci, case in enumerate(CASES):
        if sum(m for (_p, m) in case) > max_virtual:
            rank_print(
                f"[l1] case {ci} {case}: SKIPPED (needs > {max_virtual} "
                f"virtual reqs; set SGLANG_EXTEND_GRAPH_BS)"
            )
            continue
        rng_ids = [
            list(rng.randint(0, 10000, (p + m,)).astype(np.int64))
            for (p, m) in case
        ]
        logits_e, dec_e = _run_leg(mr, case, rng_ids, 1000 * ci, False)
        logits_t, dec_t = _run_leg(mr, case, rng_ids, 1000 * ci + 100, True)

        cos = torch.nn.functional.cosine_similarity(logits_e, logits_t, dim=-1)
        argmax_match = bool(
            (logits_e.argmax(dim=-1) == logits_t.argmax(dim=-1)).all()
        )
        dec_match = dec_e == dec_t
        max_abs = (logits_e - logits_t).abs().max().item()
        ok = argmax_match and cos.min().item() > 0.99
        failures += 0 if ok else 1
        rank_print(
            f"[l1] case {ci} {case}: cos_min={cos.min().item():.5f} "
            f"max_abs={max_abs:.4f} argmax_match={argmax_match} "
            f"decode_match={dec_match} -> {'PASS' if ok else 'FAIL'}"
        )
        if not dec_match:
            rank_print(f"      dec_e={dec_e}\n      dec_t={dec_t}")

    if server_args.tp_size > 1:
        dist.barrier()
    if tp_rank == 0:
        if failures:
            print(f"[l1] EXTEND-VERIFY L1: {failures} case(s) FAILED")
        else:
            print("[l1] EXTEND-VERIFY L1: ALL CASES PASSED")
    assert failures == 0


def main(server_args):
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    if server_args.tp_size == 1:
        run_test(server_args, port_args, 0, 0)
        return
    workers = []
    for tp_rank in range(server_args.tp_size):
        with maybe_reindex_device_id(tp_rank) as gpu_id:
            proc = multiprocessing.Process(
                target=run_test,
                args=(server_args, port_args, gpu_id, tp_rank),
            )
            proc.start()
            workers.append(proc)
    exit_code = 0
    for proc in workers:
        proc.join()
        exit_code = exit_code or proc.exitcode
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main(server_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
