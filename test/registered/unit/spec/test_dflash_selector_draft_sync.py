import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.dflash import CandidateSelector
from sglang.srt.speculative import dflash_worker_v2 as dflash
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Group:
    def __init__(self, rank, messages, world_size=2):
        self.world_size = world_size
        self.rank_in_group = rank
        self.ranks = [4, 5][:world_size]
        self.device_group = object()
        self.messages = messages
        self.calls = 0

    def broadcast(self, values, src):
        def transfer(tensor, *, src, group):
            assert src == self.ranks[0]
            assert group is self.device_group
            if self.rank_in_group == 0:
                self.messages.append(tensor.clone())
            else:
                tensor.copy_(self.messages[self.calls])
            self.calls += 1

        with (
            mock.patch("torch.distributed.broadcast", side_effect=transfer),
            mock.patch(
                "sglang.srt.distributed.parallel_state.is_hip", return_value=False
            ),
        ):
            return GroupCoordinator.broadcast(self, values, src=src)


def _sync(group, policy="all"):
    with envs.SGLANG_SPEC_TP_SYNC.override(policy):
        return SpecTpSync(group)


class TestSelectorPacket(CustomTestCase):
    def test_packet_preserves_tokens_candidates_and_probability_bits(self):
        """Peers must consume one coherent proposal, without rounding IDs or q."""
        messages = []
        groups = [_Group(rank, messages) for rank in range(2)]
        tokens = torch.tensor([[2**40 + 1, 2**40 + 3]], dtype=torch.int64)
        candidates = torch.tensor(
            [[[2**40 + 1, 2**40 + 2], [2**40 + 3, 2**40 + 4]]], dtype=torch.int64
        )
        q_bits = torch.tensor(
            [[[0x3F000001, -2147483648], [1, 0x7FC01234]]], dtype=torch.int32
        )
        buffers = [torch.full((3, 2, 5), -7, dtype=torch.int64) for _ in groups]
        for rank, group in enumerate(groups):
            local_tokens = tokens.clone() if rank == 0 else torch.zeros_like(tokens)
            local_ids = (
                candidates.clone() if rank == 0 else torch.zeros_like(candidates)
            )
            local_q = q_bits.clone().view(torch.float32)
            if rank:
                local_q.zero_()
            pointers = [v.data_ptr() for v in (local_tokens, local_ids, local_q)]
            dflash._sync_dflash_selector_draft(
                local_tokens,
                local_ids,
                local_q,
                tp_sync=_sync(group),
                pack_buffer=buffers[rank],
            )
            self.assertTrue(torch.equal(local_tokens, tokens))
            self.assertTrue(torch.equal(local_ids, candidates))
            self.assertTrue(torch.equal(local_q.view(torch.int32), q_bits))
            self.assertEqual(
                pointers, [v.data_ptr() for v in (local_tokens, local_ids, local_q)]
            )
            self.assertTrue(torch.all(buffers[rank][1:] == -7))
            self.assertEqual(group.calls, 1)
        self.assertEqual(len(messages), 1)

    def test_single_rank_and_disabled_site_leave_state_unchanged(self):
        for world_size, policy in ((1, "all"), (2, "all,-17")):
            with self.subTest(world_size=world_size, policy=policy):
                group = _Group(0, [], world_size=world_size)
                worker = dflash.DFlashWorkerV2.__new__(dflash.DFlashWorkerV2)
                worker._tp_sync = _sync(group, policy)
                worker._selector_sync_buf = None
                tokens = torch.tensor([[1, 2]], dtype=torch.int64)
                ids = torch.tensor([[[1, 3], [2, 4]]], dtype=torch.int64)
                q = torch.tensor([[[0.3, 0.7], [0.8, 0.2]]])
                worker._selector_sample = (ids, q)
                before = [v.clone() for v in (tokens, ids, q)]
                worker._sync_selector_draft(tokens)
                for actual, expected in zip((tokens, ids, q), before):
                    self.assertTrue(torch.equal(actual, expected))
                self.assertIsNone(worker._selector_sync_buf)
                self.assertEqual(group.calls, 0)

    def test_invalid_packet_contracts_fail_before_collective(self):
        tokens = torch.zeros((2, 3), dtype=torch.int64)
        ids = torch.zeros((2, 3, 4), dtype=torch.int64)
        q = torch.zeros((2, 3, 4), dtype=torch.float32)
        packet = torch.empty((2, 3, 9), dtype=torch.int64)
        for field, replacement, message in (
            ("draft_next", tokens[:, :2], "match"),
            ("candidate_ids", ids.to(torch.float32), "int64"),
            ("q_rows", q[:, :, :1], "match"),
            ("q_rows", q.to(torch.float16), "float32"),
            ("pack_buffer", packet[:1], "pack buffer"),
            ("pack_buffer", packet[:, :, :8], "pack buffer"),
            ("pack_buffer", packet.to(torch.int32), "pack buffer"),
        ):
            with self.subTest(
                field=field, shape=replacement.shape, dtype=replacement.dtype
            ):
                group = _Group(0, [])
                args = dict(
                    draft_next=tokens, candidate_ids=ids, q_rows=q, pack_buffer=packet
                )
                args[field] = replacement
                with self.assertRaisesRegex(ValueError, message):
                    dflash._sync_dflash_selector_draft(**args, tp_sync=_sync(group))
                self.assertEqual(group.calls, 0)


class _ReachedVerify(Exception):
    pass


def _worker(
    group, *, captured, sampling_enabled=True, max_bs=4, selector=True, domino=False
):
    worker = dflash.DFlashWorkerV2.__new__(dflash.DFlashWorkerV2)
    worker.device = "cpu"
    worker.block_size = 4
    worker.selector = (
        CandidateSelector(hidden_size=4, vocab_size=256, state_rank=2, top_k=4)
        if selector
        else None
    )
    worker._selector_sampling_enabled = sampling_enabled
    worker._is_domino = domino
    worker._warned_sampling_fallback = True
    worker._tp_sync = _sync(group)
    worker._selector_sample = None
    worker._selector_sync_buf = None
    worker._draft_block_ids_buf = None
    worker._block_pos_offsets = torch.arange(4).unsqueeze(0)
    worker._use_triton_prepare_block = False
    worker._mask_token_id = 0
    worker._full_embed_gpu = None
    worker._noise_embed_scale = 1.0
    worker.use_compact_draft_cache = False
    worker._need_mamba_verify_commit = False
    worker._draft_block_spec_info = None
    worker.draft_tp_context = lambda *_: nullcontext()
    embedding = torch.nn.Embedding(256, 4)
    target_model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(256, 4)),
        get_input_embeddings=lambda: embedding,
    )
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=target_model,
            model_config=SimpleNamespace(vocab_size=256),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros(max_bs, 32, dtype=torch.int64)
            ),
        )
    )
    worker.draft_model = SimpleNamespace(
        lm_head=None,
        candidate_selector=worker.selector,
        prefix_gru=object(),
        embed_proj=object(),
        shift_label=False,
    )
    worker.domino_candidate_pool_size = 4
    worker._draft_sampler = (
        dflash._SelectorDraftSampler(
            draft_model=worker.draft_model,
            block_size=4,
            max_bs=max_bs,
            device="cpu",
            sampling_enabled=sampling_enabled,
        )
        if captured and selector and not domino
        else None
    )
    return worker


def _run_to_verify(worker, *, rank, bs, top_ks, captured, seed=1):
    """Run the real worker through proposal creation and stop at target inference."""
    sampling_info = SimpleNamespace(
        is_all_greedy=all(k <= 1 for k in top_ks),
        is_any_greedy=any(k <= 1 for k in top_ks),
        top_ks=torch.tensor(top_ks),
        temperatures=torch.full((bs, 1), 0.8),
    )
    hidden = torch.zeros(bs * 4, 4)
    ids = torch.arange(bs * 3 * 4, dtype=torch.int64).reshape(bs, 3, 4) + rank * 64
    scores = torch.randn(
        bs, 3, 4, 4, generator=torch.Generator().manual_seed(seed + rank)
    )
    draft_out = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden), can_run_graph=captured
    )

    def forward(batch):
        if captured:
            worker._draft_sampler(hidden, batch.input_ids)
        return draft_out

    worker.draft_model_runner = SimpleNamespace(tp_group=None, forward=forward)
    seq_lens = torch.full((bs,), 5, dtype=torch.int64)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        spec_info=dflash.make_draft_input_v2(
            bonus_tokens=torch.arange(bs, dtype=torch.int64) + 200,
            new_seq_lens=seq_lens,
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=5 * bs,
        req_pool_indices=torch.arange(bs),
        sampling_info=sampling_info,
        has_grammar=False,
    )
    observed = {}

    def verify_boundary(verify_input, *_):
        observed["tokens"] = verify_input.draft_token.view(bs, 4).clone()
        observed["state"] = (
            None
            if worker._selector_sample is None
            else tuple(v.clone() for v in worker._selector_sample)
        )
        raise _ReachedVerify

    with (
        torch.random.fork_rng(devices=[]),
        torch.compiler.set_stance("force_eager"),
        mock.patch.object(dflash, "_selector_lattice", return_value=(ids, scores)),
        mock.patch.object(
            dflash,
            "assign_extend_cache_locs_func",
            return_value=torch.zeros(bs * 4, dtype=torch.int64),
        ),
        mock.patch.object(dflash, "enable_num_token_non_padded", return_value=False),
        mock.patch.object(
            dflash.DFlashVerifyInput, "prepare_for_verify", new=verify_boundary
        ),
        mock.patch.object(torch.Tensor, "record_stream", new=lambda *_: None),
    ):
        torch.manual_seed(seed + rank * 101)
        try:
            worker.forward_batch_generation(batch)
        except _ReachedVerify:
            return observed
    raise AssertionError("worker did not reach verification")


class TestSelectorProposalCallers(CustomTestCase):
    def test_eager_and_captured_proposals_share_tokens_and_q_before_verify(self):
        """Acceptance sees the root's tokens and the q that sampled those tokens."""
        for captured in (False, True):
            for top_ks in ([4, 4], [1, 4]):
                with self.subTest(captured=captured, top_ks=top_ks):
                    messages = []
                    groups = [_Group(rank, messages) for rank in range(2)]
                    workers = [_worker(group, captured=captured) for group in groups]
                    results = [
                        _run_to_verify(
                            worker, rank=rank, bs=2, top_ks=top_ks, captured=captured
                        )
                        for rank, worker in enumerate(workers)
                    ]
                    root, peer = results
                    self.assertTrue(torch.equal(root["tokens"], peer["tokens"]))
                    for root_value, peer_value in zip(root["state"], peer["state"]):
                        self.assertTrue(torch.equal(root_value, peer_value))
                    self.assertEqual(groups[0].calls, groups[1].calls)
                    self.assertEqual(len(messages), 1 + int(top_ks[0] == 1))
                    ids, q = peer["state"]
                    self.assertTrue(
                        torch.all((peer["tokens"][:, 1:, None] == ids).any(dim=-1))
                    )
                    if top_ks[0] == 1:
                        self.assertTrue(torch.all((q[0] == 0) | (q[0] == 1)))
                    if captured:
                        for worker in workers:
                            self.assertEqual(
                                worker._selector_sample[0].data_ptr(),
                                worker._draft_sampler.candidate_out.data_ptr(),
                            )
                            self.assertEqual(
                                worker._selector_sample[1].data_ptr(),
                                worker._draft_sampler.q_out.data_ptr(),
                            )

    def test_eager_packet_grows_and_reuses_capacity(self):
        messages = []
        groups = [_Group(rank, messages) for rank in range(2)]
        workers = [_worker(group, captured=False) for group in groups]
        for bs in (1, 5, 2):
            results = [
                _run_to_verify(
                    worker, rank=rank, bs=bs, top_ks=[4] * bs, captured=False
                )
                for rank, worker in enumerate(workers)
            ]
            self.assertTrue(torch.equal(results[0]["tokens"], results[1]["tokens"]))
            self.assertTrue(torch.equal(results[0]["state"][1], results[1]["state"][1]))
            if bs == 5:
                buffers = [worker._selector_sync_buf for worker in workers]
            elif bs == 2:
                for worker, buffer in zip(workers, buffers):
                    self.assertIs(worker._selector_sync_buf, buffer)

    def test_captured_buffers_survive_repeated_results(self):
        messages = []
        groups = [_Group(rank, messages) for rank in range(2)]
        workers = [_worker(group, captured=True) for group in groups]
        samplers = [worker._draft_sampler for worker in workers]
        for bs, seed in ((4, 1), (2, 20), (4, 40)):
            tails = [
                tuple(v[bs:].clone() for v in (sampler.candidate_out, sampler.q_out))
                for sampler in samplers
            ]
            results = [
                _run_to_verify(
                    worker, rank=rank, bs=bs, top_ks=[4] * bs, captured=True, seed=seed
                )
                for rank, worker in enumerate(workers)
            ]
            self.assertTrue(torch.equal(results[0]["tokens"], results[1]["tokens"]))
            self.assertTrue(torch.equal(results[0]["state"][1], results[1]["state"][1]))
            for sampler, tail in zip(samplers, tails):
                self.assertTrue(torch.equal(sampler.candidate_out[bs:], tail[0]))
                self.assertTrue(torch.equal(sampler.q_out[bs:], tail[1]))

    def test_greedy_and_sampling_disabled_callers_preserve_greedy_sync(self):
        for captured in (False, True):
            for sampling_enabled, top_ks in ((True, [1, 1]), (False, [1, 4])):
                with self.subTest(captured=captured, sampling_enabled=sampling_enabled):
                    messages = []
                    groups = [_Group(rank, messages) for rank in range(2)]
                    workers = [
                        _worker(
                            group, captured=captured, sampling_enabled=sampling_enabled
                        )
                        for group in groups
                    ]
                    results = [
                        _run_to_verify(
                            worker, rank=rank, bs=2, top_ks=top_ks, captured=captured
                        )
                        for rank, worker in enumerate(workers)
                    ]
                    self.assertTrue(
                        torch.equal(results[0]["tokens"], results[1]["tokens"])
                    )
                    self.assertIsNone(results[1]["state"])
                    self.assertEqual(len(messages), 1)

    def test_plain_and_domino_callers_sync_greedy_proposals(self):
        for domino in (False, True):
            with self.subTest(domino=domino):
                messages = []
                workers = [
                    _worker(
                        _Group(rank, messages),
                        captured=False,
                        selector=False,
                        domino=domino,
                    )
                    for rank in range(2)
                ]
                results = []
                for rank, worker in enumerate(workers):
                    proposals = (
                        torch.arange(6, dtype=torch.int64).view(2, 3) + rank * 64
                    )
                    with (
                        mock.patch.object(
                            dflash, "domino_greedy_rollout", return_value=proposals
                        ),
                        mock.patch.object(
                            worker,
                            "_greedy_sample_from_vocab_parallel_head",
                            return_value=proposals,
                        ),
                        mock.patch.object(dflash, "get_tp_group", return_value=None),
                    ):
                        results.append(
                            _run_to_verify(
                                worker, rank=rank, bs=2, top_ks=[4, 4], captured=False
                            )
                        )
                self.assertTrue(torch.equal(results[0]["tokens"], results[1]["tokens"]))
                self.assertIsNone(results[1]["state"])
                self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
