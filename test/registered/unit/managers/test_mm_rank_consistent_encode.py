import unittest

from sglang.test.test_utils import CustomTestCase

"""Ranks with different cache hits must encode the same ordered image batch."""

from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_schedule import PerImageRequestInfo
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeAttnTpGroup:
    """all_reduce(SUM) over this rank's flags plus a preset peer vector."""

    def __init__(self, peer_flags):
        self.peer_flags = peer_flags
        self.calls = []

    def all_reduce(self, input_):
        self.calls.append(input_.clone())
        peer = torch.tensor(self.peer_flags, dtype=input_.dtype, device=input_.device)
        assert peer.shape == input_.shape
        return input_ + peer


def _item(hash_value, tokens):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=hash_value,
        offsets=[(0, tokens - 1)],
        feature=torch.zeros(tokens, 2),
    )
    return item


def _requests(items):
    offsets = []
    cursor = 0
    for item in items:
        n = item.offsets[0][1] - item.offsets[0][0] + 1
        offsets.append((cursor, cursor + n - 1))
        cursor += n
    return [
        PerImageRequestInfo(
            req_idx=0,
            items=list(items),
            items_offset=offsets,
            extend_prefix_len=0,
            extend_seq_len=cursor,
        )
    ]


def _run(items, group, encode):
    with patch.object(mm_schedule, "_mm_encode_sync_group", return_value=group):
        return mm_schedule._batch_encode_per_image_misses(
            encode, _requests(items), torch.device("cpu")
        )


class TestMultimodalCacheSync(CustomTestCase):
    def setUp(self):
        mm_schedule.init_mm_embedding_cache(1 << 20)
        self.addCleanup(mm_schedule.init_mm_embedding_cache, 1 << 20)

    def test_local_hit_peer_miss_forces_reencode(self):
        item = _item(11, 4)
        cached = torch.ones(4, 2)
        mm_schedule.embedding_cache.set(
            11, mm_schedule.EmbeddingResult(embedding=cached)
        )
        group = _FakeAttnTpGroup(peer_flags=[1])
        encode = Mock(return_value=torch.full((4, 2), 2.0))

        out = _run([item], group, encode)

        encode.assert_called_once()
        assert encode.call_args.args[0] == [item]
        assert group.calls[0].tolist() == [0]
        assert torch.equal(out[(11, 4)], torch.full((4, 2), 2.0))

    def test_all_ranks_hit_skips_encoder_and_still_syncs(self):
        item = _item(12, 3)
        mm_schedule.embedding_cache.set(
            12, mm_schedule.EmbeddingResult(embedding=torch.ones(3, 2))
        )
        group = _FakeAttnTpGroup(peer_flags=[0])
        encode = Mock()

        out = _run([item], group, encode)

        encode.assert_not_called()
        assert len(group.calls) == 1
        assert torch.equal(out[(12, 3)], torch.ones(3, 2))

    def test_encode_order_follows_batch_order_not_local_miss_order(self):
        a, b = _item(21, 2), _item(22, 3)
        # This rank hit `a` and missed `b`; a peer missed `a`.
        mm_schedule.embedding_cache.set(
            21, mm_schedule.EmbeddingResult(embedding=torch.ones(2, 2))
        )
        group = _FakeAttnTpGroup(peer_flags=[1, 0])
        encode = Mock(return_value=torch.arange(10.0).reshape(5, 2))

        out = _run([a, b], group, encode)

        assert encode.call_args.args[0] == [a, b]
        assert torch.equal(out[(21, 2)], torch.arange(4.0).reshape(2, 2))
        assert torch.equal(out[(22, 3)], torch.arange(4.0, 10.0).reshape(3, 2))

    def test_flag_off_or_single_rank_does_not_sync(self):
        item = _item(31, 2)
        mm_schedule.embedding_cache.set(
            31, mm_schedule.EmbeddingResult(embedding=torch.ones(2, 2))
        )
        encode = Mock()

        with envs.SGLANG_MM_RANK_CONSISTENT_ENCODE.override(False):
            assert mm_schedule._mm_encode_sync_group() is None

        out = _run([item], None, encode)
        encode.assert_not_called()
        assert torch.equal(out[(31, 2)], torch.ones(2, 2))

    def test_sync_group_none_without_distributed_init(self):
        with (
            envs.SGLANG_MM_RANK_CONSISTENT_ENCODE.override(True),
            patch.object(
                mm_schedule.torch.distributed, "is_initialized", return_value=False
            ),
        ):
            assert mm_schedule._mm_encode_sync_group() is None

    def test_rank_consistent_miss_keys_is_union(self):
        group = _FakeAttnTpGroup(peer_flags=[0, 1, 0])
        got = mm_schedule._rank_consistent_miss_keys(
            [1, 2, 3], {1}, group, torch.device("cpu")
        )
        assert got == {1, 2}

    def test_same_hash_different_spans_keep_distinct_collective_positions(self):
        a, b = _item(41, 2), _item(41, 3)
        mm_schedule.embedding_cache.set(
            41, mm_schedule.EmbeddingResult(embedding=torch.ones(3, 2))
        )
        group = _FakeAttnTpGroup(peer_flags=[0, 1])
        encode = Mock(return_value=torch.arange(10.0).reshape(5, 2))
        out = _run([a, b], group, encode)
        assert encode.call_args.args[0] == [a, b]
        assert set(out) == {(41, 2), (41, 3)}
        assert out[(41, 2)].shape == (2, 2)
        assert out[(41, 3)].shape == (3, 2)


if __name__ == "__main__":
    unittest.main()
