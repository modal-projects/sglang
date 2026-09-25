"""Session rewrites must retain only complete media and their original owners."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import torch

from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_key import MultimodalKeySpan
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _media(value, offsets):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.full((2, 3), float(value)),
        offsets=offsets,
    )
    item.set_pad_value()
    item.set_hash(11)
    return item


def _create(session, rid, tokens, *, parent=None, media=None, **params):
    received = TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", tokens),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=8),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(id=session.session_id, rid=parent, **params),
    )
    req = session.create_req(received, tokenizer=None, vocab_size=32)
    if media is not None and not isinstance(req.to_finish, FINISH_ABORT):
        SessionController.adjust_mm_offsets(received, req, media)
        req.extend_image_inputs(media)
    return req


def _finish(session, req, output=()):
    req.output_ids = array("q", output)
    req.finished_reason = FINISH_LENGTH(len(output))
    req._refresh_fill_ids()
    if session.streaming:
        session.finish_req(req)
    session.release_finished_req_mm_inputs(req)


def _key(req):
    return RadixKey(req.origin_input_ids, mm_spans=req.mm_cache_spans)


class TestSessionContentIdentity(CustomTestCase):
    def setUp(self):
        context = patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        )
        context.start()
        self.addCleanup(context.stop)

    def _parent(self, *, offsets=None, streaming=False, output=()):
        session = Session(0, session_id="s", streaming=streaming)
        item = _media(1, [(1, 2)] if offsets is None else offsets)
        tokens = [1, item.pad_value, item.pad_value, 2, 3]
        parent = _create(
            session, "parent", tokens, media=MultimodalInputs(mm_items=[item])
        )
        _finish(session, parent, output)
        return session, parent, item

    def test_rewrite_before_media_does_not_match_removed_content(self):
        """Replaced image slots must not mask unrelated new text during matching."""
        session, parent, item = self._parent()
        old_tokens = parent.origin_input_ids[:]
        old_spans = parent.mm_cache_spans
        child = _create(session, "child", [7, 8, 9], parent=parent.rid, offset=1)
        self.assertEqual(list(child.origin_input_ids), [1, 7, 8, 9])
        self.assertEqual(_key(child).match(_key(parent)), 1)
        self.assertIsNone(child.multimodal_inputs)
        self.assertEqual(child.mm_cache_spans, ())
        self.assertEqual(parent.origin_input_ids, old_tokens)
        self.assertEqual(parent.mm_cache_spans, old_spans)
        self.assertIs(parent.multimodal_inputs.mm_items[0], item)

    def test_cut_between_items_keeps_owner_and_places_new_media_after_prefix(self):
        """A retained IPC owner stays shared while a later image is replaced."""
        session = Session(0, session_id="s")
        first, removed, replacement = (
            _media(1, [(1, 2)]),
            _media(2, [(4, 5)]),
            _media(3, [(1, 2)]),
        )
        feature = first.feature
        proxy = CudaIpcTensorTransportProxy(
            data=feature,
            info_data=feature,
            pool_ipc_handle=(0, b"session-content", 24, 0, b"ref", 0, b"event", False),
            pool_id=uuid4().hex,
            pool_byte_offset=0,
            ready_byte_offset=32,
            ack_byte_offset=36,
            generation=1,
            total_consumer_count=1,
            use_pool_handle_cache=True,
        )
        first.feature = proxy
        tokens = [
            1,
            first.pad_value,
            first.pad_value,
            2,
            removed.pad_value,
            removed.pad_value,
            3,
        ]
        positions = torch.tensor(
            [
                [0, 1, 1, 2, 50, 50, 51],
                [0, 2, 3, 4, 60, 61, 62],
                [0, 3, 4, 5, 70, 71, 72],
            ]
        )
        media = MultimodalInputs(
            mm_items=[first, removed],
            padded_input_ids=tokens[:],
            mrope_positions=positions,
        )
        media.processor_metadata = {"layout": "retained"}
        parent = _create(session, "parent", tokens, media=media)
        _finish(session, parent)
        parent.multimodal_inputs.mrope_position_delta_repeated_cache = torch.tensor(
            [[123]]
        )
        new_tokens = [7, replacement.pad_value, replacement.pad_value, 8]
        child = _create(
            session,
            "child",
            new_tokens,
            parent=parent.rid,
            offset=4,
        )
        torch.testing.assert_close(
            child.multimodal_inputs.mrope_positions,
            torch.tensor([[0, 1, 1, 2], [0, 2, 3, 4], [0, 3, 4, 5]]),
        )
        torch.testing.assert_close(
            child.multimodal_inputs.mrope_position_delta, torch.tensor([[2]])
        )
        self.assertIsNone(child.multimodal_inputs.mrope_position_delta_repeated_cache)
        self.assertEqual(child.multimodal_inputs.padded_input_ids, tokens[:4])
        torch.testing.assert_close(parent.multimodal_inputs.mrope_positions, positions)
        torch.testing.assert_close(
            parent.multimodal_inputs.mrope_position_delta, torch.tensor([[66]])
        )
        torch.testing.assert_close(
            parent.multimodal_inputs.mrope_position_delta_repeated_cache,
            torch.tensor([[123]]),
        )
        added = MultimodalInputs(mm_items=[replacement])
        SessionController.adjust_mm_offsets(
            SimpleNamespace(input_ids=new_tokens), child, added
        )
        child.extend_image_inputs(added)
        self.assertIsNot(child.multimodal_inputs, parent.multimodal_inputs)
        self.assertEqual(
            [id(x) for x in child.multimodal_inputs.mm_items],
            [id(first), id(replacement)],
        )
        self.assertIs(first.feature, proxy)
        self.assertFalse(proxy._consumer_acknowledged)
        self.assertEqual(replacement.offsets, [(5, 6)])
        self.assertEqual(
            [(s.start, s.end) for s in child.mm_cache_spans], [(1, 3), (5, 7)]
        )
        self.assertEqual(_key(child).match(_key(parent)), 4)
        self.assertEqual(
            child.multimodal_inputs.processor_metadata, {"layout": "retained"}
        )
        self.assertEqual(parent.multimodal_inputs.mm_items, [first, removed])
        self.assertEqual(parent.multimodal_inputs.padded_input_ids, tokens)
        self.assertEqual(first.offsets, [(1, 2)])
        self.assertEqual(removed.offsets, [(4, 5)])

        child.finished_reason = FINISH_ABORT()
        session.discard_req(child)
        self.assertIsNone(child.multimodal_inputs)
        self.assertIsNone(replacement.feature)
        self.assertIsNotNone(removed.feature)
        self.assertIs(first.feature, proxy)
        self.assertFalse(proxy._consumer_acknowledged)

    def test_partial_item_and_gaps_between_its_spans_abort_before_publication(self):
        """A single media owner cannot be retained with only some of its spans."""
        for offsets, cut in (
            ([(1, 2)], 2),
            ([(1, 2), (4, 5)], 3),
            ([(1, 2), (4, 5)], 4),
            ([(1, 2), (4, 5)], 5),
        ):
            with self.subTest(offsets=offsets, cut=cut):
                session = Session(0, session_id="s")
                item = _media(1, offsets)
                tokens = [
                    1,
                    item.pad_value,
                    item.pad_value,
                    2,
                    item.pad_value,
                    item.pad_value,
                    3,
                ]
                parent = _create(
                    session, "parent", tokens, media=MultimodalInputs(mm_items=[item])
                )
                _finish(session, parent)
                child = _create(
                    session, "child", [7, 8, 9], parent=parent.rid, offset=cut
                )
                self.assertIsInstance(child.to_finish, FINISH_ABORT)
                self.assertNotIn(child.rid, session.req_nodes)
                self.assertNotIn(id(child), session._active_reqs)
                self.assertIsNone(child.multimodal_inputs)
                self.assertEqual(parent.origin_input_ids, array("q", tokens))
                self.assertEqual(item.offsets, offsets)
                self.assertIsNotNone(item.feature)

    def test_rejected_named_replace_preserves_descendant_history(self):
        """A rejected cut must not clear the branch selected for replacement."""
        session, parent, item = self._parent()
        descendant = _create(session, "descendant", [4, 5], parent=parent.rid)
        _finish(session, descendant)
        parent_node, descendant_node = (
            session.req_nodes[parent.rid],
            session.req_nodes[descendant.rid],
        )
        rejected = _create(
            session, "replacement", [7], parent=parent.rid, offset=2, replace=True
        )
        self.assertIsInstance(rejected.to_finish, FINISH_ABORT)
        self.assertEqual(set(session.req_nodes), {parent.rid, descendant.rid})
        self.assertIs(session.req_nodes[parent.rid], parent_node)
        self.assertIs(session.req_nodes[descendant.rid], descendant_node)
        self.assertEqual(parent_node.children, [descendant_node])
        self.assertIsNone(parent.to_finish)
        self.assertIsNone(descendant.to_finish)
        self.assertIs(descendant.multimodal_inputs.mm_items[0], item)
        self.assertIsNotNone(item.feature)

    def test_aggregate_and_missing_offsets_are_inseparable(self):
        """Aggregate padding cannot map a retained subset back to item owners."""
        for kind in ("override", "missing", "legacy"):
            for cut, outcome in ((1, "drop"), (3, "abort"), (4, "abort"), (6, "keep")):
                with self.subTest(kind=kind, cut=cut):
                    session = Session(0, session_id="s")
                    items = [_media(1, [(1, 2)]), _media(2, [(4, 5)])]
                    pad = items[0].pad_value
                    tokens = [1, pad, pad, 2, pad, pad, 3]
                    media = MultimodalInputs(mm_items=items)
                    if kind == "override":
                        media.cache_span_overrides = (
                            MultimodalKeySpan(1, 3, items[0].cache_key),
                            MultimodalKeySpan(4, 6, items[1].cache_key),
                        )
                    elif kind == "missing":
                        for item in items:
                            item.offsets = None
                    else:
                        media.image_pad_len = [2, 2]
                        media.image_offsets = [1, 4]
                        media.data_offsets = [0, 3]
                    parent = _create(session, "parent", tokens, media=media)
                    _finish(session, parent)
                    child = _create(
                        session, "child", [7], parent=parent.rid, offset=cut
                    )
                    if outcome == "abort":
                        self.assertIsInstance(child.to_finish, FINISH_ABORT)
                        self.assertNotIn(child.rid, session.req_nodes)
                    elif outcome == "drop":
                        self.assertIsNone(child.multimodal_inputs)
                        self.assertEqual(child.mm_cache_spans, ())
                        self.assertEqual(_key(child).match(_key(parent)), 1)
                    else:
                        self.assertEqual(
                            [id(x) for x in child.multimodal_inputs.mm_items],
                            [id(x) for x in items],
                        )
                        self.assertEqual(_key(child).match(_key(parent)), 6)
                    self.assertEqual(parent.multimodal_inputs.mm_items, items)
                    self.assertTrue(all(item.feature is not None for item in items))

    def test_encoder_prefix_tokens_are_an_indivisible_group(self):
        """An encoder prefix must survive whole or be rejected before scheduling."""
        for offset, expected in (
            (-100, "drop"),
            (1, "abort"),
            (3, "abort"),
            (4, "keep"),
        ):
            with self.subTest(offset=offset):
                session = Session(0, session_id="s")
                items = [_media(1, None), _media(2, None)]
                parent = _create(
                    session,
                    "parent",
                    [0, 0, 0, 0, 2],
                    media=MultimodalInputs(mm_items=items, num_image_tokens=4),
                )
                _finish(session, parent)
                child = _create(session, "child", [7], parent=parent.rid, offset=offset)
                if expected == "abort":
                    self.assertIsInstance(child.to_finish, FINISH_ABORT)
                    self.assertNotIn(child.rid, session.req_nodes)
                elif expected == "drop":
                    self.assertIsNone(child.multimodal_inputs)
                    self.assertEqual(child.mm_cache_spans, ())
                else:
                    self.assertEqual(child.multimodal_inputs.num_image_tokens, 4)
                    self.assertEqual(_key(child).match(_key(parent)), 4)

    def test_negative_oversized_and_zero_offsets_use_actual_retained_prefix(self):
        """Python slicing and append semantics determine which media survived."""
        for offset, retained, outcome in (
            (-100, 0, "drop"),
            (-4, 1, "drop"),
            (-3, 2, "abort"),
            (-2, 3, "keep"),
            (100, 5, "keep"),
            (0, 5, "keep"),
        ):
            with self.subTest(offset=offset):
                session, parent, item = self._parent()
                child = _create(
                    session, "child", [7, 8, 9], parent=parent.rid, offset=offset
                )
                if outcome == "abort":
                    self.assertIsInstance(child.to_finish, FINISH_ABORT)
                    self.assertNotIn(child.rid, session.req_nodes)
                else:
                    self.assertEqual(len(child.origin_input_ids), retained + 3)
                    self.assertEqual(_key(child).match(_key(parent)), retained)
                    if outcome == "drop":
                        self.assertIsNone(child.multimodal_inputs)
                    else:
                        self.assertIs(child.multimodal_inputs.mm_items[0], item)
                self.assertEqual(len(parent.origin_input_ids), 5)

    def test_drop_previous_output_shifts_new_media_after_input_only(self):
        """Discarding generated output changes the prefix used for new media."""
        for drop, start in ((False, 7), (True, 5)):
            with self.subTest(drop=drop):
                session, parent, item = self._parent(output=[20, 21])
                new_item = _media(2, [(0, 1)])
                child = _create(
                    session,
                    "child",
                    [new_item.pad_value, new_item.pad_value, 9],
                    parent=parent.rid,
                    drop_previous_output=drop,
                    media=MultimodalInputs(mm_items=[new_item]),
                )
                self.assertEqual(new_item.offsets, [(start, start + 1)])
                self.assertEqual(
                    [(s.start, s.end) for s in child.mm_cache_spans],
                    [(1, 3), (start, start + 2)],
                )
                self.assertEqual(_key(child).match(_key(parent)), 5)
                self.assertEqual(parent.multimodal_inputs.mm_items, [item])
                self.assertEqual(list(parent.output_ids), [20, 21])

    def test_streaming_append_preserves_media_and_inflight_ownership(self):
        """Streaming history keeps the same media while a new turn is active."""
        session, parent, item = self._parent(streaming=True, output=[20])
        old_key = RadixKey(parent.origin_input_ids[:], mm_spans=parent.mm_cache_spans)
        child = _create(session, "child", [7, 8])
        self.assertIsNone(child.finished_reason)
        self.assertIs(child.multimodal_inputs.mm_items[0], item)
        self.assertEqual(_key(child).match(old_key), 5)
        self.assertEqual(
            list(child.origin_input_ids),
            [1, item.pad_value, item.pad_value, 2, 3, 20, 7, 8],
        )
        self.assertIs(session.req_nodes[parent.rid].req, parent)
        self.assertIn(id(child), session._active_reqs)
        _finish(session, child)
        self.assertEqual(set(session.req_nodes), {child.rid})
        self.assertIsNone(parent.multimodal_inputs)
        self.assertIs(child.multimodal_inputs.mm_items[0], item)
        self.assertIsNotNone(item.feature)

    def test_append_preserves_and_merges_dynamic_padding_offsets(self):
        """Appending media must retain legacy model offsets in a copied wrapper."""
        session = Session(0, session_id="s")
        first, second = _media(1, [(1, 2)]), _media(2, [(1, 2)])
        original = MultimodalInputs(mm_items=[first], image_pad_len=[2])
        original.image_offsets, original.data_offsets = [1], [0]
        parent = _create(
            session,
            "parent",
            [1, first.pad_value, first.pad_value, 2, 3],
            media=original,
        )
        _finish(session, parent)
        added = MultimodalInputs(mm_items=[second], image_pad_len=[2])
        added.image_offsets, added.data_offsets = [6], [5]
        child = _create(
            session,
            "child",
            [7, second.pad_value, second.pad_value, 8],
            parent=parent.rid,
            media=added,
        )
        self.assertIsNot(child.multimodal_inputs, parent.multimodal_inputs)
        self.assertEqual(child.multimodal_inputs.image_pad_len, [2, 2])
        self.assertEqual(child.multimodal_inputs.image_offsets, [1, 6])
        self.assertEqual(child.multimodal_inputs.data_offsets, [0, 5])
        self.assertEqual(
            [id(x) for x in child.multimodal_inputs.mm_items], [id(first), id(second)]
        )
        self.assertEqual(
            [(s.start, s.end) for s in child.mm_cache_spans], [(1, 3), (6, 8)]
        )
        self.assertEqual(parent.multimodal_inputs.image_pad_len, [2])
        self.assertEqual(parent.multimodal_inputs.image_offsets, [1])
        self.assertEqual(parent.multimodal_inputs.data_offsets, [0])
        self.assertEqual(parent.multimodal_inputs.mm_items, [first])


if __name__ == "__main__":
    unittest.main()
