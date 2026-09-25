"""Session media lifetime across replacement, abort, and deferred close."""

import unittest
from array import array
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.session.session_controller import Session, SessionController
from sglang.srt.session.streaming_session import StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _media(value):
    return MultimodalDataItem(modality=Modality.IMAGE, feature=torch.tensor([value]))


def _create(session, rid, *, parent=None, replace=False, media=None):
    recv = TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [1]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(
            id=session.session_id, rid=parent, replace=replace
        ),
    )
    req = session.create_req(recv, tokenizer=None, vocab_size=32)
    if media is not None:
        req.extend_image_inputs(MultimodalInputs(mm_items=[media]))
    return req


def _finish(session, req):
    req.finished_reason = FINISH_LENGTH(1)
    req._refresh_fill_ids()
    if session.streaming:
        session.finish_req(req)
    session.release_finished_req_mm_inputs(req)


class TestSessionMediaOwnership(CustomTestCase):
    def setUp(self):
        self.parallel = patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        )
        self.parallel.start()
        self.addCleanup(self.parallel.stop)
        self.cache = Mock()
        self.controller = SessionController(self.cache)

    def _session(self, *, streaming=False):
        session = Session(0, session_id="s", streaming=streaming)
        self.controller.sessions["s"] = session
        return session

    def _kv_cache(self):
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.addCleanup(reset_context)
        pool = ReqToTokenPool(
            size=4, max_context_len=32, device="cpu", enable_memory_saver=False
        )
        kv_pool = MHATokenToKVPool(
            size=16,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
        self.controller.tree_cache = cache
        return cache

    def _allocate_kv(self, req, cache):
        cache.req_to_token_pool.alloc([req])
        length = req.seqlen
        cache.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, length)),
            cache.token_to_kv_pool_allocator.alloc(length),
        )
        req.kv.kv_committed_len = length
        req.kv.kv_allocated_len = length

    def _assert_all_kv_returned(self, cache):
        self.assertEqual(cache.req_to_token_pool.available_size(), 4)
        self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), 16)

    def test_close_waits_for_live_tree_turns_with_shared_items(self):
        """Close must keep history readable until every live branch finishes."""
        session = self._session()
        shared, own = _media(10), _media(20)
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        child = _create(session, "child", parent=parent.rid, media=own)
        sibling = _create(session, "sibling", parent=parent.rid)

        self.assertIsNot(child.multimodal_inputs, parent.multimodal_inputs)
        self.controller._close("s")
        self.assertIn("s", self.controller.sessions)
        self.assertTrue(session.close_on_finish)
        self.assertEqual(shared.feature.tolist(), [10])
        self.assertEqual(own.feature.tolist(), [20])
        self.cache.release_session.assert_not_called()

        child.finished_reason = FINISH_ABORT()
        session.release_finished_req_mm_inputs(child)
        self.assertEqual(own.feature.tolist(), [20])
        self.assertEqual(shared.feature.tolist(), [10])
        self.controller._close("s")
        self.assertIn("s", self.controller.sessions)

        _finish(session, sibling)
        self.controller._close("s")
        self.assertNotIn("s", self.controller.sessions)
        self.assertIsNone(shared.feature)
        self.assertIsNone(own.feature)
        self.cache.release_session.assert_called_once_with("s")

    def test_replace_finished_descendant_preserves_other_history(self):
        """Replacing a branch releases removed media without invalidating siblings."""
        session = self._session()
        shared, branch, removed, sibling_media = [_media(i) for i in range(4)]
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        child = _create(session, "child", parent=parent.rid, media=branch)
        _finish(session, child)
        leaf = _create(session, "leaf", parent=child.rid, media=removed)
        _finish(session, leaf)
        sibling = _create(session, "sibling", parent=parent.rid, media=sibling_media)
        _finish(session, sibling)

        replacement = _create(session, "replacement", parent=child.rid, replace=True)
        self.assertNotIn(leaf.rid, session.req_nodes)
        self.assertIsNone(removed.feature)
        self.assertIsNone(leaf.multimodal_inputs)
        for item, value in ((shared, 0), (branch, 1), (sibling_media, 3)):
            self.assertEqual(item.feature.tolist(), [value])
        self.assertEqual(replacement.multimodal_inputs.mm_items, [shared, branch])

    def test_removed_live_turn_keeps_close_deferred(self):
        """An unfinished removed branch remains an owner until terminal cleanup."""
        session = self._session()
        shared, own = _media(1), _media(2)
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        removed = _create(session, "removed", parent=parent.rid, media=own)
        replacement = _create(session, "replacement", parent=parent.rid, replace=True)
        _finish(session, replacement)

        self.assertNotIn(removed.rid, session.req_nodes)
        self.assertIsInstance(removed.to_finish, FINISH_ABORT)
        self.assertTrue(session.has_unfinished_request())
        self.controller._close("s")
        self.assertIn("s", self.controller.sessions)
        self.assertEqual(shared.feature.tolist(), [1])
        self.assertEqual(own.feature.tolist(), [2])

        removed.finished_reason = removed.to_finish
        removed.to_finish = None
        session.release_req_mm_inputs(removed)
        self.assertIsNone(own.feature)
        self.assertEqual(shared.feature.tolist(), [1])
        self.assertFalse(session.has_unfinished_request())
        self.controller._close("s")
        self.assertIsNone(shared.feature)
        self.assertNotIn("s", self.controller.sessions)

    def test_replace_all_releases_each_shared_item_once(self):
        """Clearing a whole tree must not lose removed nodes or double-release media."""
        session = self._session()
        shared, child_media = _media(1), _media(2)
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        child = _create(session, "child", parent=parent.rid, media=child_media)
        _finish(session, child)
        original_release = MultimodalDataItem.release_transport_proxies
        with patch.object(
            MultimodalDataItem,
            "release_transport_proxies",
            autospec=True,
            side_effect=original_release,
        ) as release:
            replacement = _create(session, "replacement", replace=True)
        self.assertEqual(set(session.req_nodes), {replacement.rid})
        self.assertIsNone(shared.feature)
        self.assertIsNone(child_media.feature)
        self.assertCountEqual(
            [id(call.args[0]) for call in release.call_args_list],
            [id(shared), id(child_media)],
        )

    def test_close_releases_shared_items_once_across_distinct_containers(self):
        """Completed branches can hold separate containers with identical media items."""
        session = self._session()
        shared, child_media = _media(1), _media(2)
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        child = _create(session, "child", parent=parent.rid, media=child_media)
        _finish(session, child)
        original_release = MultimodalDataItem.release_transport_proxies
        with patch.object(
            MultimodalDataItem,
            "release_transport_proxies",
            autospec=True,
            side_effect=original_release,
        ) as release:
            self.controller._close("s")
        self.assertIsNone(shared.feature)
        self.assertIsNone(child_media.feature)
        self.assertCountEqual(
            [id(call.args[0]) for call in release.call_args_list],
            [id(shared), id(child_media)],
        )

    def test_streaming_abort_retains_media_until_terminal_cleanup(self):
        """An aborted uncommitted turn must remain visible to deferred close."""
        for with_history in (False, True):
            with self.subTest(with_history=with_history):
                session = self._session(streaming=True)
                shared, own = _media(1), _media(2)
                if with_history:
                    parent = _create(session, "parent", media=shared)
                    _finish(session, parent)
                req = _create(session, "live", media=own)
                req.finished_reason = FINISH_ABORT()
                self.controller._close("s")
                self.assertIn("s", self.controller.sessions)
                self.assertEqual(own.feature.tolist(), [2])
                session.abort_req(req.rid)
                session.release_req_mm_inputs(req)
                self.assertIsNone(own.feature)
                if with_history:
                    self.assertEqual(shared.feature.tolist(), [1])
                self.controller._close("s")
                self.assertNotIn("s", self.controller.sessions)
                if with_history:
                    self.assertIsNone(shared.feature)

    def test_abort_reason_does_not_end_deferred_resource_ownership(self):
        """A deferred abort must keep the session alive until KV cleanup completes."""
        for round_index, streaming in enumerate((False, True)):
            with self.subTest(streaming=streaming):
                session = self._session(streaming=streaming)
                own = _media(4)
                req = _create(session, "live", media=own)
                req.finished_reason = FINISH_ABORT()
                self.controller._close("s")
                self.assertIn("s", self.controller.sessions)
                self.assertEqual(own.feature.tolist(), [4])
                self.assertTrue(session.has_unfinished_request())
                now = 4.0 * round_index + 2.0
                self.assertIsNone(self.controller.plan_reap(now=now))

                session.release_finished_req_mm_inputs(req)
                if streaming:
                    self.assertIsNone(own.feature)
                else:
                    self.assertEqual(own.feature.tolist(), [4])
                self.assertFalse(session.has_unfinished_request())
                plan = self.controller.plan_reap(now=now + 2.0)
                self.assertEqual(plan.deferred, ["s"])
                self.controller.apply_reap(plan)
                self.controller.apply_reap(plan)
                self.assertNotIn("s", self.controller.sessions)
                self.assertIsNone(own.feature)

    def test_streaming_commit_transfers_shared_items_before_old_cleanup(self):
        """Advancing the checkpoint must not release items inherited by the new turn."""
        session = self._session(streaming=True)
        shared, own = _media(1), _media(2)
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        req = _create(session, "next", media=own)
        _finish(session, req)
        self.assertIsNone(parent.multimodal_inputs)
        self.assertEqual(shared.feature.tolist(), [1])
        self.assertEqual(own.feature.tolist(), [2])
        self.controller._close("s")
        self.assertIsNone(shared.feature)
        self.assertIsNone(own.feature)

    def test_abort_cleanup_preserves_prompt_usage_metadata(self):
        """Early media release must retain usage counts until the output is streamed."""
        session = self._session(streaming=True)
        media = _media(1)
        media.offsets = [(2, 4)]
        req = _create(session, "live", media=media)
        expected = req.multimodal_inputs.compute_mm_token_counts()
        req.finished_reason = FINISH_ABORT()
        session.abort_req(req.rid)
        self.assertIsNone(req.multimodal_inputs)
        self.assertIsNone(media.feature)
        self.assertEqual(
            (req.mm_image_tokens, req.mm_audio_tokens, req.mm_video_tokens), expected
        )

    def test_real_kv_release_retires_text_and_media_owners(self):
        """Normal completion ends active ownership after returning real pool resources."""
        cache = self._kv_cache()
        for with_media in (False, True):
            with self.subTest(with_media=with_media):
                session = self._session()
                media = _media(1) if with_media else None
                req = _create(session, "live", media=media)
                self._allocate_kv(req, cache)
                req.finished_reason = FINISH_LENGTH(1)
                self.controller._close("s")
                self.assertIn("s", self.controller.sessions)

                original_finish = session.release_finished_req_mm_inputs

                def finish_after_pool_release(finished_req):
                    self._assert_all_kv_returned(cache)
                    original_finish(finished_req)

                with patch.object(
                    session,
                    "release_finished_req_mm_inputs",
                    side_effect=finish_after_pool_release,
                ):
                    release_kv_cache(req, cache)
                self._assert_all_kv_returned(cache)
                self.assertFalse(session.has_unfinished_request())
                if with_media:
                    self.assertEqual(media.feature.tolist(), [1])
                self.controller._close("s")
                self.assertNotIn("s", self.controller.sessions)
                if with_media:
                    self.assertIsNone(media.feature)

    def test_real_kv_release_retires_removed_media_once(self):
        """Terminal release frees abandoned branch media and preserves committed media."""
        cache = self._kv_cache()
        for aborted in (False, True):
            with self.subTest(aborted=aborted):
                session = self._session()
                shared, own = _media(1), _media(2)
                parent = _create(session, "parent", media=shared)
                _finish(session, parent)
                req = _create(session, "live", parent=parent.rid, media=own)
                self._allocate_kv(req, cache)
                session.req_nodes[parent.rid].clear_children(session.req_nodes)
                req.finished_reason = FINISH_ABORT() if aborted else FINISH_LENGTH(1)
                original_release = MultimodalDataItem.release_transport_proxies
                with patch.object(
                    MultimodalDataItem,
                    "release_transport_proxies",
                    autospec=True,
                    side_effect=original_release,
                ) as release:
                    release_kv_cache(req, cache, is_insert=False)
                    self.assertIsNone(req.multimodal_inputs)
                    self.assertIsNone(own.feature)
                    session.release_finished_req_mm_inputs(req)
                self._assert_all_kv_returned(cache)
                self.assertIsNone(req.multimodal_inputs)
                self.assertIsNone(own.feature)
                self.assertEqual(shared.feature.tolist(), [1])
                self.assertEqual(
                    [call.args[0] for call in release.call_args_list], [own]
                )
                self.assertFalse(session.has_unfinished_request())
                self.controller._close("s")
                self.assertIsNone(shared.feature)

    def test_real_kv_retract_preserves_owner_until_terminal_release(self):
        """Explicit retry intent preserves media ownership despite a stamped reason."""
        cache = self._kv_cache()
        session = self._session()
        media = _media(1)
        req = _create(session, "live", media=media)
        self._allocate_kv(req, cache)
        req.finished_reason = FINISH_ABORT()
        release_kv_cache(req, cache, is_insert=False, is_retract=True)
        self._assert_all_kv_returned(cache)
        self.assertTrue(session.has_unfinished_request())
        self.assertEqual(media.feature.tolist(), [1])

        req.finished_reason = None
        self._allocate_kv(req, cache)
        req.finished_reason = FINISH_ABORT()
        release_kv_cache(req, cache, is_insert=False)
        self._assert_all_kv_returned(cache)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(media.feature.tolist(), [1])
        self.controller._close("s")
        self.assertIsNone(media.feature)

    def test_deferred_kv_completion_keeps_session_until_release_callback(self):
        """Recording an abort must not release resources before the transfer completes."""
        cache = self._kv_cache()
        session = self._session()
        media = _media(1)
        req = _create(session, "live", media=media)
        self._allocate_kv(req, cache)
        req.finished_reason = FINISH_ABORT()
        completion = Future()
        completion.add_done_callback(
            lambda _: release_kv_cache(req, cache, is_insert=False)
        )
        self.controller._close("s")
        self.assertIn("s", self.controller.sessions)
        self.assertTrue(req.kv.holds_kv)
        self.assertTrue(session.has_unfinished_request())
        self.assertEqual(media.feature.tolist(), [1])

        completion.set_result(None)
        self._assert_all_kv_returned(cache)
        self.assertFalse(req.kv.holds_kv)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(media.feature.tolist(), [1])
        self.controller._close("s")
        self.assertNotIn("s", self.controller.sessions)
        self.assertIsNone(media.feature)

    def test_nonstream_abort_history_remains_usable_for_append_and_replace(self):
        """Aborted tree history must retain the media required by later turns."""
        cache = self._kv_cache()
        session = self._session()
        media, child_media = _media(1), _media(2)
        parent = _create(session, "parent", media=media)
        parent.output_ids.append(7)
        self._allocate_kv(parent, cache)
        parent.finished_reason = FINISH_ABORT()
        release_kv_cache(parent, cache, is_insert=False)
        self._assert_all_kv_returned(cache)
        self.assertFalse(session.has_unfinished_request())
        self.assertIsNotNone(media.feature)
        self.assertEqual(media.feature.tolist(), [1])

        child = _create(session, "child", parent=parent.rid, media=child_media)
        self.assertIsNone(child.to_finish)
        self.assertEqual(list(child.origin_input_ids), [1, 7, 1])
        self.assertEqual(child.multimodal_inputs.mm_items, [media, child_media])
        self.assertEqual(child.multimodal_inputs.mm_items[0].feature.tolist(), [1])
        _finish(session, child)

        replacement = _create(session, "replacement", parent=parent.rid, replace=True)
        self.assertIsNone(replacement.to_finish)
        self.assertEqual(list(replacement.origin_input_ids), [1, 7, 1])
        self.assertEqual(replacement.multimodal_inputs.mm_items, [media])
        self.assertEqual(
            replacement.multimodal_inputs.mm_items[0].feature.tolist(), [1]
        )
        self.assertIsNone(child_media.feature)
        _finish(session, replacement)
        self.controller._close("s")
        self.assertIsNone(media.feature)

    def test_discarded_intake_turn_cannot_be_used_as_history(self):
        """Failed intake must remove its history entry while preserving its ancestor."""
        session = self._session()
        shared, own = _media(1), _media(2)
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        req = _create(session, "invalid", parent=parent.rid, media=own)
        req.finished_reason = FINISH_ABORT()
        session.discard_req(req)
        session.discard_req(req)
        self.assertEqual(set(session.req_nodes), {parent.rid})
        self.assertEqual(session.req_nodes[parent.rid].children, [])
        self.assertFalse(session.has_unfinished_request())
        self.assertIsNone(own.feature)
        self.assertEqual(shared.feature.tolist(), [1])

        for replace in (False, True):
            rejected = _create(
                session, f"rejected-{replace}", parent=req.rid, replace=replace
            )
            self.assertIsInstance(rejected.to_finish, FINISH_ABORT)
            self.assertIsNone(rejected.multimodal_inputs)
        accepted = _create(session, "accepted", parent=parent.rid)
        self.assertIsNone(accepted.to_finish)
        self.assertEqual(accepted.multimodal_inputs.mm_items, [shared])
        self.assertEqual(shared.feature.tolist(), [1])

    def test_discard_reparents_surviving_children_without_releasing_their_media(self):
        """Removing one history entry must preserve descendants and their shared items."""
        session = self._session()
        shared, middle_media, leaf_media = [_media(i) for i in range(3)]
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        middle = _create(session, "middle", parent=parent.rid, media=middle_media)
        _finish(session, middle)
        leaf = _create(session, "leaf", parent=middle.rid, media=leaf_media)

        session.discard_req(middle)
        self.assertEqual(set(session.req_nodes), {parent.rid, leaf.rid})
        self.assertEqual(
            session.req_nodes[parent.rid].children, [session.req_nodes[leaf.rid]]
        )
        self.assertIs(session.req_nodes[leaf.rid].parent, session.req_nodes[parent.rid])
        self.assertEqual(
            leaf.multimodal_inputs.mm_items, [shared, middle_media, leaf_media]
        )
        self.assertEqual(middle_media.feature.tolist(), [1])
        _finish(session, leaf)
        self.controller._close("s")
        for item in (shared, middle_media, leaf_media):
            self.assertIsNone(item.feature)

    def test_late_abort_callback_cannot_retire_new_request_with_reused_rid(self):
        """A delayed old completion must not retire a newer request with the same RID."""
        cache = self._kv_cache()
        session = self._session()
        old_media, new_media = _media(1), _media(2)
        old = _create(session, "reused", media=old_media)
        self._allocate_kv(old, cache)
        old.finished_reason = FINISH_ABORT()

        new = _create(session, "reused", media=new_media)
        self._allocate_kv(new, cache)
        release_kv_cache(old, cache, is_insert=False)
        self.assertIsNone(old_media.feature)
        self.assertIs(session.req_nodes[new.rid].req, new)
        self.assertEqual(new_media.feature.tolist(), [2])
        self.assertTrue(new.kv.holds_kv)

        new.finished_reason = FINISH_ABORT()
        self.controller._close("s")
        self.assertIn("s", self.controller.sessions)
        self.assertTrue(session.has_unfinished_request())
        self.assertEqual(new_media.feature.tolist(), [2])
        release_kv_cache(new, cache, is_insert=False)
        self._assert_all_kv_returned(cache)
        self.controller._close("s")
        self.assertNotIn("s", self.controller.sessions)
        self.assertIsNone(new_media.feature)

    def test_reused_rid_releases_unreferenced_completed_history(self):
        """Replacing a completed RID must not orphan its media or its parent link."""
        session = self._session()
        shared, old_media, new_media = [_media(i) for i in range(3)]
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        old = _create(session, "reused", parent=parent.rid, media=old_media)
        _finish(session, old)
        original_release = MultimodalDataItem.release_transport_proxies
        with patch.object(
            MultimodalDataItem,
            "release_transport_proxies",
            autospec=True,
            side_effect=original_release,
        ) as release:
            new = _create(session, "reused", media=new_media)
            self.assertIsNone(old_media.feature)
            self.assertIsNone(old.multimodal_inputs)
            self.assertEqual(shared.feature.tolist(), [0])
            self.assertEqual(new_media.feature.tolist(), [2])
            self.assertEqual(session.req_nodes[parent.rid].children, [])
            self.assertIsNone(session.req_nodes[new.rid].parent)
            _finish(session, new)
            self.controller._close("s")
        self.assertCountEqual(
            [id(call.args[0]) for call in release.call_args_list],
            [id(shared), id(old_media), id(new_media)],
        )

    def test_reused_rid_can_append_to_its_own_history_without_cycle(self):
        """A reused RID can inherit its old media while replacing the old tree node."""
        session = self._session()
        shared, old_media, new_media = [_media(i) for i in range(3)]
        parent = _create(session, "parent", media=shared)
        _finish(session, parent)
        old = _create(session, "reused", parent=parent.rid, media=old_media)
        old.output_ids.append(7)
        _finish(session, old)
        old_node = session.req_nodes[old.rid]
        original_release = MultimodalDataItem.release_transport_proxies
        with patch.object(
            MultimodalDataItem,
            "release_transport_proxies",
            autospec=True,
            side_effect=original_release,
        ) as release:
            new = _create(session, "reused", parent=old.rid, media=new_media)
            self.assertIsNone(new.to_finish)
            self.assertEqual(list(new.origin_input_ids), [1, 1, 7, 1])
            self.assertEqual(
                new.multimodal_inputs.mm_items, [shared, old_media, new_media]
            )
            self.assertIsNone(old.multimodal_inputs)
            self.assertIsNone(old_node.parent)
            self.assertEqual(old_node.children, [])
            new_node = session.req_nodes[new.rid]
            self.assertIs(new_node.parent, session.req_nodes[parent.rid])
            self.assertEqual(session.req_nodes[parent.rid].children, [new_node])
            self.assertEqual(old_media.feature.tolist(), [1])
            _finish(session, new)
            self.controller._close("s")
        self.assertCountEqual(
            [id(call.args[0]) for call in release.call_args_list],
            [id(shared), id(old_media), id(new_media)],
        )

    def test_preabort_then_queue_cleanup_preserves_borrowed_checkpoint_media(self):
        """Detaching a rejected KV borrower must release only its newly appended media."""
        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()
        from sglang.srt.managers.scheduler import Scheduler

        cache = StreamingSession(self._kv_cache())
        self.controller.tree_cache = cache
        session = self._session(streaming=True)
        shared, own = _media(1), _media(2)
        parent = _create(session, "parent", media=shared)
        self._allocate_kv(parent, cache)
        parent.finished_reason = FINISH_LENGTH(0)
        parent.finished_len = 0
        parent._refresh_fill_ids()
        release_kv_cache(parent, cache)

        slot = cache.slots[session.session_id]
        saved_kv = slot.kv
        saved_row = saved_kv.req_pool_idx
        saved_length = saved_kv.kv_allocated_len
        saved_indices = cache.req_to_token_pool.req_to_token[
            saved_row, :saved_length
        ].clone()
        counts = (
            cache.req_to_token_pool.available_size(),
            cache.token_to_kv_pool_allocator.available_size(),
        )
        rejected = _create(session, "rejected", media=own)
        rejected.init_next_round_input(cache, cow_mamba=False)
        self.assertIs(rejected.kv, saved_kv)
        rejected.to_finish = FINISH_ABORT("rejected before admission")
        original_release = MultimodalDataItem.release_transport_proxies
        with patch.object(
            MultimodalDataItem,
            "release_transport_proxies",
            autospec=True,
            side_effect=original_release,
        ) as release:
            self.assertIsNone(cache.find_active_slot(rejected))
            self.assertIsNone(rejected.session)
            fallback_had_media = rejected.multimodal_inputs is not None
            Scheduler._release_dropped_waiting_req_mm_inputs(
                SimpleNamespace(), rejected
            )
            Scheduler._release_dropped_waiting_req_mm_inputs(
                SimpleNamespace(), rejected
            )
        self.assertIsNotNone(shared.feature)
        self.assertEqual(shared.feature.tolist(), [1])
        self.assertFalse(fallback_had_media)
        self.assertIsNone(rejected.multimodal_inputs)
        self.assertIsNone(own.feature)
        self.assertEqual([call.args[0] for call in release.call_args_list], [own])
        self.assertIsNot(rejected.kv, saved_kv)
        self.assertFalse(rejected.kv.holds_kv)
        self.assertIs(cache.slots[session.session_id], slot)
        self.assertIs(slot.kv, saved_kv)
        self.assertEqual(slot.kv.req_pool_idx, saved_row)
        self.assertEqual(slot.kv.kv_allocated_len, saved_length)
        torch.testing.assert_close(
            cache.req_to_token_pool.req_to_token[saved_row, :saved_length],
            saved_indices,
        )
        self.assertEqual(
            (
                cache.req_to_token_pool.available_size(),
                cache.token_to_kv_pool_allocator.available_size(),
            ),
            counts,
        )
        accepted = _create(session, "accepted")
        self.assertIsNone(accepted.to_finish)
        self.assertEqual(accepted.multimodal_inputs.mm_items, [shared])
        accepted.init_next_round_input(cache, cow_mamba=False)
        self.assertIs(accepted.kv, saved_kv)
        accepted.to_finish = FINISH_ABORT("cleanup")
        self.assertIsNone(cache.find_active_slot(accepted))
        self.controller._close("s")
        self._assert_all_kv_returned(cache)
        self.assertIsNone(shared.feature)


if __name__ == "__main__":
    unittest.main()
