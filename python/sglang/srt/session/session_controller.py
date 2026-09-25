# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import time
import uuid
from array import array
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    SessionReapPlan,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.utils.common import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


class SessionReqNode:
    def __init__(
        self,
        req: Req,
        parent: Optional[SessionReqNode] = None,
        children=None,
    ):
        self.req = req
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = [] if not children else children

    def clear_children(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)
        self.children = []

    def clear(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)

        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()
        del req_dict[self.req.rid]
        if self.req.session is not None:
            self.req.session.retire_req(self.req)

    def abort(self):
        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()

    def __str__(self):
        return self._str_helper(self.req.rid)

    def _str_helper(self, prefix=""):
        if len(self.children) == 0:
            return prefix + "\n"
        else:
            origin_prefix = prefix
            prefix += " -- " + self.children[0].req.rid
            ret = self.children[0]._str_helper(prefix)
            for child in self.children[1:]:
                prefix = " " * len(origin_prefix) + " \\- " + child.req.rid
                ret += child._str_helper(prefix)
            return ret


class Session:
    def __init__(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
    ):
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.capacity_of_str_len = capacity_of_str_len
        self.streaming = streaming
        self.timeout = timeout
        self.last_active_time: float = time.monotonic()
        self.req_nodes: Dict[str, SessionReqNode] = {}
        self.close_on_finish: bool = False
        self._inflight: bool = False
        self._inflight_rid: Optional[str] = None
        self._active_reqs: Dict[int, Req] = {}
        self._retired_reqs: Dict[int, Req] = {}
        # Token-array lengths of last_req as of its finish_req. The share path
        # appends speculatively beyond these; only finish_req confirms them, so
        # _share_token_arrays trims back first (heals aborted turns).
        self.committed_origin_len: Optional[int] = None
        self.committed_unpadded_len: Optional[int] = None
        self.committed_fill_len: Optional[int] = None

    def is_timed_out(self) -> bool:
        if self.timeout is None:
            return False
        return time.monotonic() - self.last_active_time > self.timeout

    def has_unfinished_request(self) -> bool:
        if self._active_reqs or (self.streaming and self._inflight):
            return True
        return any(not req.finished() for req in self._iter_reqs())

    def _iter_reqs(self):
        yield from (node.req for node in self.req_nodes.values())
        yield from self._retired_reqs.values()
        yield from self._active_reqs.values()

    def retire_req(self, req: Req) -> None:
        if req.finished() and id(req) not in self._active_reqs:
            self.release_req_mm_inputs(req)
        else:
            # Replacement removes history before the scheduler finishes its work.
            self._retired_reqs[id(req)] = req

    def _unlink_req_node(self, node: SessionReqNode) -> None:
        if self.req_nodes.get(node.req.rid) is node:
            del self.req_nodes[node.req.rid]
        parent = node.parent
        if parent is not None:
            index = parent.children.index(node)
            parent.children[index : index + 1] = node.children
        for child in node.children:
            child.parent = parent
        node.parent = None
        node.children = []

    def discard_req(self, req: Req) -> None:
        assert not (
            req.kv.holds_kv
            or req.kv.holds_mamba
            or req.kv.retraction_backup is not None
        ), "An abandoned session turn must release its KV ownership before discard"
        node = self.req_nodes.get(req.rid)
        if node is not None and node.req is req:
            self._unlink_req_node(node)
        self._retire_aborted_req_mm_inputs(req)

    def release_finished_req_mm_inputs(self, req: Req) -> None:
        # A finish reason can precede deferred KV release. The scheduler calls
        # this only after its terminal resource cleanup is complete.
        if isinstance(req.finished_reason, FINISH_ABORT):
            self._retire_aborted_req_mm_inputs(req)
        elif id(req) in self._retired_reqs:
            self.release_req_mm_inputs(req)
        else:
            self._active_reqs.pop(id(req), None)

    def _retire_aborted_req_mm_inputs(self, req: Req) -> None:
        if self._active_reqs.get(id(req)) is req and self._inflight_rid == req.rid:
            self._inflight = False
            self._inflight_rid = None
        node = self.req_nodes.get(req.rid)
        if not self.streaming and node is not None and node.req is req:
            # Tree sessions allow later turns to append to aborted history.
            self._active_reqs.pop(id(req), None)
        else:
            self.release_req_mm_inputs(req)

    def release_req_mm_inputs(self, req: Req) -> None:
        mm = req.multimodal_inputs
        if mm is not None:
            if not (req.mm_image_tokens or req.mm_audio_tokens or req.mm_video_tokens):
                (
                    req.mm_image_tokens,
                    req.mm_audio_tokens,
                    req.mm_video_tokens,
                ) = mm.compute_mm_token_counts()
            retained = {
                id(item)
                for other in self._iter_reqs()
                if other is not req and other.multimodal_inputs is not None
                for item in other.multimodal_inputs.mm_items
            }
            owned = []
            for item in mm.mm_items:
                if id(item) not in retained:
                    retained.add(id(item))
                    owned.append(item)
            if owned:
                mm.release_features(owned)
            req.multimodal_inputs = None
        self._retired_reqs.pop(id(req), None)
        self._active_reqs.pop(id(req), None)

    @staticmethod
    def _strip_bos_token(req: TokenizedGenerateReqInput, tokenizer) -> None:
        """Trim a leading BOS on an appended turn; shift mm offsets to match."""
        if not (
            tokenizer is not None
            and req.input_ids
            and req.input_ids[0] == tokenizer.bos_token_id
        ):
            return
        req.input_ids = req.input_ids[1:]
        if req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if item.offsets:
                    if any(s == 0 for s, _ in item.offsets):
                        logging.warning(
                            "mm_item offset starts at 0 (BOS position), "
                            "clamping to 0 after BOS strip"
                        )
                    item.offsets = [
                        (max(0, s - 1), max(0, e - 1)) for s, e in item.offsets
                    ]

    def _share_token_arrays(self, last_req: Req, new_input_ids):
        """Plain streaming append: reuse last_req's token arrays in place.

        Trims each array back to its committed length first — an earlier turn
        may have appended its tokens and then aborted before finish_req, and
        req_nodes still points at last_req, so anything beyond the committed
        lengths is unconfirmed. Then extends with last turn's output and the
        new input. Returns (input_ids, input_ids_unpadded, carry_fill);
        carry_fill (== the new origin) spares the first fill_ids rebuild.
        """
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        input_ids = last_req.origin_input_ids
        del input_ids[self.committed_origin_len :]
        if last_req.origin_input_ids_unpadded is input_ids:
            input_ids_unpadded = input_ids
        else:
            input_ids_unpadded = last_req.origin_input_ids_unpadded
            del input_ids_unpadded[self.committed_unpadded_len :]

        carry_fill = last_req.full_untruncated_fill_ids
        if (
            not isinstance(carry_fill, array)
            or carry_fill is input_ids
            or carry_fill is input_ids_unpadded
        ):
            # Unexpected type or aliased with an origin array (extending it
            # below would double-append): let _refresh_fill_ids rebuild.
            carry_fill = None
        else:
            del carry_fill[self.committed_fill_len :]
            baked = len(carry_fill) - len(input_ids)
            if 0 <= baked <= len(out_tail):
                carry_fill.extend(out_tail[baked:])
                carry_fill.extend(new_input_ids)
            else:
                carry_fill = None

        input_ids.extend(out_tail)
        input_ids.extend(new_input_ids)
        if input_ids_unpadded is not input_ids:
            input_ids_unpadded.extend(out_tail)
            input_ids_unpadded.extend(new_input_ids)
        return input_ids, input_ids_unpadded, carry_fill

    @staticmethod
    def _concat_token_arrays(
        last_req: Req, req: TokenizedGenerateReqInput, session_params
    ):
        """Copy-based assembly for replace/offset/drop_previous_output turns."""
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        input_ids = last_req.origin_input_ids + out_tail
        if session_params.drop_previous_output:
            input_ids = last_req.origin_input_ids[:]
        if session_params.offset and session_params.offset != 0:
            input_ids = input_ids[: session_params.offset] + req.input_ids
        else:
            input_ids += req.input_ids

        input_ids_unpadded = last_req.origin_input_ids_unpadded + out_tail
        if session_params.drop_previous_output:
            input_ids_unpadded = last_req.origin_input_ids_unpadded[:]
        if session_params.offset and session_params.offset != 0:
            input_ids_unpadded = (
                input_ids_unpadded[: session_params.offset] + req.input_ids
            )
        else:
            input_ids_unpadded += req.input_ids
        return input_ids, input_ids_unpadded

    def create_req(
        self,
        req: TokenizedGenerateReqInput,
        tokenizer,
        vocab_size: int,
        eos_token_ids=None,
        disagg_mode: Optional[DisaggregationMode] = None,
    ):
        assert req.session_params is not None
        session_params = req.session_params

        last_req_node = None
        last_req = None
        abort = False
        abort_message = ""
        if self.streaming:
            # Streaming sessions: only simple appends allowed; reject otherwise.
            if self._inflight:
                abort = True
                abort_message = "Streaming session already has an active request."
            elif session_params.replace:
                abort = True
                abort_message = "Streaming sessions do not support replace."
            elif session_params.drop_previous_output:
                abort = True
                abort_message = (
                    "Streaming sessions do not support drop_previous_output."
                )
            elif session_params.offset and session_params.offset != 0:
                abort = True
                abort_message = "Streaming sessions do not support offset."
            elif self.req_nodes:
                assert len(self.req_nodes) == 1
                # Peek (don't pop) the single req_node. req_nodes is updated
                # only in finish_req after the request completes successfully.
                [last_req_node] = self.req_nodes.values()
                last_req = last_req_node.req
        elif session_params.replace:
            if session_params.rid is None:
                while self.req_nodes:
                    next(iter(self.req_nodes.values())).clear(self.req_nodes)
            else:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req_node.abort()
                    last_req = last_req_node.req
                    last_req_node.clear_children(self.req_nodes)
        else:
            if session_params.rid is not None:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req = last_req_node.req
                    if not last_req.finished():
                        abort = True
                        abort_message = "Session request is appending to a request that hasn't finished."
                        logging.warning(abort_message)

        carry_fill = None
        if last_req is not None:
            self._strip_bos_token(req, tokenizer)
            # In-place sharing is only safe for the plain streaming append:
            # streaming sessions allow a single inflight request, last_req has
            # finished, and the committed_* lengths recorded by finish_req let
            # _share_token_arrays trim away tokens appended by an aborted turn.
            # offset / drop_previous_output rewrite history and must copy.
            can_share_token_arrays = (
                self.streaming
                and self.committed_origin_len is not None
                and not session_params.drop_previous_output
                and not (session_params.offset and session_params.offset != 0)
            )
            if can_share_token_arrays:
                input_ids, input_ids_unpadded, carry_fill = self._share_token_arrays(
                    last_req, req.input_ids
                )
            else:
                input_ids, input_ids_unpadded = self._concat_token_arrays(
                    last_req, req, session_params
                )
        else:
            input_ids = req.input_ids
            input_ids_unpadded = req.input_ids

        if not abort and len(input_ids) == 0:
            abort = True
            abort_message = (
                "A session request must contain input tokens after restoring history."
            )

        new_req = Req(
            rid=req.rid,
            origin_input_text=None,
            origin_input_ids=input_ids,
            origin_input_ids_unpadded=input_ids_unpadded,
            sampling_params=req.sampling_params,
            lora_id=req.lora_id,
            session=self,
            custom_logit_processor=req.custom_logit_processor,
            stream=req.stream,
            return_logprob=req.return_logprob,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            return_sampling_mask=req.return_sampling_mask,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
            require_reasoning=req.require_reasoning,
            return_hidden_states=req.return_hidden_states,
            return_routed_experts=req.return_routed_experts,
            routed_experts_start_len=req.routed_experts_start_len,
            bootstrap_host=req.bootstrap_host,
            bootstrap_port=req.bootstrap_port,
            bootstrap_room=req.bootstrap_room,
            disagg_mode=disagg_mode,
            routed_dp_rank=req.routed_dp_rank,
            disagg_prefill_dp_rank=req.disagg_prefill_dp_rank,
            priority=req.priority,
            routing_key=req.routing_key,
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
            http_worker_ipc=req.http_worker_ipc,
            time_stats=req.time_stats,
        )
        if last_req is not None and not abort:
            new_req.multimodal_inputs = last_req.multimodal_inputs
        new_req.tokenizer = tokenizer
        if carry_fill is not None:
            new_req.full_untruncated_fill_ids = carry_fill

        if abort:
            new_req.set_finish_with_abort(abort_message)
        elif self.streaming:
            self.last_active_time = time.monotonic()
            # req_nodes is NOT updated here — finish_req() handles it.
            self._inflight = True
            self._inflight_rid = req.rid
        else:
            self.last_active_time = time.monotonic()
            previous = self.req_nodes.get(req.rid)
            new_req_node = SessionReqNode(new_req, last_req_node)
            self.req_nodes[req.rid] = new_req_node
            if previous is not None:
                self._unlink_req_node(previous)
                self.retire_req(previous.req)

        if not abort:
            self._active_reqs[id(new_req)] = new_req

        return new_req

    def finish_req(self, req):
        """Update req_nodes after a streaming request finishes successfully."""
        self._inflight = False
        self._inflight_rid = None
        self._active_reqs.pop(id(req), None)
        previous = None
        if self.req_nodes:
            [prev_node] = self.req_nodes.values()
            prev_node.req.session = None
            previous = prev_node.req
            self.req_nodes.clear()
        self.req_nodes[req.rid] = SessionReqNode(req)
        if previous is not None and previous is not req:
            self.release_req_mm_inputs(previous)
        # Confirm this req's token arrays as the session's rollback point.
        self.committed_origin_len = len(req.origin_input_ids)
        self.committed_unpadded_len = len(req.origin_input_ids_unpadded)
        self.committed_fill_len = len(req.full_untruncated_fill_ids)

    def abort_req(self, rid: Optional[str] = None):
        """Retire an aborted turn after its KV ownership has been released."""
        if self.streaming and rid is not None and self._inflight_rid != rid:
            return
        target_rid = self._inflight_rid if rid is None else rid
        self._inflight = False
        self._inflight_rid = None
        for req in list(self._active_reqs.values()):
            if req.rid == target_rid:
                self._retire_aborted_req_mm_inputs(req)


class SessionController:
    def __init__(self, tree_cache: BasePrefixCache):
        self.sessions: Dict[str, Session] = {}
        self._last_reap_time: float = 0.0
        self.tree_cache = tree_cache

    def __contains__(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def open(self, recv_req: OpenSessionReqInput) -> OpenSessionReqOutput:
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len,
                session_id,
                streaming=bool(recv_req.streaming),
                timeout=recv_req.timeout,
            )
            log_info_on_rank0(
                logger, f"Session opened: {session_id} (active={len(self.sessions)})"
            )
            return OpenSessionReqOutput(session_id=session_id, success=True)

    def close(self, recv_req: CloseSessionReqInput):
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            self._close(session_id)

    def _close(self, session_id: str):
        session = self.sessions[session_id]
        req = None
        if session.streaming and session.req_nodes:
            assert len(session.req_nodes) == 1
            [last_node] = session.req_nodes.values()
            req = last_node.req

        if session.has_unfinished_request():
            # A finish reason may precede deferred KV release. Keep media and
            # session storage until the scheduler retires every active owner.
            session.close_on_finish = True
            logger.info(
                "Deferring session close for %s (unfinished request)",
                session_id,
            )
            return

        # No owning request -- safe to release immediately.
        if session.streaming and session.req_nodes:
            req = next(iter(session.req_nodes.values())).req
            req.session = None

        # Release multimodal features held by session requests.
        # Session reqs skip the normal mm cleanup path (scheduler and
        # output_processor) so features stay alive until the session closes.
        seen_items = set()
        for retained_req in session._iter_reqs():
            mm = retained_req.multimodal_inputs
            if mm is not None:
                owned = []
                for item in mm.mm_items:
                    if id(item) not in seen_items:
                        seen_items.add(id(item))
                        owned.append(item)
                if owned:
                    mm.release_features(owned)
            retained_req.multimodal_inputs = None
        session._retired_reqs.clear()

        self.tree_cache.release_radix_session(session_id)
        self.tree_cache.release_session(session_id)
        del self.sessions[session_id]
        log_info_on_rank0(
            logger, f"Session closed: {session_id} (active={len(self.sessions)})"
        )

    def maybe_reap(self, now: float, interval: float = 1.0):
        plan = self.plan_reap(now, interval)
        if plan is not None:
            self.apply_reap(plan)

    def plan_reap(self, now: float, interval: float = 1.0) -> Optional[SessionReapPlan]:
        # reap sessions every second
        if now - self._last_reap_time <= interval:
            return None
        self._last_reap_time = now

        # Finish deferred closes for sessions whose requests completed.
        deferred = [
            sid
            for sid, session in self.sessions.items()
            if session.close_on_finish and not session.has_unfinished_request()
        ]
        timed_out = [
            sid for sid, session in self.sessions.items() if session.is_timed_out()
        ]
        if not deferred and not timed_out:
            return None
        return SessionReapPlan(deferred=deferred, timed_out=timed_out)

    def apply_reap(self, plan: SessionReapPlan) -> None:
        for sid in plan.deferred:
            session = self.sessions.get(sid)
            if session is None or not session.close_on_finish:
                continue
            log_info_on_rank0(
                logger, f"Deferred close ready for session {sid}, releasing."
            )
            # Reset close_on_finish so _close proceeds with the release.
            session.close_on_finish = False
            self._close(sid)
        for sid in plan.timed_out:
            if sid not in self.sessions:
                continue
            log_info_on_rank0(logger, f"Session {sid} timed out, closing.")
            self._close(sid)

    @staticmethod
    def _all_requests_finished(session: Session) -> bool:
        return not session.has_unfinished_request()

    @staticmethod
    def adjust_mm_offsets(recv_req: TokenizedGenerateReqInput, req: Req, image_inputs):
        # For session requests, adjust mm_inputs offsets by the prefix length.
        # Session.create_req prepends previous context to origin_input_ids,
        # so offsets from the new prompt need to be shifted.
        if len(recv_req.input_ids) >= len(req.origin_input_ids):
            return
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        for mm_item in image_inputs.mm_items:
            if mm_item.offsets:
                mm_item.offsets = [
                    (start + prefix_len, end + prefix_len)
                    for start, end in mm_item.offsets
                ]
