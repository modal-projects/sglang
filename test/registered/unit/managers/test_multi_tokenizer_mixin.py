import unittest
from collections import deque
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchStrOutput,
    ContinueGenerationReqInput,
    PauseContinueBroadcastAckReq,
    PauseContinueBroadcastCompleteReq,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    TokenizerWorker,
    _handle_output_by_index,
    get_tokenizer_worker_class,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CustomTokenizerWorker(TokenizerWorker):
    pass


class NotAWorker:
    pass


class DefaultServerArgs:
    def get_tokenizer_worker_class(self):
        return TokenizerWorker


class CustomServerArgs:
    def get_tokenizer_worker_class(self):
        return CustomTokenizerWorker


class InvalidServerArgs:
    def get_tokenizer_worker_class(self):
        return NotAWorker


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_token_sampling_mask=[[], []],
        output_token_sampling_logprobs=[[], []],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        retraction_counts=[0, 0],
    )


class TestMultiTokenizerMixin(unittest.TestCase):
    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

    def test_get_tokenizer_worker_class_uses_default(self):
        self.assertIs(get_tokenizer_worker_class(DefaultServerArgs()), TokenizerWorker)

    def test_get_tokenizer_worker_class_resolves_custom_class(self):
        self.assertIs(
            get_tokenizer_worker_class(CustomServerArgs()),
            CustomTokenizerWorker,
        )

    def test_get_tokenizer_worker_class_rejects_non_worker(self):
        with self.assertRaisesRegex(TypeError, "TokenizerWorker"):
            get_tokenizer_worker_class(InvalidServerArgs())


class TestPauseContinueBarrier(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        self.router.all_worker_ipcs = {"worker-a", "worker-b"}
        self.router._tokenizer_worker_num = 2
        self.router._pending_pause_continue = None
        self.router._waiting_pause_continue = None
        self.router._deferred_scheduler_requests = deque()
        self.router.socket_mapping = mock.Mock()
        self.router.send_to_scheduler = object()

    async def _ack(self, request_id, worker_ipc_name):
        await self.router._handle_pause_continue_ack(
            PauseContinueBroadcastAckReq(
                request_id=request_id,
                worker_ipc_name=worker_ipc_name,
            )
        )

    async def test_continue_reaches_scheduler_after_every_worker_acknowledges(self):
        request = ContinueGenerationReqInput(
            rid="continue-1",
            http_worker_ipc="worker-a",
        )
        with mock.patch(
            "sglang.srt.managers.multi_tokenizer_mixin.async_sock_send",
            new_callable=mock.AsyncMock,
        ) as send:
            await self.router._start_pause_continue(request)

            broadcasts = [
                call.args[1]
                for call in self.router.socket_mapping.send_output.call_args_list
            ]
            self.assertEqual(len(broadcasts), 2)
            self.assertTrue(
                all(isinstance(obj, PauseContinueBroadcastReq) for obj in broadcasts)
            )

            await self._ack("continue-1", "worker-a")
            send.assert_not_awaited()

            await self._ack("continue-1", "worker-b")
            send.assert_awaited_once_with(self.router.send_to_scheduler, request)

        complete = self.router.socket_mapping.send_output.call_args_list[-1].args[1]
        self.assertIsInstance(complete, PauseContinueBroadcastCompleteReq)
        self.assertEqual(complete.request_id, "continue-1")
        self.assertIsNone(self.router._pending_pause_continue)

    async def test_pause_continue_transitions_remain_fifo(self):
        pause = PauseGenerationReqInput(
            rid="pause-1",
            http_worker_ipc="worker-a",
            mode="in_place",
        )
        resume = ContinueGenerationReqInput(
            rid="continue-2",
            http_worker_ipc="worker-b",
        )
        with mock.patch(
            "sglang.srt.managers.multi_tokenizer_mixin.async_sock_send",
            new_callable=mock.AsyncMock,
        ) as send:
            await self.router._start_pause_continue(pause)
            self.router._deferred_scheduler_requests.append(resume)

            await self._ack("pause-1", "worker-a")
            await self._ack("pause-1", "worker-b")
            send.assert_awaited_once_with(self.router.send_to_scheduler, pause)
            self.assertEqual(self.router._pending_pause_continue[0], "continue-2")

            await self._ack("continue-2", "worker-b")
            await self._ack("continue-2", "worker-a")
            self.assertEqual(
                [call.args[1] for call in send.await_args_list],
                [pause, resume],
            )

    async def test_control_message_waits_for_scheduler_pause(self):
        pause = PauseGenerationReqInput(
            rid="pause-2",
            http_worker_ipc="worker-a",
            mode="in_place",
        )
        control_request = object()
        with mock.patch(
            "sglang.srt.managers.multi_tokenizer_mixin.async_sock_send",
            new_callable=mock.AsyncMock,
        ) as send:
            await self.router._start_pause_continue(pause)
            self.router._deferred_scheduler_requests.append(control_request)

            await self._ack("pause-2", "worker-a")
            send.assert_not_awaited()
            await self._ack("pause-2", "worker-b")

            self.assertEqual(
                [call.args[1] for call in send.await_args_list],
                [pause, control_request],
            )

    async def test_abort_requests_are_part_of_abort_pause_barrier(self):
        pause = PauseGenerationReqInput(
            rid="abort-1",
            http_worker_ipc="worker-a",
            mode="abort",
        )
        await self.router._start_pause_continue(pause)

        broadcasts = [
            call.args[1]
            for call in self.router.socket_mapping.send_output.call_args_list
        ]
        self.assertTrue(all(obj.abort_all for obj in broadcasts))
        self.assertTrue(
            self.router._is_abort_for_pending_pause(AbortReq(rid="", abort_all=True))
        )
        self.assertFalse(
            self.router._is_abort_for_pending_pause(
                AbortReq(rid="request-1", abort_all=False)
            )
        )


if __name__ == "__main__":
    unittest.main()
