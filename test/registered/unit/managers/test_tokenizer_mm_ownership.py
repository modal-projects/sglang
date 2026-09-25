"""Request failures and sample expansion must preserve transport lease ownership."""

import asyncio
import pickle
import threading
import unittest
from array import array
from contextlib import contextmanager, nullcontext
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
import zmq
import zmq.asyncio

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    TokenizedGenerateReqInput,
    async_sock_recv,
)
from sglang.srt.managers.mm_utils import unwrap_shm_features, wrap_shm_features
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.llava import LlavaMultimodalProcessor
from sglang.srt.multimodal.transport import cuda_ipc
from sglang.srt.multimodal.transport.producer_lifecycle import (
    cancel_undispatched_inputs,
    detach_for_parallel_sampling,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.cuda_vmm_transport_utils import CudaVmmFeatureTransport

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _CpuLeaseStorage:
    def __init__(self):
        self.control_words_per_slot = 3
        self.base_address = 0
        self._lock = threading.Lock()
        self._occupied = {}
        self.cancelled = []

    def cancel_lease(self, *, ready_byte_offset, ack_byte_offset, generation):
        slot = ready_byte_offset // 12
        lease = self._occupied.pop(slot)
        assert lease.generation == generation
        assert lease.ack_byte_offset == ack_byte_offset
        self.cancelled.append(slot)


def _make_pool():
    pool = cuda_ipc.MmItemMemoryPool.__new__(cuda_ipc.MmItemMemoryPool)
    pool.device_id = 0
    pool._pool_id = ("synthetic-pool",)
    pool._export_lock = threading.Lock()
    pool._exports = {}
    pool._cancel_states = {}
    pool.memory_pool = torch.zeros(1024, dtype=torch.uint8)
    pool._pool = _CpuLeaseStorage()
    return pool


def _make_output(pool, slot=0):
    feature = torch.tensor([slot + 3, slot + 7], dtype=torch.float32)
    start = 128 + slot * 16
    data = pool.memory_pool[start : start + 8]
    data.copy_(feature.view(torch.uint8))
    pool._pool._occupied[slot] = SimpleNamespace(
        start=start,
        nbytes=8,
        generation=1,
        ready_byte_offset=slot * 12,
        ack_byte_offset=slot * 12 + 4,
    )
    pool._exports[slot * 12] = (1, (), set())
    pool._cancel_states.pop(slot * 12, None)
    proxy = cuda_ipc.CudaIpcTensorTransportProxy(
        data=data,
        info_data=feature,
        pool_ipc_handle=pool._pool_id,
        pool_byte_offset=start,
        ready_byte_offset=slot * 12,
        ack_byte_offset=slot * 12 + 4,
        generation=1,
        total_consumer_count=2,
        use_pool_handle_cache=False,
    )
    return MultimodalProcessorOutput(
        input_ids=[1],
        mm_items=[
            MultimodalDataItem(
                modality=Modality.IMAGE, feature=proxy, precomputed_embeddings=proxy
            )
        ],
    )


def _tokenized(mm_inputs, rid="request"):
    obj = TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [1]),
        input_embeds=None,
        mm_inputs=mm_inputs,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    obj.time_stats = APIServerReqTimeStats()
    return obj


def _manager(pool, output=None):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.mm_processor = SimpleNamespace(
        use_cuda_ipc=True,
        cudaipc_mmfeature_pool=pool,
        prefer_tokenized_input=True,
        process_mm_data_async=AsyncMock(return_value=output),
    )
    manager.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[]))
    manager.max_req_input_len = 16
    manager.tokenizer = None
    manager._validate_mm_limits = Mock()
    manager._validate_one_request = Mock()
    manager._create_tokenized_object = lambda *args: _tokenized(args[4])
    manager.rid_to_state = {}
    manager.encoder_dispatch_ready = {}
    manager.cuda_vmm_feature_transport = CudaVmmFeatureTransport.__new__(
        CudaVmmFeatureTransport
    )
    manager.cuda_vmm_feature_transport.pool = None
    manager._async_dispatch_to_scheduler = AsyncMock()
    return manager


@contextmanager
def _llava_processor(transport, pool, *, skip_mm_pool=False):
    override = get_context().override_server_args(
        mm_feature_transport=transport,
        mm_process_config={},
        mm_preprocess_cache_size_mb=0,
        mm_processor_worker_num=1,
        mm_io_worker_num=1,
        tokenizer_worker_num=1,
    )
    override.install()
    try:
        with (
            patch.object(
                BaseMultimodalProcessor,
                "__init__",
                autospec=True,
                side_effect=BaseMultimodalProcessor.__init__,
            ) as initialize,
            patch(
                "sglang.srt.multimodal.processors.base_processor.concurrent.futures.ThreadPoolExecutor"
            ) as io_factory,
            patch.object(
                BaseMultimodalProcessor, "_create_cpu_executor"
            ) as cpu_factory,
            patch(
                "sglang.srt.multimodal.processors.base_processor.MmItemMemoryPool",
                return_value=pool,
            ) as pool_factory,
        ):
            processor = LlavaMultimodalProcessor(
                SimpleNamespace(
                    vision_config=SimpleNamespace(model_type="clip_vision_model"),
                    text_config=SimpleNamespace(),
                ),
                SimpleNamespace(base_gpu_id=0, tp_size=2),
                SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: [])),
                transport_mode=None,
                skip_mm_pool=skip_mm_pool,
            )
            processor.inner.shutdown = Mock(wraps=processor.inner.shutdown)
            processor.inner.clear_preprocess_cache = Mock(
                wraps=processor.inner.clear_preprocess_cache
            )
            try:
                yield SimpleNamespace(
                    processor=processor,
                    initialize=initialize,
                    pool_factory=pool_factory,
                    io_executor=io_factory.return_value,
                    cpu_executor=cpu_factory.return_value,
                )
            finally:
                processor.shutdown()
    finally:
        override.restore()


class TestTokenizerMultimodalOwnership(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            speculative_algorithm=None,
            language_only=False,
            language_model_only=False,
            enable_tokenizer_batch_encode=False,
        )
        override.install()
        self.addCleanup(override.restore)
        for target, replacement in (
            ("torch.cuda.device", lambda *_: nullcontext()),
            ("torch.cuda.current_stream", Mock(return_value=Mock())),
            ("torch.cuda.ipc_collect", lambda: None),
            ("sglang.srt.multimodal.transport.cuda_ipc.stream_wait_value32", Mock()),
        ):
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_llava_wrapper_has_one_transport_and_shutdown_owner(self):
        for transport, skip_mm_pool in (
            ("cpu", False),
            ("cuda_vmm", False),
            ("cuda_ipc", False),
            ("cuda_ipc", True),
        ):
            with self.subTest(transport=transport, skip_mm_pool=skip_mm_pool):
                pool = Mock()
                owns_pool = transport == "cuda_ipc" and not skip_mm_pool
                with _llava_processor(
                    transport, pool, skip_mm_pool=skip_mm_pool
                ) as fixture:
                    processor = fixture.processor
                    manager = _manager(None)
                    manager.mm_processor = processor
                    fixture.initialize.assert_called_once()
                    self.assertIs(fixture.initialize.call_args.args[0], processor.inner)
                    self.assertEqual(fixture.pool_factory.call_count, int(owns_pool))
                    self.assertEqual(processor.mm_feature_transport, transport)
                    self.assertEqual(processor.use_cuda_ipc, transport == "cuda_ipc")
                    self.assertEqual(
                        processor.keep_mm_features_on_device, transport != "cpu"
                    )
                    expected_pool = pool if owns_pool else None
                    self.assertIs(processor.cudaipc_mmfeature_pool, expected_pool)
                    self.assertIs(manager._mm_feature_pool(), expected_pool)
                    self.assertNotIn("cudaipc_mmfeature_pool", vars(processor))
                    processor.clear_preprocess_cache()
                    processor.inner.clear_preprocess_cache.assert_called_once_with()
                processor.inner.shutdown.assert_called_once_with()
                self.assertEqual(processor.inner.clear_preprocess_cache.call_count, 2)
                fixture.io_executor.shutdown.assert_called_once_with(
                    wait=False, cancel_futures=True
                )
                fixture.cpu_executor.shutdown.assert_called_once_with(
                    wait=False, cancel_futures=True
                )
                self.assertEqual(pool.shutdown.call_count, int(owns_pool))
                self.assertIs(manager._mm_feature_pool(), expected_pool)

    def test_llava_wrapper_uses_inner_pool_for_snapshot_and_cancellation(self):
        pool = _make_pool()
        pool.shutdown = Mock()
        output = _make_output(pool)
        proxy = output.mm_items[0].feature
        pool.copy_proxy_to_cpu = Mock(wraps=pool.copy_proxy_to_cpu)
        pool.cancel_proxy = Mock(wraps=pool.cancel_proxy)
        with (
            _llava_processor("cuda_ipc", pool) as fixture,
            patch.object(proxy, "acknowledge_consumption") as acknowledge,
        ):
            manager = _manager(None)
            manager.mm_processor = fixture.processor
            owner = manager._mm_feature_pool()
            self.assertIs(owner, fixture.processor.inner.cudaipc_mmfeature_pool)
            detached = detach_for_parallel_sampling(owner, [output, output])
            cancel_undispatched_inputs(owner, [output, output])
            pool.copy_proxy_to_cpu.assert_called_once_with(proxy)
            pool.cancel_proxy.assert_called_once_with(proxy)
            acknowledge.assert_not_called()
            self.assertEqual(pool._pool.cancelled, [0])
            self.assertFalse(pool._pool._occupied)
            pool.memory_pool.zero_()
            for clone in detached:
                self.assertEqual(clone.mm_items[0].feature.tolist(), [3.0, 7.0])
                self.assertIs(
                    clone.mm_items[0].feature, clone.mm_items[0].precomputed_embeddings
                )
            self.assertIsNone(output.mm_items[0].feature)
            self.assertIsNone(output.mm_items[0].precomputed_embeddings)
        pool.shutdown.assert_called_once_with()

    def test_clone_detachment_survives_pool_reuse_and_deduplicates_aliases(self):
        """Samples retain their values after their shared source lease is recycled."""
        pool = _make_pool()
        output = _make_output(pool)
        detached = detach_for_parallel_sampling(pool, [output, output])
        cancel_undispatched_inputs(pool, [output, output])
        pool.memory_pool.zero_()
        self.assertEqual(pool._pool.cancelled, [0])
        self.assertIsNone(output.mm_items[0].feature)
        for result in detached:
            item = result.mm_items[0]
            self.assertEqual(item.feature.tolist(), [3, 7])
            self.assertEqual(item.feature.device.type, "cpu")
            self.assertIs(item.feature, item.precomputed_embeddings)
        self.assertIsNot(detached[0].mm_items[0], detached[1].mm_items[0])

    def test_tokenization_failures_release_every_proxy(self):
        """Hash, validation and construction failures occur after lease publication."""
        for failure in ("hash", "validate", "create", "cancel"):
            with self.subTest(failure=failure):
                pool = _make_pool()
                output = _make_output(pool)
                manager = _manager(pool, output)
                obj = GenerateReqInput(input_ids=[1], image_data=["synthetic"])
                error = (
                    asyncio.CancelledError()
                    if failure == "cancel"
                    else ValueError(failure)
                )
                if failure in ("validate", "cancel"):
                    manager._validate_one_request.side_effect = error
                elif failure == "create":
                    manager._create_tokenized_object = Mock(side_effect=error)
                with (
                    envs.SGLANG_MM_PRECOMPUTE_HASH.override(failure == "hash"),
                    patch.object(
                        MultimodalDataItem, "set_pad_value", side_effect=error
                    ),
                    self.assertRaises(type(error)),
                ):
                    asyncio.run(manager._tokenize_one_request(obj))
                self.assertEqual(pool._pool.cancelled, [0])
                self.assertFalse(pool._pool._occupied)

    def test_preprocessing_cancellation_waits_for_late_output(self):
        """A canceled request cleans leases published later by its processor worker."""
        pool = _make_pool()
        manager = _manager(pool)

        async def run():
            started = asyncio.Event()
            finish = asyncio.Event()

            async def process(**kwargs):
                started.set()
                await finish.wait()
                return _make_output(pool)

            manager.mm_processor.process_mm_data_async = process
            task = asyncio.create_task(
                manager._tokenize_one_request(
                    GenerateReqInput(input_ids=[1], image_data=["synthetic"])
                )
            )
            await started.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertEqual(pool._pool.cancelled, [0])
        self.assertFalse(pool._pool._occupied)

    def test_dispatch_failure_uses_unserialized_outputs(self):
        """Single and batch send failures retain raw outputs for producer cleanup."""
        for batch in (False, True):
            with self.subTest(batch=batch):
                pool = _make_pool()
                outputs = [
                    _make_output(pool, slot) for slot in range(2 if batch else 1)
                ]
                requests = [
                    _tokenized(output, str(i)) for i, output in enumerate(outputs)
                ]
                manager = _manager(pool)
                manager._async_dispatch_to_scheduler.side_effect = ValueError(
                    "send failed"
                )
                with self.assertRaisesRegex(ValueError, "send failed"):
                    asyncio.run(
                        manager._send_batch_request(requests)
                        if batch
                        else manager._send_one_request(requests[0])
                    )
                self.assertEqual(pool._pool.cancelled, list(range(len(outputs))))
                self.assertTrue(
                    all(output.mm_items[0].feature is None for output in outputs)
                )

    def test_failed_dispatch_unlinks_shared_memory_from_cpu_clones(self):
        """Detached CPU samples must release SHM after an asynchronous socket failure."""
        for batch in (False, True):
            with self.subTest(batch=batch):
                pool = _make_pool()
                output = _make_output(pool)
                clone = detach_for_parallel_sampling(pool, [output])[0]
                cancel_undispatched_inputs(pool, [output])
                manager = _manager(pool)
                manager.tokenizer_ipc_name = None
                segment_names = []

                async def dispatch(request):
                    item = (request.batch[0] if batch else request).mm_inputs.mm_items[
                        0
                    ]
                    segment_names.extend(
                        [item.feature.shm_name, item.precomputed_embeddings.shm_name]
                    )
                    await TokenizerManager._async_dispatch_to_scheduler(
                        manager, request
                    )

                manager._async_dispatch_to_scheduler = dispatch

                async def run():
                    context = zmq.asyncio.Context()
                    sender = context.socket(zmq.PUSH)
                    sender.setsockopt(zmq.SNDTIMEO, 10)
                    manager.send_to_scheduler = sender
                    request = _tokenized(clone)
                    try:
                        with self.assertRaises(zmq.Again):
                            if batch:
                                wrap_shm_features(request)
                                await manager._send_batch_request([request])
                            else:
                                await manager._send_one_request(request)
                    finally:
                        sender.close(linger=0)
                        context.term()

                asyncio.run(run())
                self.assertEqual(len(segment_names), 2)
                remaining = []
                for name in segment_names:
                    try:
                        segment = shared_memory.SharedMemory(name=name)
                    except FileNotFoundError:
                        continue
                    remaining.append(name)
                    segment.close()
                    segment.unlink()
                self.assertFalse(
                    remaining, "Undispatched shared memory remained allocated"
                )
                self.assertEqual(pool._pool.cancelled, [0])

    def test_post_dispatch_failure_keeps_consumer_ownership(self):
        """Bookkeeping failure after dispatch must never cancel scheduler-owned leases."""
        for batch in (False, True):
            with self.subTest(batch=batch):
                pool = _make_pool()
                output = _make_output(pool)
                manager = _manager(pool)
                manager._mark_state_dispatched = Mock(
                    side_effect=ValueError("stats failed")
                )
                with self.assertRaisesRegex(ValueError, "stats failed"):
                    asyncio.run(
                        manager._send_batch_request([_tokenized(output)])
                        if batch
                        else manager._send_one_request(_tokenized(output))
                    )
                self.assertEqual(pool._pool.cancelled, [])
                self.assertEqual(len(pool._pool._occupied), 1)

    def test_socket_acceptance_settles_before_ownership_changes(self):
        """A pending send cannot release leases until delivery or socket failure."""
        for batch in (False, True):
            for cancel in (False, True):
                for accept in (False, True):
                    with self.subTest(batch=batch, cancel=cancel, accept=accept):
                        pool = _make_pool()
                        outputs = [
                            _make_output(pool, slot)
                            for slot in range(2 if batch else 1)
                        ]
                        requests = [
                            _tokenized(output, str(i))
                            for i, output in enumerate(outputs)
                        ]
                        manager = _manager(pool)
                        manager.tokenizer_ipc_name = None
                        manager._async_dispatch_to_scheduler = (
                            TokenizerManager._async_dispatch_to_scheduler.__get__(
                                manager
                            )
                        )
                        for request in requests:
                            manager.rid_to_state[request.rid] = SimpleNamespace(
                                dispatched=False, encoder_dispatch_ready=None
                            )

                        async def run():
                            context = zmq.asyncio.Context()
                            sender = context.socket(zmq.PUSH)
                            receiver = context.socket(zmq.PULL)
                            sender.bind("inproc://producer-ownership")
                            sender.setsockopt(zmq.SNDTIMEO, 100 if not accept else 2000)
                            manager.send_to_scheduler = sender
                            task = asyncio.create_task(
                                manager._send_batch_request(requests)
                                if batch
                                else manager._send_one_request(requests[0])
                            )
                            try:
                                # No peer exists yet, so the actual send Future is pending.
                                await asyncio.sleep(0)
                                await asyncio.sleep(0)
                                self.assertFalse(task.done())
                                self.assertFalse(
                                    any(
                                        state.dispatched
                                        for state in manager.rid_to_state.values()
                                    )
                                )
                                if cancel:
                                    task.cancel()
                                    await asyncio.sleep(0)
                                    task.cancel()
                                    await asyncio.sleep(0)
                                    self.assertFalse(task.done())
                                    self.assertEqual(pool._pool.cancelled, [])
                                if accept:
                                    receiver.connect("inproc://producer-ownership")
                                if cancel:
                                    with self.assertRaises(
                                        asyncio.CancelledError
                                    ) as raised:
                                        await task
                                    if not accept:
                                        self.assertIsInstance(
                                            raised.exception.__cause__, zmq.Again
                                        )
                                elif not accept:
                                    with self.assertRaises(zmq.Again):
                                        await task
                                else:
                                    await task
                                if accept:
                                    received = await asyncio.wait_for(
                                        async_sock_recv(receiver), timeout=2
                                    )
                                    received_requests = (
                                        received.batch if batch else [received]
                                    )
                                    self.assertEqual(
                                        [request.rid for request in received_requests],
                                        [request.rid for request in requests],
                                    )
                                    self.assertTrue(
                                        all(
                                            state.dispatched
                                            for state in manager.rid_to_state.values()
                                        )
                                    )
                                    self.assertEqual(pool._pool.cancelled, [])
                                    self.assertEqual(
                                        len(pool._pool._occupied), len(outputs)
                                    )
                                else:
                                    self.assertEqual(
                                        pool._pool.cancelled, list(range(len(outputs)))
                                    )
                                    self.assertFalse(pool._pool._occupied)
                            finally:
                                sender.close(linger=0)
                                receiver.close(linger=0)
                                if not task.done():
                                    task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                                context.term()

                        asyncio.run(run())

    def test_cleanup_failure_keeps_original_error_and_retryable_proxies(self):
        """A failed cancel keeps its proxy while other leases are still released."""
        pool = _make_pool()
        outputs = [_make_output(pool, slot) for slot in range(2)]
        original_proxy = outputs[0].mm_items[0].feature
        original_cancel = pool.cancel_proxy

        def cancel(proxy):
            if proxy is original_proxy:
                raise RuntimeError("cleanup failed")
            original_cancel(proxy)

        pool.cancel_proxy = cancel
        manager = _manager(pool)
        send_error = ValueError("send failed")
        manager._async_dispatch_to_scheduler.side_effect = send_error
        with self.assertRaises(ValueError) as raised:
            asyncio.run(
                manager._send_batch_request([_tokenized(output) for output in outputs])
            )
        self.assertIs(raised.exception, send_error)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(pool._pool.cancelled, [1])
        self.assertIs(outputs[0].mm_items[0].feature, original_proxy)
        self.assertIs(outputs[0].mm_items[0].precomputed_embeddings, original_proxy)
        self.assertIsNone(outputs[1].mm_items[0].feature)
        pool.cancel_proxy = original_cancel
        cancel_undispatched_inputs(pool, outputs)
        self.assertEqual(pool._pool.cancelled, [1, 0])
        self.assertFalse(pool._pool._occupied)

    def test_dispatch_does_not_mark_a_reused_request_id(self):
        """Send completion must update its original state after a fast RID reuse."""
        for batch in (False, True):
            with self.subTest(batch=batch):
                pool = _make_pool()
                output = _make_output(pool)
                request = _tokenized(output)
                manager = _manager(pool)
                original = SimpleNamespace(
                    dispatched=False, encoder_dispatch_ready=None
                )
                replacement = SimpleNamespace(
                    dispatched=False, encoder_dispatch_ready=None
                )
                original_ready = threading.Event()
                replacement_ready = threading.Event()
                original.encoder_dispatch_ready = original_ready
                replacement.encoder_dispatch_ready = replacement_ready
                manager.rid_to_state[request.rid] = original
                manager.encoder_dispatch_ready[request.rid] = original_ready

                async def dispatch(_):
                    await asyncio.sleep(0)
                    manager.rid_to_state[request.rid] = replacement
                    manager.encoder_dispatch_ready[request.rid] = replacement_ready

                manager._async_dispatch_to_scheduler = dispatch
                asyncio.run(
                    manager._send_batch_request([request])
                    if batch
                    else manager._send_one_request(request)
                )
                self.assertTrue(original.dispatched)
                self.assertFalse(replacement.dispatched)
                self.assertIs(
                    manager.encoder_dispatch_ready[request.rid], replacement_ready
                )
                self.assertFalse(replacement_ready.is_set())
                if not batch:
                    self.assertTrue(original_ready.is_set())
                self.assertEqual(pool._pool.cancelled, [])

    def test_parallel_tokenization_failure_cleans_completed_sibling(self):
        """A sibling failure cannot orphan an output completed before gather raises."""
        pool = _make_pool()
        manager = _manager(pool)
        obj = GenerateReqInput(input_ids=[[1], [2]], sampling_params={"n": 2})
        obj.normalize_batch_and_arguments()

        async def tokenize(request):
            if request.input_ids == [1]:
                return _tokenized(_make_output(pool))
            await asyncio.sleep(0)
            raise ValueError("sibling failed")

        manager._tokenize_one_request = tokenize
        with self.assertRaisesRegex(ValueError, "sibling failed"):
            asyncio.run(manager._handle_batch_request(obj).__anext__())
        self.assertEqual(pool._pool.cancelled, [0])
        self.assertFalse(pool._pool._occupied)

    def test_pretokenized_embedding_batch_releases_earlier_outputs_on_failure(self):
        """A later multimodal embedding failure cannot orphan preceding leases."""
        for failure in (ValueError("processor failed"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                pool = _make_pool()
                manager = _manager(pool)
                manager.is_generation = False
                request = EmbeddingReqInput(
                    input_ids=[[1], [2], [3]],
                    image_data=[["synthetic"]] * 3,
                )
                request.normalize_batch_and_arguments()
                processed = []

                async def process(**kwargs):
                    value = kwargs["request_obj"].input_ids[0]
                    processed.append(value)
                    if value == 2:
                        raise failure
                    return _make_output(pool, value - 1)

                manager.mm_processor.process_mm_data_async = process
                self.assertTrue(
                    manager._should_use_batch_tokenization(request.batch_size, request)
                )
                with self.assertRaises(type(failure)):
                    asyncio.run(
                        manager._batch_tokenize_and_process(request.batch_size, request)
                    )
                self.assertEqual(processed, [1, 2])
                self.assertEqual(pool._pool.cancelled, [0])
                self.assertFalse(pool._pool._occupied)

    def test_parallel_gather_cancellation_cleans_completed_and_late_outputs(self):
        """Cancellation during gather cleans both already-returned and late leases."""
        pool = _make_pool()
        manager = _manager(pool)
        obj = GenerateReqInput(input_ids=[[1], [2]], sampling_params={"n": 2})
        obj.normalize_batch_and_arguments()

        async def run():
            started = asyncio.Event()
            finish = asyncio.Event()

            async def process(**kwargs):
                slot = kwargs["request_obj"].input_ids[0] - 1
                if slot:
                    started.set()
                    await finish.wait()
                return _make_output(pool, slot)

            manager.mm_processor.process_mm_data_async = process
            for request in [obj[i] for i in range(obj.batch_size)]:
                request.image_data = ["synthetic"]
            task = asyncio.create_task(manager._handle_batch_request(obj).__anext__())
            await started.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertCountEqual(pool._pool.cancelled, [0, 1])
        self.assertFalse(pool._pool._occupied)

    def test_parallel_sampling_sends_owned_tensors_for_prefix_and_samples(self):
        """Prefix warming and every sample must survive reuse of the original lease."""
        pool = _make_pool()
        output = _make_output(pool)
        manager = _manager(pool, output)
        obj = GenerateReqInput(
            input_ids=[[1]], image_data=[["synthetic"]], sampling_params={"n": 2}
        )
        obj.normalize_batch_and_arguments()
        sent = []

        def init_state(request):
            manager.rid_to_state[request.rid] = SimpleNamespace(
                time_stats=APIServerReqTimeStats(),
                dispatched=False,
                encoder_dispatch_ready=None,
            )

        async def response(request, http_request):
            yield {"rid": request.rid}

        async def dispatch(request):
            pool.memory_pool.zero_()
            received = pickle.loads(pickle.dumps(request))
            unwrap_shm_features(received)
            sent.append(request)
            self.assertEqual(pool._pool.cancelled, [0])
            self.assertEqual(received.mm_inputs.mm_items[0].feature.tolist(), [3, 7])

        manager._init_req_state = init_state
        manager._wait_one_response = response
        manager._async_dispatch_to_scheduler = dispatch
        init_state(obj[0])
        result = asyncio.run(manager._handle_batch_request(obj).__anext__())
        self.assertEqual(len(result), 2)
        self.assertEqual(len(sent), 3)
        self.assertEqual(len({id(item.mm_inputs.mm_items[0]) for item in sent}), 3)
        self.assertFalse(pool._pool._occupied)

    def test_single_dispatch_preserves_the_original_lease(self):
        """The ordinary one-sample path transfers its lease to the scheduler."""
        pool = _make_pool()
        output = _make_output(pool)
        manager = _manager(pool, output)
        request = asyncio.run(
            manager._tokenize_one_request(
                GenerateReqInput(input_ids=[1], image_data=["synthetic"])
            )
        )
        asyncio.run(manager._send_one_request(request))
        self.assertEqual(pool._pool.cancelled, [])
        self.assertIs(request.mm_inputs, output)
        self.assertEqual(len(pool._pool._occupied), 1)

    def test_parallel_detachment_failure_cleans_all_originals(self):
        """An invalid second lease does not leak the first or any later original."""
        pool = _make_pool()
        outputs = [_make_output(pool, slot) for slot in range(2)]
        manager = _manager(pool)
        obj = GenerateReqInput(input_ids=[[1], [2]], sampling_params={"n": 2})
        obj.normalize_batch_and_arguments()
        manager._tokenize_one_request = AsyncMock(
            side_effect=[_tokenized(x) for x in outputs]
        )
        pool._pool._occupied[1].start += 1
        manager._init_req_state = Mock()
        manager._async_dispatch_to_scheduler.side_effect = AssertionError(
            "Dispatched an inactive lease"
        )
        with self.assertRaisesRegex(RuntimeError, "inactive"):
            asyncio.run(manager._handle_batch_request(obj).__anext__())
        self.assertEqual(pool._pool.cancelled, [0, 1])
        self.assertFalse(pool._pool._occupied)


if __name__ == "__main__":
    unittest.main()
