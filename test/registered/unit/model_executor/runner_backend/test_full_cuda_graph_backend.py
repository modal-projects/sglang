"""Unit tests for ``FullCudaGraphBackend.capture_one`` profiling hooks — CPU-only.

These cover the changes from the "cuda graph profile traces" PR that wire the
runner's torch profiler into the capture loop:

  * When profiling is disabled, ``capture_one`` runs exactly two warmups + one
    capture and never touches a profiler (behavior-identical to before the PR).
  * When the runner exposes an active ``_profiler`` (per-bs capture profiling,
    ``--enable-profile-cuda-graph`` + ``SGLANG_GRAPH_BATCH_CAPTURE``),
    ``capture_one`` calls ``profiler.step()`` past the two warmups and once after
    the capture (schedule ``wait=2, warmup=0, active=1``). The captured forward is
    NOT wrapped in a ``record_function``; per-bs trace naming is handled by the
    profiler's ``on_trace_ready`` callback instead.
  * The ``getattr`` guards mean a runner that sets the flag but has no
    ``_profiler`` attribute degrades gracefully (no stepping, no crash).

The real capture path needs CUDA (``torch.cuda.CUDAGraph`` + device graph
context), so those are mocked; the logic under test (call counts, ordering,
profiler stepping) is pure-Python and runs on CPU.
"""

import contextlib
import gc
import weakref
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
    _copy_output_to_buffer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Sentinel: distinguishes "runner has no _profiler attribute" from
# "_profiler is None" in the test fixtures.
_UNSET = object()


class _FakeGraphCtx:
    """Stand-in for ``device_module.graph(...)`` — a no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_backend(runner):
    """Build a ``FullCudaGraphBackend`` without running ``__init__`` (which would
    touch CUDA), wiring just the attributes ``capture_one`` reads."""
    backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._precarve = SimpleNamespace(
        measure=contextlib.nullcontext, mint=mock.Mock()
    )
    backend._reuse_output_buffer = False
    backend._output_buffer = None
    backend._memory_saver_adapter = None
    backend._cuda_graph_runner = runner
    backend._device_module = runner.device_module
    backend._tp_group = runner.model_runner.tp_group
    return backend


def _make_runner(*, enable_profile, profiler, num_tokens_per_bs=1, mode_name="DECODE"):
    device_module = SimpleNamespace(
        synchronize=mock.Mock(name="synchronize"),
        graph=mock.Mock(name="graph", side_effect=lambda **kw: _FakeGraphCtx()),
    )
    tp_group = SimpleNamespace(barrier=mock.Mock(name="barrier"))
    runner = SimpleNamespace(
        device_module=device_module,
        model_runner=SimpleNamespace(tp_group=tp_group),
        num_tokens_per_bs=num_tokens_per_bs,
        capture_forward_mode=SimpleNamespace(name=mode_name),
        enable_profile_cuda_graph=enable_profile,
    )
    if profiler is not _UNSET:
        runner._profiler = profiler
    return runner


class TestCaptureOneNoProfiling(CustomTestCase):
    def test_dedup_registers_raw_graph_and_releases_registry_on_cleanup(self):
        backend = _make_backend(_make_runner(enable_profile=False, profiler=None))
        registry = mock.Mock()
        backend.deduped_cuda_graph = registry
        backend._deduped_cuda_graph_registries = [registry]
        shape_key = ShapeKey(size=4)
        with mock.patch("torch.cuda.CUDAGraph", return_value="RAW_GRAPH") as create:
            backend.capture_one(shape_key, lambda: torch.ones(4))
        create.assert_called_once_with(keep_graph=True)
        registry.register.assert_called_once_with("RAW_GRAPH")
        self.assertIs(backend._graphs[shape_key], registry.register.return_value)
        backend.replay(shape_key, None)
        registry.register.return_value.replay.assert_called_once_with()
        backend.cleanup()
        registry.close.assert_called_once_with()
        self.assertFalse(backend._graphs)
        self.assertFalse(backend._capture_inputs)
        self.assertIsNone(backend.deduped_cuda_graph)

    def test_capture_session_seals_dedup_after_capture_failure(self):
        backend = _make_backend(_make_runner(enable_profile=False, profiler=None))
        backend._pool = (0, 1)
        module = "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend"
        with (
            mock.patch(module + ".set_graph_pool_id"),
            mock.patch.object(backend, "begin_cuda_graph_capture") as begin,
            mock.patch.object(backend, "end_cuda_graph_capture") as end,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with backend.capture_session("STREAM"):
                    self.assertEqual(backend._capture_stream, "STREAM")
                    raise RuntimeError("capture failed")
        begin.assert_called_once_with()
        end.assert_called_once_with()
        self.assertIsNone(backend._capture_stream)

    def test_nested_draft_outputs_share_storage_across_shapes(self):
        backend = _make_backend(_make_runner(enable_profile=False, profiler=None))
        backend._reuse_output_buffer = True

        def output(rows, value):
            return (
                torch.full((rows, 2), value),
                [torch.full((rows, 3), value + 1), None],
            )

        with mock.patch("torch.cuda.CUDAGraph", side_effect=["GRAPH4", "GRAPH2"]):
            backend.capture_one(ShapeKey(size=4), lambda: output(4, 1.0))
            backend.capture_one(ShapeKey(size=2), lambda: output(2, 7.0))
        large = backend._outputs[ShapeKey(size=4)]
        small = backend._outputs[ShapeKey(size=2)]
        self.assertIsInstance(small, tuple)
        self.assertIsInstance(small[1], list)
        self.assertIsNone(small[1][1])
        for a, b in [(large[0], small[0]), (large[1][0], small[1][0])]:
            self.assertEqual(a.data_ptr(), b.data_ptr())
            self.assertEqual(b.shape[0], 2)
            self.assertTrue(torch.equal(a[:2], b))
        self.assertTrue(torch.equal(small[0], torch.full((2, 2), 7.0)))
        self.assertTrue(torch.equal(small[1][0], torch.full((2, 3), 8.0)))
        self.assertTrue(backend._reuse_output_buffer)

    def test_nested_structure_mismatch_does_not_partially_copy(self):
        buffer = (torch.zeros(4, 2), [torch.zeros(4, 3)])
        output = (torch.ones(2, 2), [torch.ones(2, 3), torch.ones(2, 3)])
        self.assertIsNone(_copy_output_to_buffer(output, buffer))
        self.assertEqual(buffer[0].count_nonzero(), 0)
        self.assertEqual(buffer[1][0].count_nonzero(), 0)

    def test_capture_inputs_remain_owned_until_cleanup(self):
        class Inputs:
            pass

        backend = _make_backend(_make_runner(enable_profile=False, profiler=None))
        inputs = Inputs()
        reference = weakref.ref(inputs)
        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(
                ShapeKey(size=4), lambda: torch.ones(4), capture_inputs=inputs
            )
        del inputs
        gc.collect()
        self.assertIsNotNone(reference())
        backend.cleanup()
        gc.collect()
        self.assertIsNone(reference())

    def test_runs_two_warmups_and_capture_without_stepping(self):
        runner = _make_runner(enable_profile=False, profiler=None)
        backend = _make_backend(runner)

        sentinel_out = object()
        forward_fn = mock.Mock(return_value=sentinel_out)
        post_warmup_hook = mock.Mock()
        shape_key = ShapeKey(size=4)

        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(
                shape_key, forward_fn, post_warmup_hook=post_warmup_hook
            )

        # 2 warmups + 1 capture.
        self.assertEqual(forward_fn.call_count, 3)
        # post_warmup_hook only runs in the two warmup iterations.
        self.assertEqual(post_warmup_hook.call_count, 2)
        # Graph + output are recorded against the shape key.
        self.assertEqual(backend._graphs[shape_key], "GRAPH")
        self.assertIs(backend._outputs[shape_key], sentinel_out)

    def test_prefill_shapes_share_one_output_storage(self):
        runner = _make_runner(enable_profile=False, profiler=None, mode_name="EXTEND")
        backend = _make_backend(runner)
        backend._reuse_output_buffer = True

        outputs = iter(
            [
                torch.ones((4, 2)),
                torch.ones((4, 2)),
                torch.ones((4, 2)),
                torch.ones((2, 2)),
                torch.ones((2, 2)),
                torch.ones((2, 2)),
            ]
        )
        with mock.patch("torch.cuda.CUDAGraph", side_effect=["GRAPH4", "GRAPH2"]):
            backend.capture_one(ShapeKey(size=4), lambda: next(outputs))
            backend.capture_one(ShapeKey(size=2), lambda: next(outputs))

        large = backend._outputs[ShapeKey(size=4)]
        small = backend._outputs[ShapeKey(size=2)]
        self.assertEqual(large.shape, (4, 2))
        self.assertEqual(small.shape, (2, 2))
        self.assertEqual(large.data_ptr(), small.data_ptr())
        self.assertEqual(backend._output_buffer.shape, (4, 2))

    def test_enable_flag_set_but_no_profiler_attr_does_not_step(self):
        # The runner advertises the flag but never created a profiler; the
        # getattr guard must keep capture_one on the non-profiling path.
        runner = _make_runner(enable_profile=True, profiler=_UNSET)
        backend = _make_backend(runner)
        self.assertFalse(hasattr(runner, "_profiler"))

        forward_fn = mock.Mock(return_value=object())
        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(ShapeKey(size=2), forward_fn)

        self.assertEqual(forward_fn.call_count, 3)


class TestCaptureOneWithProfiling(CustomTestCase):
    def _run(self, *, size, num_tokens_per_bs, mode_name):
        profiler = SimpleNamespace(step=mock.Mock(name="step"))
        runner = _make_runner(
            enable_profile=True,
            profiler=profiler,
            num_tokens_per_bs=num_tokens_per_bs,
            mode_name=mode_name,
        )
        backend = _make_backend(runner)

        forward_fn = mock.Mock(return_value=object())
        rf_names = []

        def _fake_record_function(name):
            rf_names.append(name)
            return contextlib.nullcontext()

        with (
            mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"),
            mock.patch(
                "torch.profiler.record_function", side_effect=_fake_record_function
            ),
        ):
            backend.capture_one(ShapeKey(size=size), forward_fn)

        return profiler, forward_fn, rf_names

    def test_steps_twice_in_warmup_and_once_after_capture(self):
        profiler, forward_fn, _ = self._run(
            size=4, num_tokens_per_bs=1, mode_name="DECODE"
        )
        # Schedule wait=2 + active=1 => one step per warmup (x2) + one post-capture.
        self.assertEqual(profiler.step.call_count, 3)
        self.assertEqual(forward_fn.call_count, 3)

    def test_capture_not_wrapped_in_record_function(self):
        # The capture forward is no longer wrapped in a record_function; per-bs
        # trace naming is handled by the profiler's on_trace_ready callback.
        _, _, rf_names = self._run(size=4, num_tokens_per_bs=1, mode_name="DECODE")
        self.assertEqual(rf_names, [])


class TestNativeProjectionReplay(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.backend = _make_backend(_make_runner(enable_profile=False, profiler=None))
        self.key = ShapeKey(size=128)
        self.backend._outputs[self.key] = "OUTPUT"
        self.batch = SimpleNamespace(extend_seq_lens_cpu=[40, 64])
        self.native = "sglang.srt.layers.attention.dsa.kpool_native_projection_graph"
        patch = mock.patch(
            "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.get_bool_env_var",
            side_effect=lambda name: name == "SGLANG_KPOOL_PREFILL_NATIVE_PROJECTION_GRAPHS",
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_dedup_replay_uses_live_lengths(self):
        from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import DedupedCudaGraph

        graph = mock.Mock(spec=DedupedCudaGraph)
        graph.raw_graph = 170
        self.backend._graphs[self.key] = graph
        with mock.patch(self.native + ".replay_callback") as callback:
            self.assertEqual(self.backend.replay(self.key, self.batch), "OUTPUT")
        callback.assert_called_once_with(170, 104)
        graph.replay.assert_called_once_with(before_launch=callback.return_value)

    def test_non_dedup_updates_before_launch(self):
        graph = mock.Mock()
        graph.raw_cuda_graph.return_value = 170
        graph.raw_cuda_graph_exec.return_value = 990
        self.backend._graphs[self.key] = graph
        order = mock.Mock()
        order.attach_mock(graph.replay, "launch")
        with mock.patch(self.native + ".replay_callback") as callback:
            order.attach_mock(callback.return_value, "update")
            self.backend.replay(self.key, self.batch)
        callback.assert_called_once_with(170, 104)
        self.assertEqual(order.mock_calls, [mock.call.update(990, 170), mock.call.launch()])

    def test_decode_without_native_children_replays_normally(self):
        graph = mock.Mock()
        graph.raw_cuda_graph.return_value = 170
        self.backend._graphs[self.key] = graph
        with mock.patch(self.native + ".replay_callback", return_value=None) as callback:
            self.backend.replay(self.key, SimpleNamespace(extend_seq_lens_cpu=None))
        callback.assert_called_once_with(170, 0)
        graph.raw_cuda_graph_exec.assert_not_called()
        graph.replay.assert_called_once_with()

    def test_failed_update_prevents_launch(self):
        graph = mock.Mock()
        self.backend._graphs[self.key] = graph
        with mock.patch(self.native + ".replay_callback") as callback:
            callback.return_value.side_effect = ValueError("wrong owner")
            with self.assertRaisesRegex(ValueError, "wrong owner"):
                self.backend.replay(self.key, self.batch)
        graph.replay.assert_not_called()

    def test_non_dedup_capture_instantiates_retained_graph(self):
        graph = mock.Mock()
        with mock.patch("torch.cuda.CUDAGraph", return_value=graph) as create:
            self.backend.capture_one(self.key, lambda: torch.ones(128))
        create.assert_called_once_with(keep_graph=True)
        graph.instantiate.assert_called_once_with()

    def test_cleanup_releases_native_templates_after_executables(self):
        from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import DedupedCudaGraph

        graph = mock.Mock(spec=DedupedCudaGraph)
        graph.raw_graph = 170
        registry = mock.Mock()
        self.backend.deduped_cuda_graph = registry
        self.backend._deduped_cuda_graph_registries = [registry]
        self.backend._graphs[self.key] = graph
        order = mock.Mock()
        order.attach_mock(registry.close, "close")
        with mock.patch(self.native + ".release_graphs") as release:
            order.attach_mock(release, "release")
            self.backend.cleanup()
        self.assertEqual(order.mock_calls, [mock.call.close(), mock.call.release([170])])
        self.assertFalse(self.backend._graphs)
        self.assertFalse(self.backend._outputs)


if __name__ == "__main__":
    unittest.main()
