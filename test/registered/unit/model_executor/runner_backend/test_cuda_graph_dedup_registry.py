import unittest
from unittest import mock

from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraph,
    DedupedCudaGraphRegistry,
    GraphExecGroup,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDedupCompatibility(unittest.TestCase):
    def test_before_launch_uses_executable_owner_after_update_and_on_repeat(self):
        registry = DedupedCudaGraphRegistry(allow_kernel_updates=True)
        original = DedupedCudaGraph(1, None, registry)
        current = DedupedCudaGraph(2, None, registry)
        group = GraphExecGroup(10, 1, None, [original, current])
        original.group = current.group = group
        events = []

        def update(executable, raw):
            events.append(("update", executable, raw))
            return True, ""

        def launch(executable, stream):
            events.append(("launch", executable, stream))
            return (0,)

        def before_launch(executable, original_raw):
            events.append(("hook", executable, original_raw))

        module = "sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin"
        with (
            mock.patch(module + ".dedup_update", side_effect=update),
            mock.patch(module + ".cuda_rt") as runtime,
            mock.patch(module + ".checkCudaErrors"),
        ):
            runtime.cudaGraphLaunch.side_effect = launch
            current.replay(7, before_launch=before_launch)
            current.replay(7, before_launch=before_launch)
            current.replay(7)
        self.assertEqual(
            events,
            [
                ("update", 10, 2),
                ("hook", 10, 1),
                ("launch", 10, 7),
                ("hook", 10, 1),
                ("launch", 10, 7),
                ("launch", 10, 7),
            ],
        )

    def test_before_launch_failure_prevents_launch(self):
        registry = DedupedCudaGraphRegistry()
        graph = DedupedCudaGraph(1, None, registry)
        graph.group = GraphExecGroup(10, 1, None, [graph])
        callback = mock.Mock(side_effect=RuntimeError("child update failed"))
        module = "sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin"
        with mock.patch(module + ".cuda_rt") as runtime:
            with self.assertRaisesRegex(RuntimeError, "child update failed"):
                graph.replay(7, before_launch=callback)
            runtime.cudaGraphLaunch.assert_not_called()
        callback.assert_called_once_with(10, 1)

    def test_failed_graph_update_prevents_hook_and_launch(self):
        registry = DedupedCudaGraphRegistry(allow_kernel_updates=True)
        original = DedupedCudaGraph(1, None, registry)
        current = DedupedCudaGraph(2, None, registry)
        group = GraphExecGroup(10, 1, None, [original, current])
        original.group = current.group = group
        callback = mock.Mock()
        module = "sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin"
        with (
            mock.patch(module + ".dedup_update", return_value=(False, "incompatible")),
            mock.patch(module + ".cuda_rt") as runtime,
        ):
            with self.assertRaisesRegex(AssertionError, "incompatible"):
                current.replay(7, before_launch=callback)
            callback.assert_not_called()
            runtime.cudaGraphLaunch.assert_not_called()
        self.assertEqual(group.current_raw_graph, 1)

    def test_incompatible_update_keeps_separate_executable(self):
        registry = DedupedCudaGraphRegistry(allow_kernel_updates=True)
        captures = [mock.Mock() for _ in range(3)]
        for index, graph in enumerate(captures):
            graph.raw_cuda_graph.return_value = index + 1
        module = "sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin"
        with (
            mock.patch(module + ".graph_signature", return_value=("same-topology",)),
            mock.patch(module + ".dedup_update", side_effect=[(False, "incompatible"), (True, "")]),
            mock.patch.object(registry, "instantiate", side_effect=[10, 11, 20, 21]),
            mock.patch.object(registry, "destroy_exec") as destroy,
        ):
            graphs = [registry.register(graph) for graph in captures]
            self.assertIsNot(graphs[0].group, graphs[1].group)
            self.assertIs(graphs[0].group, graphs[2].group)
            self.assertEqual(registry.stats(), (3, 2))
            registry.seal()
            self.assertEqual(destroy.call_count, 2)
            registry.close()
            self.assertEqual(destroy.call_count, 4)
        for capture in captures:
            capture.reset.assert_called_once_with()

    def test_strict_signature_path_still_rejects_failed_update(self):
        registry = DedupedCudaGraphRegistry()
        capture = mock.Mock()
        capture.raw_cuda_graph.return_value = 1
        module = "sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin"
        with (
            mock.patch(module + ".graph_signature", return_value=("same-signature",)),
            mock.patch(module + ".dedup_update", return_value=(False, "incompatible")),
            mock.patch.object(registry, "instantiate", side_effect=[10, 11]),
        ):
            registry.register(capture)
            with self.assertRaisesRegex(AssertionError, "incompatible"):
                registry.register(capture)
