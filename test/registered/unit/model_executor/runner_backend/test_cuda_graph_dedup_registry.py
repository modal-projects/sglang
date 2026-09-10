import unittest
from unittest import mock

from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDedupCompatibility(unittest.TestCase):
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
