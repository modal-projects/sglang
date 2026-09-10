import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._sparse_prefill_full_graph = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
        )

    def test_sparse_full_requires_a_matching_pool_plan(self):
        runner = self._make_runner()
        runner._is_full_backend = True
        runner._sparse_prefill_full_graph = True
        runner._capture_req_slots = 32
        runner.kpool_prefill_graph_variants = mock.Mock()
        runner.kpool_prefill_graph_variants.resolve.return_value = None
        batch = self._make_forward_batch(8)
        self.assertFalse(runner.can_run_graph(batch))
        runner.kpool_prefill_graph_variants.resolve.return_value = (16, 2048, 4096)
        self.assertTrue(runner.can_run_graph(batch))
        batch.batch_size = 33
        self.assertFalse(runner.can_run_graph(batch))

    def test_sparse_full_captures_only_pool_variants_largest_first(self):
        runner = self._make_runner()
        runner._is_full_backend = True
        runner._sparse_prefill_full_graph = True
        runner.device = "cpu"
        runner.model_runner = SimpleNamespace(device="cuda", gpu_id=0)
        runner.device_module = SimpleNamespace(synchronize=mock.Mock())
        variants = [(128, 2048, 4096), (256, 2048, 4096), (4096, 32768, 65536)]
        runner.kpool_prefill_graph_variants = SimpleNamespace(
            variants=variants,
            bucket=lambda v: v[0],
            dsa_variant=lambda v: None,
            label=lambda v: str(v),
        )
        runner.dsa_prefill_graph_variants = SimpleNamespace(label=lambda v: None)
        runner.capture_one_shape = mock.Mock()
        module = "sglang.srt.model_executor.runner.prefill_cuda_graph_runner"
        with (
            mock.patch(module + ".get_available_gpu_memory", return_value=100),
            mock.patch(
                module + ".get_parallel", return_value=SimpleNamespace(tp_rank=0)
            ),
            mock.patch(
                "torch.cuda.mem_get_info", return_value=(100 * 1024**3, 100 * 1024**3)
            ),
            mock.patch("torch.distributed.get_rank", return_value=0),
        ):
            runner._capture_one_stream()
        self.assertEqual(
            runner.capture_one_shape.call_args_list,
            [
                mock.call(v[0], dsa_variant=None, kpool_variant=v)
                for v in reversed(variants)
            ],
        )

    def test_sparse_full_metadata_keeps_real_request_count(self):
        runner = self._make_runner()
        runner._is_full_backend = True
        runner._sparse_prefill_full_graph = True
        runner.use_captured_attn_metadata = False
        backend = mock.Mock()
        runner.model_runner = SimpleNamespace(attn_backend=backend)
        batch = self._make_forward_batch(8)
        batch.batch_size = 2
        batch.extend_seq_lens_cpu = [3, 5]
        runner._prepare_forward_metadata_for_replay(
            batch, self._make_forward_batch(16), 16
        )
        backend.init_forward_metadata.assert_called_once_with(batch)
        backend.init_forward_metadata_out_graph.assert_not_called()
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.extend_seq_lens_cpu, [3, 5])
        backend.prepare_prefill_shared_read_snapshot.assert_called_once_with(
            batch, num_qo_tokens=16
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_replay_snapshot_uses_padded_token_count(self):
        runner = self._make_runner()
        runner.use_captured_attn_metadata = False
        attn_backend = mock.Mock()
        runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
        forward_batch = self._make_forward_batch(8)
        static_forward_batch = self._make_forward_batch(16)

        runner._prepare_forward_metadata_for_replay(
            forward_batch,
            static_forward_batch,
            num_tokens=16,
        )

        attn_backend.init_forward_metadata.assert_called_once_with(forward_batch)
        attn_backend.prepare_prefill_shared_read_snapshot.assert_called_once_with(
            forward_batch, num_qo_tokens=16
        )


if __name__ == "__main__":
    unittest.main()
