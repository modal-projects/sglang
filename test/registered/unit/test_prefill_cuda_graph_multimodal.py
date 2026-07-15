import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)


class TestPrefillCudaGraphMultimodalEmbeddings(unittest.TestCase):
    def _runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.capture_hidden_mode = object()
        runner.max_num_tokens = 4096
        runner.bcg_capture_bs_of = None
        runner._has_inactive_dp_rank = lambda _forward_batch: False
        return runner

    def _batch(self, mm_inputs):
        mode = SimpleNamespace(is_target_verify=lambda: False)
        hidden_mode = object()
        batch = SimpleNamespace(
            batch_size=1,
            input_ids=[0] * 128,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=mm_inputs,
            forward_mode=mode,
            capture_hidden_mode=hidden_mode,
            global_num_tokens_cpu=None,
            return_logprob=False,
        )
        return batch

    def test_multimodal_prefill_falls_back_to_eager(self):
        runner = self._runner()
        batch = self._batch([object()])
        batch.capture_hidden_mode = runner.capture_hidden_mode

        self.assertFalse(runner.can_run_graph(batch))

    def test_text_only_prefill_remains_graph_eligible(self):
        runner = self._runner()
        batch = self._batch([None])
        batch.capture_hidden_mode = runner.capture_hidden_mode

        self.assertTrue(runner.can_run_graph(batch))


if __name__ == "__main__":
    unittest.main()
