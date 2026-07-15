import unittest

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)


class _Slot:
    def __init__(self):
        self.calls = []
        self.value = object()

    def slice_for(self, batch_size, num_tokens):
        self.calls.append((batch_size, num_tokens))
        return self.value


class _Registry:
    def __init__(self, with_input_embeds):
        self.slot = _Slot() if with_input_embeds else None

    def has_slot(self, name):
        return name == "input_embeds" and self.slot is not None

    def get_slot(self, name):
        assert name == "input_embeds"
        return self.slot


class TestPrefillCudaGraphMultimodalEmbeddings(unittest.TestCase):
    def test_multimodal_capture_uses_stable_input_embeds_slot(self):
        registry = _Registry(with_input_embeds=True)

        result = PrefillCudaGraphRunner._graph_input_embeds(registry, 4, 1536)

        self.assertIs(result, registry.slot.value)
        self.assertEqual(registry.slot.calls, [(4, 1536)])

    def test_text_only_capture_has_no_input_embeds(self):
        registry = _Registry(with_input_embeds=False)

        result = PrefillCudaGraphRunner._graph_input_embeds(registry, 1, 128)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
