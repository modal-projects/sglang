import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.image_prefill_attn_backend import (
    ImagePrefillAttnBackend,
    is_image_prefill,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Backend(AttentionBackend):
    def __init__(self, name):
        self.name = name
        self.forward_metadata = SimpleNamespace(custom_mask=None, mask_indptr=None)
        self.initialized = []

    def init_forward_metadata(self, forward_batch):
        self.initialized.append(forward_batch)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        return self.name

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        return self.name


class _MaskBackend(_Backend):
    supports_custom_mask = True

    def install_custom_mask(self, *, custom_mask, mask_indptr):
        self.forward_metadata.custom_mask = custom_mask
        self.forward_metadata.mask_indptr = mask_indptr


def _batch(mode, *, images):
    mm_inputs = [SimpleNamespace(contains_image_inputs=lambda: images)]
    return SimpleNamespace(
        forward_mode=mode,
        mm_inputs=mm_inputs,
        contains_image_inputs=lambda: images,
    )


def _wrapper():
    runner = SimpleNamespace(
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        kv_index_translator=None,
        model_config=SimpleNamespace(context_len=4096),
    )
    text, image = _Backend("text"), _MaskBackend("image")
    return (
        ImagePrefillAttnBackend(runner, text_backend=text, image_backend=image),
        text,
        image,
    )


class TestImagePrefillAttnBackend(CustomTestCase):
    def test_image_extend_batch_runs_on_the_mask_backend(self):
        wrapper, text, image = _wrapper()
        batch = _batch(ForwardMode.EXTEND, images=True)

        wrapper.init_forward_metadata(batch)

        self.assertEqual(image.initialized, [batch])
        self.assertEqual(text.initialized, [])
        self.assertTrue(wrapper.supports_custom_mask)
        self.assertIs(wrapper.forward_metadata, image.forward_metadata)
        wrapper.install_custom_mask(custom_mask="mask", mask_indptr="indptr")
        self.assertEqual(image.forward_metadata.custom_mask, "mask")
        self.assertEqual(image.forward_metadata.mask_indptr, "indptr")
        self.assertEqual(wrapper.forward_extend(None, None, None, None, batch), "image")

    def test_every_other_forward_stays_on_the_text_backend(self):
        wrapper, text, image = _wrapper()
        for batch in (
            _batch(ForwardMode.EXTEND, images=False),
            _batch(ForwardMode.MIXED, images=True),
            _batch(ForwardMode.DECODE, images=True),
            _batch(ForwardMode.TARGET_VERIFY, images=True),
        ):
            wrapper.init_forward_metadata(batch)
            self.assertFalse(wrapper.supports_custom_mask, batch.forward_mode)
            self.assertIs(wrapper.forward_metadata, text.forward_metadata)
            self.assertEqual(
                wrapper.forward(None, None, None, None, batch),
                "text",
                batch.forward_mode,
            )
        self.assertEqual(len(text.initialized), 4)
        self.assertEqual(image.initialized, [])

    def test_forward_keeps_the_backend_chosen_at_metadata_init(self):
        """The embed routine drops forward_batch.mm_inputs after embedding the images; forwards after that must still run on the backend whose metadata was initialized."""
        wrapper, text, image = _wrapper()
        batch = _batch(ForwardMode.EXTEND, images=True)
        wrapper.init_forward_metadata(batch)
        batch.mm_inputs = None
        batch.contains_image_inputs = lambda: False

        self.assertEqual(wrapper.forward(None, None, None, None, batch), "image")
        self.assertEqual(wrapper.forward_extend(None, None, None, None, batch), "image")
        self.assertIs(wrapper.forward_metadata, image.forward_metadata)

    def test_routing_predicate_matches_the_mask_install_condition(self):
        self.assertTrue(is_image_prefill(_batch(ForwardMode.EXTEND, images=True)))
        self.assertFalse(is_image_prefill(_batch(ForwardMode.EXTEND, images=False)))
        self.assertFalse(is_image_prefill(_batch(ForwardMode.MIXED, images=True)))

    def test_triton_declares_custom_mask_support(self):
        self.assertTrue(TritonAttnBackend.supports_custom_mask)
        self.assertFalse(AttentionBackend.supports_custom_mask)


if __name__ == "__main__":
    unittest.main()
