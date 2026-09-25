"""Model padding publishes the same media ownership consumed by prefix caching."""

import unittest
from array import array
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternTokenPairs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.models.llava import LlavaBaseForCausalLM
from sglang.srt.models.llavavid import LlavaVidForCausalLM
from sglang.srt.models.minicpmv import (
    MiniCPMV2_6,
    MiniCPMV4_0,
    MiniCPMV4_5,
    MiniCPMV4_6,
)
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.models.moss_vl import MossVLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _media(value, *, offsets=None, feature=None, **metadata):
    if feature is None:
        feature = torch.full((1, 3, 2, 2), float(value))
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=feature,
        offsets=offsets,
        model_specific_data=metadata,
    )
    item.set_pad_value()
    return item


def _llava_model(feature_len=3, aspect="pad"):
    return SimpleNamespace(
        image_feature_len=feature_len,
        image_size=4,
        patch_size=2,
        num_patches_per_side=2,
        config=SimpleNamespace(image_token_index=99, image_aspect_ratio=aspect),
        vision_tower=SimpleNamespace(config=SimpleNamespace(image_size=4)),
        image_grid_pinpoints=[(4, 4)],
        _infer_image_aspect_ratio=LlavaBaseForCausalLM._infer_image_aspect_ratio,
    )


def _pairs(items):
    return MultimodalInputs(
        mm_items=items,
        im_start_id=100,
        im_end_id=102,
        slice_start_id=103,
        slice_end_id=104,
    )


def _spans(inputs, tokens):
    return inputs.cache_spans(array("q", tokens))


class TestMultimodalPaddingSpans(CustomTestCase):
    def test_llava_repeated_regions_keep_their_item_and_feature_offset(self):
        first = _media(
            1,
            feature=torch.ones(2, 3, 2, 2),
            image_sizes=[(2, 2), (2, 2)],
            image_aspect_ratio="pad",
        )
        second = _media(2, image_sizes=[(2, 2)], image_aspect_ratio="pad")
        inputs = MultimodalInputs(mm_items=[first, second])
        tokens = LlavaBaseForCausalLM.pad_input_ids(
            _llava_model(), array("q", [10, 99, 20, 99, 30, 99, 40]), inputs
        )
        self.assertEqual(first.offsets, [(1, 3), (5, 7)])
        self.assertEqual(second.offsets, [(9, 11)])
        self.assertEqual(list(tokens[1:4]), [first.pad_value] * 3)
        self.assertEqual(list(tokens[5:8]), [first.pad_value] * 3)
        self.assertEqual(list(tokens[9:12]), [second.pad_value] * 3)
        self.assertEqual(
            [(span.start, span.end, span.offset) for span in _spans(inputs, tokens)],
            [(1, 4, 0), (5, 8, 3), (9, 12, 0)],
        )

    def test_llava_repadding_publishes_absolute_session_positions(self):
        item = _media(1, image_sizes=[(2, 2)], image_aspect_ratio="pad")
        inputs = MultimodalInputs(mm_items=[item])
        for prefix in ([10, 11], [10, 11, 12, 13, 14]):
            with self.subTest(prefix_length=len(prefix)):
                tokens = LlavaBaseForCausalLM.pad_input_ids(
                    _llava_model(), array("q", prefix + [99, 20]), inputs
                )
                self.assertEqual(list(tokens[: len(prefix)]), prefix)
                self.assertEqual(item.offsets, [(len(prefix), len(prefix) + 2)])
                self.assertEqual(_spans(inputs, tokens)[0].start, len(prefix))

    def test_llava_anyres_span_matches_expanded_length(self):
        item = _media(1, image_sizes=[(2, 2)], image_aspect_ratio="anyres")
        inputs = MultimodalInputs(mm_items=[item])
        tokens = LlavaBaseForCausalLM.pad_input_ids(
            _llava_model(feature_len=4, aspect="anyres"),
            array("q", [10, 99, 20]),
            inputs,
        )
        self.assertEqual(inputs.image_pad_len, [10])
        self.assertEqual(item.offsets, [(1, 10)])
        self.assertEqual(list(tokens[1:11]), [item.pad_value] * 10)
        self.assertEqual(
            (_spans(inputs, tokens)[0].start, _spans(inputs, tokens)[0].end), (1, 11)
        )

    def test_llava_different_later_image_keeps_the_earlier_prefix(self):
        keys = []
        for later in (2, 3):
            inputs = MultimodalInputs(
                mm_items=[
                    _media(1, image_sizes=[(2, 2)], image_aspect_ratio="pad"),
                    _media(later, image_sizes=[(2, 2)], image_aspect_ratio="pad"),
                ]
            )
            tokens = LlavaBaseForCausalLM.pad_input_ids(
                _llava_model(), array("q", [10, 99, 20, 99, 30]), inputs
            )
            keys.append(RadixKey(tokens, mm_spans=_spans(inputs, tokens)))
        self.assertEqual(keys[0].match(keys[1]), 5)

    def test_minicpm_missing_offsets_are_published_for_every_slice(self):
        raw = [10, 100, 101, 101, 102, 103, 101, 101, 104, 20, 100, 101, 101, 102, 30]
        for model in (MiniCPMV2_6, MiniCPMV4_0, MiniCPMV4_5, MiniCPMV4_6):
            with self.subTest(model=model.__name__):
                first, second = _media(1), _media(2)
                inputs = _pairs([first, second])
                tokens = model.pad_input_ids(None, raw, inputs)
                self.assertEqual(first.offsets, [(2, 3), (6, 7)])
                self.assertEqual(second.offsets, [(11, 12)])
                self.assertEqual(
                    [(s.start, s.end, s.offset) for s in _spans(inputs, tokens)],
                    [(2, 4, 0), (6, 8, 2), (11, 13, 0)],
                )

    def test_minicpm_patch_offsets_choose_each_patch_padding(self):
        raw = [10, 100, 101, 101, 102, 103, 101, 101, 104, 20, 100, 101, 101, 102, 30]
        items = [
            _media(1, offsets=[(2, 3)]),
            _media(2, offsets=[(6, 7)]),
            _media(3, offsets=[(11, 12)]),
        ]
        inputs = _pairs(items)
        tokens = MiniCPMV4_6.pad_input_ids(None, raw, inputs)
        for item in items:
            start, end = item.offsets[0]
            self.assertEqual(tokens[start : end + 1], [item.pad_value] * 2)
        self.assertEqual(inputs.data_offsets, [1, 10])
        self.assertEqual([span.offset for span in _spans(inputs, tokens)], [0, 0, 0])

    def test_minicpm_session_padding_does_not_rewrite_historical_media(self):
        old = _media(1, offsets=[(2, 3)])
        raw = [
            10,
            100,
            old.pad_value,
            old.pad_value,
            102,
            20,
            21,
            100,
            101,
            101,
            102,
            30,
        ]
        for model in (MiniCPMV2_6, MiniCPMV4_6):
            with self.subTest(model=model.__name__):
                new = _media(2, offsets=[(8, 9)])
                inputs = _pairs([new])
                tokens = model.pad_input_ids(None, raw, inputs)
                self.assertEqual(tokens[:7], raw[:7])
                self.assertEqual(tokens[8:10], [new.pad_value] * 2)
                self.assertEqual(inputs.data_offsets, [7])
                combined = _pairs([old, new])
                self.assertEqual(
                    [(s.start, s.end) for s in _spans(combined, tokens)],
                    [(2, 4), (8, 10)],
                )

    def test_minicpm_shared_processor_ranges_fall_back_to_pair_ownership(self):
        """Legacy precomputed inputs repeat the complete image range list per item."""
        items = [_media(value, offsets=[(2, 3), (7, 8)]) for value in (1, 2)]
        inputs = _pairs(items)
        raw = [10, 100, 101, 101, 102, 20, 100, 101, 101, 102, 30]
        tokens = MiniCPMV2_6.pad_input_ids(None, raw, inputs)
        self.assertEqual(tokens[2:4], [items[0].pad_value] * 2)
        self.assertEqual(tokens[7:9], [items[1].pad_value] * 2)
        self.assertEqual([item.offsets for item in items], [[(2, 3)], [(7, 8)]])
        self.assertEqual(
            [(s.start, s.end, s.offset) for s in _spans(inputs, tokens)],
            [(2, 4, 0), (7, 9, 0)],
        )

    def test_token_pairs_leave_missing_or_unbalanced_markers_unchanged(self):
        helper = MultiModalityDataPaddingPatternTokenPairs([(100, 102)])
        for raw in ([10, 20], [10, 100, 101, 20]):
            with self.subTest(raw=raw):
                inputs = _pairs([_media(1)])
                self.assertEqual(helper.pad_input_tokens(raw, inputs), raw)
                self.assertIsNone(inputs.mm_items[0].offsets)

    def test_video_declares_one_aggregate_span_without_changing_item_offsets(self):
        keys = []
        for later in (2, 3):
            inputs = MultimodalInputs(mm_items=[_media(1), _media(later)])
            model = SimpleNamespace(
                image_feature_len=6, config=SimpleNamespace(image_token_index=99)
            )
            tokens = LlavaVidForCausalLM.pad_input_ids(
                model, array("q", [10, 11, 99, 20]), inputs
            )
            spans = _spans(inputs, tokens)
            self.assertEqual([(s.start, s.end, s.offset) for s in spans], [(2, 8, 0)])
            self.assertTrue(all(item.offsets is None for item in inputs.mm_items))
            self.assertRegex(spans[0].identity, r"^sha256:[0-9a-f]{64}$")
            keys.append(RadixKey(tokens, mm_spans=spans))
        self.assertEqual(keys[0].match(keys[1]), 2)

    def test_encoder_prefixes_cover_the_emitted_mllama_and_moss_regions(self):
        for media_count in (1, 2):
            with self.subTest(model="mllama", media_count=media_count):
                feature = torch.ones(1, media_count, 2, 3, 2, 2)
                inputs = MultimodalInputs(mm_items=[_media(1, feature=feature)])
                model = SimpleNamespace(vision_model=SimpleNamespace(num_patches=3))
                tokens = MllamaForConditionalGeneration.pad_input_ids(
                    model, array("q", [10, 20]), inputs
                )
                expected = media_count * 2 * 3
                self.assertEqual(inputs.num_image_tokens, expected)
                self.assertEqual(len(tokens), expected + 2)
                self.assertEqual(
                    [(s.start, s.end) for s in _spans(inputs, tokens)], [(0, expected)]
                )
        for multiple, expected in ((1, 10), (4, 12)):
            with self.subTest(model="moss", pad_multiple=multiple):
                inputs = MultimodalInputs(
                    mm_items=[_media(1, grid_thw=torch.tensor([[2, 4, 4]]))]
                )
                model = SimpleNamespace(
                    spatial_merge_size=2, vision_seq_pad_multiple=multiple
                )
                model._get_encoder_len = MethodType(
                    MossVLForConditionalGeneration._get_encoder_len, model
                )
                model._build_encoder_prefix_pad_ids = MethodType(
                    MossVLForConditionalGeneration._build_encoder_prefix_pad_ids, model
                )
                tokens = MossVLForConditionalGeneration.pad_input_ids(
                    model, array("q", [10, 20]), inputs
                )
                self.assertEqual(inputs.num_image_tokens, expected)
                self.assertEqual(len(tokens), expected + 2)
                self.assertEqual(
                    [(s.start, s.end) for s in _spans(inputs, tokens)], [(0, expected)]
                )


if __name__ == "__main__":
    unittest.main()
