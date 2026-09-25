"""Media identities retain content, layout, and namespace across cache and wire boundaries."""

import unittest
from array import array
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.managers.mm_schedule import _get_multimodal_mask
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
    Req,
    _compute_pad_value,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.multimodal.cache.identity import (
    build_artifact_key,
    multimodal_hash_as_int,
    parse_content_hash,
    resolve_multimodal_item_hash,
)
from sglang.srt.multimodal.encoder_preprocessing import hash_raw_encoder_item
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.session.session_controller import SessionController
from sglang.srt.utils.msgpack_utils import enc_hook, ext_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestContentIdentity(unittest.TestCase):
    def test_content_grid_and_namespace_each_affect_the_complete_identity(self):
        features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        namespace = build_artifact_key(
            "sha256:" + "12" * 32,
            modality="image",
            processor_fingerprint="processor-v1",
        )
        other_namespace = build_artifact_key(
            "sha256:" + "12" * 32,
            modality="image",
            processor_fingerprint="processor-v2",
        )
        hashes = [
            resolve_multimodal_item_hash(
                feature=feature,
                namespace=scope,
                model_specific_data={"grid": grid},
            )
            for feature, scope, grid in (
                (features, namespace, (1, 2, 3)),
                (features + 1, namespace, (1, 2, 3)),
                (features, namespace, (1, 3, 2)),
                (features, other_namespace, (1, 2, 3)),
            )
        ]
        self.assertEqual(len(set(hashes)), len(hashes))
        for value in hashes:
            self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")

    def test_logical_tensor_content_is_independent_of_storage_layout(self):
        features = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).T
        self.assertEqual(
            resolve_multimodal_item_hash(feature=features),
            resolve_multimodal_item_hash(feature=features.contiguous()),
        )
        self.assertNotEqual(
            resolve_multimodal_item_hash(feature=features),
            resolve_multimodal_item_hash(feature=features.reshape(4, 6)),
        )
        self.assertNotEqual(
            resolve_multimodal_item_hash(feature=features),
            resolve_multimodal_item_hash(feature=features.float()),
        )

    def test_item_grid_is_hashed_before_transport(self):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.ones(6, 4),
                model_specific_data={"image_grid_thw": torch.tensor([grid])},
            )
            for grid in ((1, 2, 3), (1, 3, 2))
        ]
        for item in items:
            item.set_pad_value()
        self.assertNotEqual(items[0].cache_key, items[1].cache_key)

    def test_typed_round_trip_keeps_full_identity_and_bounded_padding(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            offsets=[(1, 3)],
        )
        item.set_pad_value()
        output = MultimodalProcessorOutput(mm_items=[item], input_ids=[2, 1, 1, 1, 3])
        encoded = msgspec.msgpack.encode(output, enc_hook=enc_hook)
        decoded = msgspec.msgpack.decode(
            encoded, type=MultimodalProcessorOutput, ext_hook=ext_hook
        )
        self.assertEqual(decoded.mm_items[0].hash, item.hash)
        self.assertEqual(decoded.mm_items[0].cache_identity, item.cache_identity)
        self.assertRegex(decoded.mm_items[0].hash, r"^sha256:[0-9a-f]{64}$")
        padded = MultimodalProcessorOutput.build_padded_input_ids(
            decoded.input_ids, decoded.mm_items
        )
        self.assertEqual(list(array("q", padded)), padded)
        self.assertTrue(torch.equal(decoded.mm_items[0].feature, item.feature))

    def test_legacy_and_wide_caller_hashes_round_trip_without_bit_loss(self):
        for identity in (1234, (1 << 64) - 1, (1 << 250) + 123456789):
            with self.subTest(identity=identity):
                item = MultimodalDataItem(modality=Modality.IMAGE, hash=identity)
                item.set_pad_value()
                encoded = msgspec.msgpack.encode(item)
                decoded = msgspec.msgpack.decode(encoded, type=MultimodalDataItem)
                self.assertEqual(multimodal_hash_as_int(decoded.hash), identity)
                if identity < 1 << 64:
                    self.assertEqual(decoded.hash, identity)

    def test_combined_embedding_cache_keeps_order_and_full_keys(self):
        hashes = [
            resolve_multimodal_item_hash(feature=torch.full((2, 3), value))
            for value in (1.0, 2.0)
        ]
        cache = MultiModalStaticCache(max_size=1024)
        embedding = EmbeddingResult(embedding=torch.arange(12).reshape(4, 3))
        key = cache.combine_hashes(hashes)
        cache.set(key, embedding)
        self.assertIs(cache.get(hashes), embedding)
        self.assertIsNone(cache.get(list(reversed(hashes))))
        self.assertEqual(key, tuple(hashes))

    def test_raw_encoder_identity_retains_complete_digest(self):
        value = hash_raw_encoder_item(
            torch.arange(24, dtype=torch.uint8).reshape(2, 4, 3)
        )
        self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")

    def test_routing_hash_override_keeps_content_authority(self):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.full((2, 3), value),
                offsets=[(1, 2)],
            )
            for value in (1.0, 2.0)
        ]
        identities = []
        for item in items:
            item.set_pad_value()
            original_identity = item.cache_identity
            item.set_hash(1234)
            self.assertEqual(item.cache_identity, original_identity)
            inputs = MultimodalInputs(mm_items=[item])
            spans = inputs.cache_spans(
                array("q", [1, item.pad_value, item.pad_value, 2])
            )
            identities.append(spans[0].identity)
        self.assertEqual(items[0].pad_value, items[1].pad_value)
        self.assertNotEqual(identities[0], identities[1])

    def test_item_ranges_retain_embedding_positions_across_gaps(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.ones(4, 3),
            offsets=[(1, 2), (4, 5)],
        )
        item.set_pad_value()
        spans = MultimodalInputs(mm_items=[item]).cache_spans(array("q", [1] * 7))
        self.assertEqual([span.offset for span in spans], [0, 2])

    def test_video_identity_keeps_frame_layout_but_not_unrelated_prompt_text(self):
        items = []
        for first, separator, last in ((1, 3, 7), (1, 3, 8), (2, 3, 7), (1, 4, 7)):
            item = MultimodalDataItem(
                modality=Modality.VIDEO,
                feature=torch.arange(12, dtype=torch.float32).reshape(4, 3),
                offsets=[(1, 2), (4, 5)],
                model_specific_data={
                    "thw_grids": [(2, 1, 2)],
                    "pre_chunked_input_ids": [first, 0, 0, separator, 0, 0, last],
                },
            )
            item.set_pad_value()
            items.append(item)
        self.assertEqual(items[0].cache_key, items[1].cache_key)
        self.assertEqual(items[0].cache_key, items[2].cache_key)
        self.assertNotEqual(items[0].cache_key, items[3].cache_key)

    def test_session_append_retains_historical_media_spans(self):
        prior = MultimodalDataItem(
            modality=Modality.IMAGE, feature=torch.ones(2, 3), offsets=[(1, 2)]
        )
        prior.set_pad_value()
        old_inputs = MultimodalInputs(mm_items=[prior])
        prompt = array("q", [1, prior.pad_value, prior.pad_value, 2, 3, 4])
        old_key = RadixKey(prompt, mm_spans=old_inputs.cache_spans(prompt))
        req = Req.__new__(Req)
        req.session = object()
        req.multimodal_inputs = old_inputs
        req.origin_input_ids = prompt + array("q", [5, 6])
        req._mm_cache_spans = None
        self.assertEqual(
            RadixKey(req.origin_input_ids, mm_spans=req.mm_cache_spans).match(old_key),
            len(old_key),
        )
        new_item = MultimodalDataItem(
            modality=Modality.IMAGE, feature=torch.full((1, 3), 2.0), offsets=[(1, 1)]
        )
        new_item.set_pad_value()
        next_inputs = MultimodalInputs(mm_items=[new_item])
        SessionController.adjust_mm_offsets(
            SimpleNamespace(input_ids=[5, 6]), req, next_inputs
        )
        req.origin_input_ids[-1] = new_item.pad_value
        req.extend_image_inputs(next_inputs)
        self.assertEqual([span.start for span in req.mm_cache_spans], [1, 7])
        self.assertEqual(len(old_inputs.mm_items), 1)
        self.assertEqual(
            RadixKey(req.origin_input_ids, mm_spans=req.mm_cache_spans).match(old_key),
            len(old_key),
        )

    def test_embedding_mask_uses_request_offsets_with_shared_routing_hints(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                input_ids = torch.tensor(
                    [1, 1000001, 1000001, 2, 1000001], device=device
                )
                mask = _get_multimodal_mask(
                    input_ids,
                    torch.tensor([1000001], device=device),
                    prefix_length=[0, 2],
                    extend_length=[4, 1],
                    items_offset_list=[[(1, 2)], []],
                )
                self.assertEqual(
                    mask.flatten().cpu().tolist(), [False, True, True, False, False]
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cpu_and_cuda_tensor_content_produce_the_same_identity(self):
        for dtype in (torch.uint8, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                feature = (
                    torch.arange(120, dtype=dtype).reshape(5, 8, 3).transpose(0, 1)
                )
                cpu_hash = resolve_multimodal_item_hash(feature=feature)
                cuda_hash = resolve_multimodal_item_hash(feature=feature.cuda())
                self.assertEqual(cpu_hash, cuda_hash)


class TestContentIdentityIngress(CustomTestCase):
    def test_content_hash_requires_exactly_64_ascii_hex_characters(self):
        """Whitespace accepted by a byte decoder is not part of the digest schema."""
        self.assertIsNone(parse_content_hash(None))
        for digest in ("ab" * 32, "AB" * 32, "aB09" * 16):
            with self.subTest(valid=digest):
                self.assertEqual(
                    parse_content_hash("sha256:" + digest), "sha256:" + digest.lower()
                )
        for digest in (
            "a" * 63,
            "a" * 65,
            "a" * 62 + " \t",
            "ab" * 15 + "\n " + "cd" * 16,
            "g" * 64,
            "ａ" * 64,
        ):
            with self.subTest(invalid=digest), self.assertRaises(ValueError):
                parse_content_hash("sha256:" + digest)
        with self.assertRaises(ValueError):
            parse_content_hash("SHA256:" + "ab" * 32)

    @staticmethod
    def _collect(data):
        processor = SimpleNamespace(
            ATTR_NAME_TO_MODALITY={
                "pixel_values": Modality.IMAGE,
                "image_grid_thw": Modality.IMAGE,
                "audio_features": Modality.AUDIO,
            },
            FEATURE_NAMES=["pixel_values", "audio_features"],
        )
        return BaseMultimodalProcessor.collect_mm_items_from_processor_output(
            processor, data
        )

    def test_payload_cache_key_is_derived_before_any_explicit_padding(self):
        """A routing hint cannot replace available feature or embedding content."""
        feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        for field in ("feature", "precomputed_embeddings"):
            with self.subTest(field=field):
                item = MultimodalDataItem(
                    modality=Modality.IMAGE, hash=17, **{field: feature}
                )
                other_hint = MultimodalDataItem(
                    modality=Modality.IMAGE, hash=18, **{field: feature}
                )
                identity = item.cache_key
                self.assertRegex(identity, r"^sha256:[0-9a-f]{64}$")
                self.assertEqual(identity, other_hint.cache_key)
                self.assertEqual(item.hash, 17)
                encoded = msgspec.msgpack.encode(item, enc_hook=enc_hook)
                decoded = msgspec.msgpack.decode(
                    encoded, type=MultimodalDataItem, ext_hook=ext_hook
                )
                self.assertEqual(decoded.cache_key, identity)
                torch.testing.assert_close(getattr(decoded, field), feature)

    def test_offline_identity_survives_collection_routing_and_wire(self):
        """Explicit processor authority survives payload omission and route overrides."""
        feature = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        identity = resolve_multimodal_item_hash(
            feature=feature, model_specific_data={"processor": "offline-v1"}
        )
        for field in ("pixel_values", "precomputed_embeddings", None):
            for routing_hash, pad_value in ((None, None), (17, None), (17, 1000123)):
                with self.subTest(
                    field=field, routing_hash=routing_hash, pad_value=pad_value
                ):
                    data = {
                        "format": "processor_output",
                        "modality": "image",
                        "cache_identity": identity,
                        "offsets": torch.tensor([[1, 3]]),
                        "image_grid_thw": torch.tensor([[1, 1, 3]]),
                    }
                    if field is not None:
                        data[field] = feature
                    if routing_hash is not None:
                        data["hash"] = routing_hash
                    if pad_value is not None:
                        data["pad_value"] = pad_value
                    items = self._collect(data)
                    self.assertEqual(len(items), 1)
                    item = items[0]
                    expected_hash = identity if routing_hash is None else routing_hash
                    expected_pad = (
                        _compute_pad_value(expected_hash)
                        if pad_value is None
                        else pad_value
                    )
                    self.assertEqual(item.cache_identity, identity)
                    self.assertEqual(item.hash, expected_hash)
                    self.assertEqual(item.pad_value, expected_pad)
                    output = MultimodalProcessorOutput(mm_items=items)
                    encoded = msgspec.msgpack.encode(output, enc_hook=enc_hook)
                    decoded = msgspec.msgpack.decode(
                        encoded, type=MultimodalProcessorOutput, ext_hook=ext_hook
                    )
                    inputs = MultimodalInputs.from_processor_output(decoded)
                    self.assertEqual(inputs.mm_items[0].cache_key, identity)
                    self.assertEqual(inputs.mm_items[0].pad_value, expected_pad)
                    self.assertEqual(inputs.mm_items[0].offsets, [(1, 3)])
                    spans = inputs.cache_spans(
                        array("q", [1, expected_pad, expected_pad, expected_pad])
                    )
                    self.assertEqual(spans[0].identity, identity)
                    self.assertEqual((spans[0].start, spans[0].end), (1, 4))

    def test_routing_only_padding_does_not_authorize_cache_use(self):
        """Legacy padding remains valid while payload-free cache use is rejected."""
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=17, offsets=[(1, 2)])
        item.set_pad_value()
        self.assertEqual(item.hash, 17)
        self.assertEqual(item.pad_value, _compute_pad_value(17))
        self.assertIsNone(item.cache_identity)
        with self.assertRaisesRegex(ValueError, "processor content identity"):
            _ = item.cache_key
        with self.assertRaisesRegex(ValueError, "processor content identity"):
            MultimodalInputs.from_processor_output(
                MultimodalProcessorOutput(mm_items=[item])
            )
        with self.assertRaisesRegex(ValueError, "processor content identity"):
            MultimodalInputs(mm_items=[item]).cache_spans(array("q", [1, 2, 3]))

    def test_explicit_key_only_item_remains_valid_without_feature_bytes(self):
        """Artifact and native producers can retain their previously computed key."""
        identity = resolve_multimodal_item_hash(feature=torch.arange(4))
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=17,
            cache_identity=identity,
            offsets=[(1, 2)],
        )
        inputs = MultimodalInputs.from_processor_output(
            MultimodalProcessorOutput(mm_items=[item])
        )
        self.assertIs(inputs.mm_items[0], item)
        self.assertEqual(item.cache_key, identity)
        self.assertIsNone(item.feature)
        self.assertIsNone(item.precomputed_embeddings)

    def test_malformed_or_ambiguous_explicit_identity_is_rejected(self):
        """An explicit authority field must be a full digest for one item."""
        for identity in (17, "invalid", "sha256:1234"):
            with self.subTest(identity=identity):
                with self.assertRaises(ValueError):
                    self._collect(
                        {"pixel_values": torch.ones(1), "cache_identity": identity}
                    )
                item = MultimodalDataItem(
                    modality=Modality.IMAGE, hash=17, cache_identity=identity
                )
                with self.assertRaises(ValueError):
                    _ = item.cache_key
        identity = resolve_multimodal_item_hash(feature=torch.arange(4))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self._collect(
                {
                    "pixel_values": torch.ones(1),
                    "audio_features": torch.ones(1),
                    "cache_identity": identity,
                }
            )


if __name__ == "__main__":
    unittest.main()
