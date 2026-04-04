import unittest

try:
    import torch
    from safetensors.torch import save as save_safetensors

    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
    from sglang.srt.weight_sync.update_bytes import (
        build_update_weights_request_from_named_tensors,
        load_named_tensors_from_bytes,
    )
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Missing test dependency: {exc}")


class TestUpdateWeightsFromBytes(unittest.TestCase):
    def test_load_named_tensors_from_safetensors_bytes(self):
        payload = save_safetensors(
            {
                "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "b": torch.tensor([3, 4], dtype=torch.int64),
            }
        )

        named_tensors = load_named_tensors_from_bytes(payload)

        self.assertEqual(set(name for name, _ in named_tensors), {"a", "b"})
        named_tensor_dict = dict(named_tensors)
        self.assertTrue(
            torch.equal(
                named_tensor_dict["a"],
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(named_tensor_dict["b"], torch.tensor([3, 4], dtype=torch.int64))
        )

    def test_build_update_request_uses_local_bytes(self):
        named_tensors = [("a", torch.tensor([1.0], dtype=torch.float32))]

        request = build_update_weights_request_from_named_tensors(
            named_tensors,
            tp_size=2,
            load_format="custom",
            flush_cache=False,
            abort_all_requests=True,
            base_weight_version="v0",
            weight_version="v1",
            payload_digest="abc",
            loader_metadata={"x": 1},
            crash_on_error=True,
        )

        self.assertEqual(len(request.serialized_named_tensors), 2)
        self.assertTrue(all(isinstance(item, bytes) for item in request.serialized_named_tensors))
        self.assertEqual(request.load_format, "custom")
        self.assertFalse(request.flush_cache)
        self.assertTrue(request.abort_all_requests)
        self.assertEqual(request.base_weight_version, "v0")
        self.assertEqual(request.weight_version, "v1")
        self.assertEqual(request.payload_digest, "abc")
        self.assertEqual(request.loader_metadata, {"x": 1})
        self.assertTrue(request.crash_on_error)

        round_tripped = MultiprocessingSerializer.deserialize(
            request.serialized_named_tensors[0]
        )
        self.assertEqual(len(round_tripped), 1)
        self.assertEqual(round_tripped[0][0], "a")
        self.assertTrue(torch.equal(round_tripped[0][1], named_tensors[0][1]))

    def test_build_update_request_uses_bucketed_transport(self):
        named_tensors = [
            ("a", torch.arange(8, dtype=torch.float32).reshape(2, 4)),
            ("b", torch.arange(4, dtype=torch.int64)),
        ]

        request = build_update_weights_request_from_named_tensors(
            named_tensors,
            tp_size=2,
            load_format="custom_loader",
            transport_format="flattened_bucket",
            transport_bucket_bytes=16,
        )

        self.assertEqual(request.load_format, "custom_loader")
        self.assertEqual(request.transport_format, "flattened_bucket")
        self.assertIsNotNone(request.transport_metadata)
        self.assertGreaterEqual(request.transport_metadata["bucket_count"], 2)

        payload = MultiprocessingSerializer.deserialize(request.serialized_named_tensors[0])
        self.assertIsInstance(payload, list)
        reconstructed = []
        for bucket_dict in payload:
            bucket = FlattenedTensorBucket(
                flattened_tensor=bucket_dict["flattened_tensor"],
                metadata=bucket_dict["metadata"],
            )
            reconstructed.extend(bucket.reconstruct_tensors())

        reconstructed_dict = dict(reconstructed)
        self.assertEqual(set(reconstructed_dict), {"a", "b"})
        self.assertTrue(torch.equal(reconstructed_dict["a"], named_tensors[0][1]))
        self.assertTrue(torch.equal(reconstructed_dict["b"], named_tensors[1][1]))

    def test_build_update_request_normalizes_legacy_flattened_bucket_load_format(self):
        named_tensors = [("a", torch.tensor([1.0], dtype=torch.float32))]
        request = build_update_weights_request_from_named_tensors(
            named_tensors,
            tp_size=1,
            load_format="flattened_bucket",
        )

        self.assertIsNone(request.load_format)
        self.assertEqual(request.transport_format, "flattened_bucket")


if __name__ == "__main__":
    unittest.main()
