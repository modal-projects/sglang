"""Encoder embedding caches use media content and resolved model configuration."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessResult
from sglang.srt.disaggregation.encoder.server import (
    BadRequestError,
    InternalError,
    MMEncoder,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.multimodal.encoder_preprocessing import (
    EncoderMediaProcessorConfig,
    EncoderPreprocessOutput,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _encoder(rank=0):
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.rank = rank
    encoder.model_type = "example_vision"
    encoder.model = SimpleNamespace(get_image_feature=Mock())
    encoder._global_cache_namespace = "sha256:" + "12" * 32
    encoder.preprocessor = SimpleNamespace(
        get_num_patches=lambda grid, modality: int(torch.as_tensor(grid).prod()),
    )
    encoder.mm_cache = MultiModalStaticCache(1024 * 1024)
    encoder.mm_cache_lock = asyncio.Lock()
    return encoder


def _context(encoder, feature, grid, hashes, *, items_per_req=None):
    grids = torch.as_tensor(grid)
    mm_inputs = {"pixel_values": feature, "image_grid_thw": grids}
    counts = [1] * len(grids)
    encoder.preprocessor.process_batch_mm_items = AsyncMock(
        return_value=(
            EncoderPreprocessResult(mm_inputs, grids, counts),
            items_per_req or [len(grids)],
        )
    )
    requests = [
        {"req_id": f"req-{index}", "hashes": value}
        for index, value in enumerate(hashes)
    ]
    with patch(
        "sglang.srt.disaggregation.encoder.server.encoder_metrics_collector", None
    ):
        return asyncio.run(
            encoder._prepare_encode_context(
                requests, Modality.IMAGE, use_global_cache=True
            )
        )


def test_routing_hints_do_not_replace_global_content_keys():
    encoder = _encoder()
    first = _context(encoder, torch.zeros(6, 2), [[1, 2, 3]], [["routing-hint"]])
    second = _context(encoder, torch.ones(6, 2), [[1, 2, 3]], [["routing-hint"]])
    rerouted = _context(encoder, torch.zeros(6, 2), [[1, 2, 3]], [["other-hint"]])
    assert first.str_mm_hashes != second.str_mm_hashes
    assert first.str_mm_hashes == rerouted.str_mm_hashes
    assert first.str_mm_hashes[0].startswith("sha256:")
    retained = set(first.str_mm_hashes)
    encoder.mm_global_cache = SimpleNamespace(
        batch_is_exist=AsyncMock(
            side_effect=lambda keys: [key in retained for key in keys]
        )
    )
    encoder._broadcast_global_cache_mask = lambda mask: None
    assert asyncio.run(encoder._lookup_global_cache(first)) == ([], [0])
    assert asyncio.run(encoder._lookup_global_cache(second)) == ([0], [])
    assert asyncio.run(encoder._lookup_global_cache(rerouted)) == ([], [0])


def test_generic_grid_and_auxiliary_model_inputs_contribute_to_identity():
    encoder = _encoder()
    features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    first = encoder._calculate_hashes_from_features(
        features, [[1, 2, 3]], Modality.IMAGE
    )
    changed_grid = encoder._calculate_hashes_from_features(
        features, [[1, 3, 2]], Modality.IMAGE
    )
    changed_metadata = encoder._calculate_hashes_from_features(
        features,
        [[1, 2, 3]],
        Modality.IMAGE,
        {"pixel_values": features, "position_scale": 2},
    )
    assert first != changed_grid
    assert first != changed_metadata


def test_audio_rows_include_their_actual_feature_and_length():
    encoder = _encoder()
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    keys = encoder._calculate_hashes_from_features(
        features, torch.tensor([4, 4]), Modality.AUDIO
    )
    reversed_keys = encoder._calculate_hashes_from_features(
        features.flip(0), torch.tensor([4, 4]), Modality.AUDIO
    )
    assert len(keys) == 2
    assert keys == list(reversed(reversed_keys))


def test_precomputed_authority_survives_featureless_routing_override():
    encoder = _encoder()
    item = MultimodalDataItem(modality=Modality.IMAGE)
    identity = "sha256:" + "34" * 32
    item.set_content_hash(identity)
    item.set_hash(42)
    mm_inputs = EncoderPreprocessOutput({"pixel_values": [None]}, mm_items=[item])
    assert encoder._calculate_hashes_from_features(
        [None], [[1, 2, 3]], Modality.IMAGE, mm_inputs
    ) == [identity]
    assert item.cache_identity == identity
    assert item.hash == 42


def test_featureless_routing_hint_cannot_supply_content_authority():
    encoder = _encoder()
    item = MultimodalDataItem(modality=Modality.IMAGE, hash=42)
    with pytest.raises(InternalError, match="processor content identity"):
        encoder._item_cache_identity(item)


@pytest.mark.parametrize(
    "hint", [123, "routing-hint", b"routing-hint", ["routing-hint"]]
)
def test_single_grid_keeps_scalar_and_list_routing_hint_api(hint):
    ctx = _context(_encoder(), torch.zeros(6, 2), [[1, 2, 3]], [hint])
    assert len(ctx.str_mm_hashes) == 1


@pytest.mark.parametrize("rank", [0, 1])
def test_mixed_supplied_and_absent_hints_validate_before_collectives(rank):
    encoder = _encoder(rank)
    encoder._calculate_hashes_from_features = Mock(
        side_effect=AssertionError("must validate first")
    )
    with pytest.raises(BadRequestError, match="grid count 1"):
        _context(
            encoder,
            torch.zeros(12, 2),
            [[1, 2, 3], [1, 2, 3]],
            [["first", "extra"], None],
            items_per_req=[1, 1],
        )
    encoder._calculate_hashes_from_features.assert_not_called()


def test_secondary_rank_keeps_hashing_on_primary_and_shared_mask_order():
    encoder = _encoder(rank=1)
    encoder._calculate_hashes_from_features = Mock(
        side_effect=AssertionError("rank zero owns hashing")
    )
    ctx = _context(encoder, torch.zeros(18, 2), [[1, 2, 3]] * 3, [["hint"] * 3])
    assert ctx.str_mm_hashes is None
    encoder.mm_global_cache = SimpleNamespace(batch_is_exist=AsyncMock())
    encoder._broadcast_global_cache_mask = lambda mask: mask.copy_(
        torch.tensor([1, 0, 1])
    )
    missing, hits = asyncio.run(encoder._lookup_global_cache(ctx))
    assert missing == [1]
    assert hits == [0, 2]
    encoder.mm_global_cache.batch_is_exist.assert_not_awaited()


def test_direct_cache_keeps_independent_content_under_equal_routing_hints():
    async def run():
        encoder = _encoder()
        grids = torch.tensor([[1, 1, 2]])
        calls = []

        def forward(items):
            calls.append(items[0].cache_identity)
            return items[0].feature.clone()

        def context(value):
            item = MultimodalDataItem(
                modality=Modality.IMAGE, hash=42, feature=torch.full((2, 3), value)
            )
            inputs = EncoderPreprocessOutput(
                {"pixel_values": [item.feature]}, mm_items=[item]
            )
            return SimpleNamespace(
                modality=Modality.IMAGE,
                mm_feature=[item.feature],
                preprocess_result=EncoderPreprocessResult(inputs, grids, [2]),
                num_items=1,
                items_per_req=[1],
                is_health_check=False,
                get_feature_fn=forward,
                aux_data={},
            )

        with patch(
            "sglang.srt.disaggregation.encoder.server.get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        ):
            first = await encoder._compute_direct_embedding(
                context(1.0), keep_on_gpu=False
            )
            second = await encoder._compute_direct_embedding(
                context(2.0), keep_on_gpu=False
            )
            repeated = await encoder._compute_direct_embedding(
                context(1.0), keep_on_gpu=False
            )
        assert len(calls) == 2
        assert len(encoder.mm_cache) == 2
        torch.testing.assert_close(first, repeated)
        assert not torch.equal(first, second)

    asyncio.run(run())


def test_global_namespace_tracks_public_model_revision_and_resolved_processor_config():
    encoder = _encoder()
    encoder._embedding_dtype = torch.float32
    encoder.model_config = SimpleNamespace(
        revision="revision-a",
        quantization=None,
        hf_config=SimpleNamespace(
            to_dict=lambda: {
                "model_type": "example_vision",
                "architectures": ["ExampleModel"],
            }
        ),
    )
    encoder.preprocessor.vision_config = {"image": {"size": 256}}
    encoder.preprocessor.use_image_processor_gpu = False
    encoder.preprocessor.encoder_media_processor_config = EncoderMediaProcessorConfig()
    encoder.preprocessor.model_audio_sr = 16000
    with get_context().override_server_args(
        model_path="example/model-a",
        served_model_name="example-model",
        revision="revision-a",
    ):
        original = encoder._build_global_cache_namespace()
        encoder.preprocessor.vision_config = {"image": {"size": 512}}
        changed_config = encoder._build_global_cache_namespace()
        encoder.preprocessor.vision_config = {"image": {"size": 256}}
        encoder.model_config.revision = "revision-b"
        changed_revision = encoder._build_global_cache_namespace()
    with get_context().override_server_args(
        model_path="example/model-b",
        served_model_name="example-model",
        revision="revision-a",
    ):
        encoder.model_config.revision = "revision-a"
        changed_model = encoder._build_global_cache_namespace()
    assert len({original, changed_config, changed_revision, changed_model}) == 4


def test_featureless_authority_is_scoped_for_storage_without_mutating_the_item():
    encoder = _encoder()
    item = MultimodalDataItem(modality=Modality.IMAGE)
    identity = "sha256:" + "34" * 32
    item.set_content_hash(identity)
    item.set_hash(42)
    grid = torch.tensor([[1, 2, 3]])
    inputs = EncoderPreprocessOutput({"pixel_values": [None]}, mm_items=[item])
    encoder.preprocessor.process_batch_mm_items = AsyncMock(
        return_value=(EncoderPreprocessResult(inputs, grid, [2]), [1])
    )
    with patch(
        "sglang.srt.disaggregation.encoder.server.encoder_metrics_collector", None
    ):
        ctx = asyncio.run(
            encoder._prepare_encode_context(
                [{"req_id": "featureless", "hashes": ["routing-hint"]}],
                Modality.IMAGE,
                use_global_cache=True,
            )
        )
    assert len(ctx.str_mm_hashes) == 1
    assert ctx.str_mm_hashes[0].startswith("sha256:")
    assert ctx.str_mm_hashes[0] != identity
    assert item.cache_identity == identity
    assert item.hash == 42


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
