from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers import mm_schedule, mm_utils
from sglang.srt.managers.mm_utils import _scatter_mm_embedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.cosmos3_edge import Cosmos3EdgeForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

NUM_TOKENS = 64


def _make_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
    if pattern == "interleaved":
        mask[::3] = True
    elif pattern == "blocks":
        mask[5:20] = True
        mask[40:41] = True
    elif pattern == "all_true":
        mask[:] = True
    return mask.unsqueeze(-1)


@pytest.mark.parametrize("width", [8, 24])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mask_pattern", ["interleaved", "blocks", "all_true", "all_false"]
)
def test_scatter_matches_masked_scatter_bitwise(width, src_dtype, mask_pattern):
    """The row-index mm embedding merge must stay bitwise identical to
    masked_scatter_ semantics, whose internal transients it avoids."""
    torch.manual_seed(0)
    mask = _make_mask(mask_pattern)
    dest = torch.randn(NUM_TOKENS, width).to(torch.bfloat16)
    src = torch.randn(int(mask.sum()), width, dtype=src_dtype)

    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    actual = dest.clone()
    _scatter_mm_embedding(dest=actual, mask=mask, src=src)
    assert torch.equal(actual, expected)


def test_scatter_row_count_mismatch_fails_loud():
    """A mask/src row-count mismatch must raise, not silently corrupt rows."""
    dest = torch.zeros(8, 4)
    src_short_mask = _make_mask("all_false")[:8]
    src_short_mask[1] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(dest=dest, mask=src_short_mask, src=torch.ones(3, 4))
    mask_heavy = src_short_mask.clone()
    mask_heavy[2:6] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(dest=dest, mask=mask_heavy, src=torch.ones(1, 4))


class _EmbeddingOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 3)
        with torch.no_grad():
            self.embedding.weight.fill_(-1)
        self.pp_group = SimpleNamespace(is_first_rank=True)

    def get_input_embeddings(self):
        return self.embedding

    def get_image_feature(self, items):
        return torch.cat([item.feature for item in items])

    def forward(self, *, input_embeds, **kwargs):
        return input_embeds


class TestPackedMMEmbedding(CustomTestCase):
    def _make_batch(self, precomputed):
        media_inputs = []
        padded = []
        for index, (ids, offsets, values) in enumerate(
            (
                ([9, 31, 31, 31, 8], [(1, 3)], [100, 101, 102]),
                ([7, 8, 31, 31, 9], [(2, 3)], [200, 201]),
            )
        ):
            feature = (
                torch.tensor(values, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
            )
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=offsets,
                feature=None if precomputed[index] else feature,
                precomputed_embeddings=feature if precomputed[index] else None,
            )
            item.set_pad_value()
            media = MultimodalInputs(mm_items=[item], im_token_id=31)
            media_inputs.append(media)
            padded.append(
                mm_utils.MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
                    ids, media
                )
            )

        # Text requests surround two media requests with partial prompt prefixes.
        input_ids = torch.tensor(
            [1, 2] + padded[0][2:] + [3, 4, 5] + padded[1][1:] + [6]
        )
        batch = ForwardBatch.__new__(ForwardBatch)
        batch.forward_mode = ForwardMode.EXTEND
        batch.mm_inputs = [None, media_inputs[0], None, media_inputs[1], None]
        batch.extend_prefix_lens_cpu = [0, 2, 0, 1, 0]
        batch.extend_seq_lens_cpu = [2, 3, 3, 4, 1]
        batch.input_embeds = None
        expected = (
            torch.tensor(
                [-1, -1, 101, 102, -1, -1, -1, -1, -1, 200, 201, -1, -1],
                dtype=torch.float32,
            )
            .unsqueeze(1)
            .repeat(1, 3)
        )
        return input_ids, batch, expected

    def test_general_embedding_preserves_packed_request_positions(self):
        """Text-only requests must not shift media rows, including split subgroups."""
        for adaptive, precomputed in (
            (False, (False, False)),
            (False, (True, True)),
            (True, (False, False)),
            (True, (True, True)),
            (True, (False, True)),
            (True, (True, False)),
        ):
            with self.subTest(adaptive=adaptive, precomputed=precomputed):
                input_ids, batch, expected = self._make_batch(precomputed)
                model = _EmbeddingOnlyModel()
                disagg = SimpleNamespace(
                    enable_adaptive_dispatch_to_encoder=adaptive, language_only=False
                )
                with (
                    patch.object(
                        mm_schedule, "embedding_cache", MultiModalStaticCache(1024)
                    ),
                    patch.object(mm_utils, "get_server_args", return_value=object()),
                    patch.object(mm_utils, "get_disagg", return_value=disagg),
                ):
                    actual = mm_utils.general_mm_embed_routine(
                        input_ids=input_ids,
                        forward_batch=batch,
                        language_model=model,
                        multimodal_model=model,
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cosmos_embedding_preserves_packed_request_positions(self):
        """The direct model embedding caller retains text request positions too."""
        for precomputed in ((False, False), (True, True)):
            with self.subTest(precomputed=precomputed):
                input_ids, batch, expected = self._make_batch(precomputed)
                with patch.object(
                    mm_schedule, "embedding_cache", MultiModalStaticCache(1024)
                ):
                    actual = (
                        Cosmos3EdgeForConditionalGeneration._embed_multimodal_inputs(
                            _EmbeddingOnlyModel(),
                            input_ids=input_ids,
                            forward_batch=batch,
                        )
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
