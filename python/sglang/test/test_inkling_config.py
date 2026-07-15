from __future__ import annotations

import pytest

from sglang.srt.configs.inkling import InklingMMConfig


def test_inkling_mtp_local_layer_ids():
    config = InklingMMConfig(
        text_config={"num_nextn_predict_layers": 8},
        mtp_config={
            "local_layer_ids": [0, 2, 4, 5, 6, 7],
            "num_nextn_predict_layers": 8,
        },
    )
    assert config.mtp_local_layer_ids == frozenset({0, 2, 4, 5, 6, 7})


def test_inkling_mtp_local_layer_ids_reject_invalid_config():
    config = InklingMMConfig(
        text_config={"num_nextn_predict_layers": 8},
        mtp_config={"local_layer_ids": [0, 8]},
    )
    with pytest.raises(ValueError, match="mtp_config.local_layer_ids"):
        _ = config.mtp_local_layer_ids
