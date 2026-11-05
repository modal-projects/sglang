import pytest
import torch

from sglang.srt.layers.utils.common import update_derived_buffer


def test_derived_buffer_reload_preserves_storage():
    layer = torch.nn.Module()
    update_derived_buffer(layer, "packed_weight", torch.arange(4))
    pointer = layer.packed_weight.data_ptr()

    update_derived_buffer(layer, "packed_weight", torch.arange(4) + 1)

    assert layer.packed_weight.data_ptr() == pointer
    torch.testing.assert_close(layer.packed_weight, torch.arange(4) + 1)
    assert "packed_weight" in layer._non_persistent_buffers_set


def test_derived_buffer_reload_rejects_layout_change():
    layer = torch.nn.Module()
    update_derived_buffer(layer, "packed_weight", torch.arange(4))

    with pytest.raises(RuntimeError, match="layout changed"):
        update_derived_buffer(layer, "packed_weight", torch.arange(5))
