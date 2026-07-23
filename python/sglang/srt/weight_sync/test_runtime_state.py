import torch

from sglang.srt.weight_sync.runtime_state import (
    checkpoint_module_path,
    clone_module_proxy,
)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4, bias=False) for _ in range(3)]
        )


def test_checkpoint_module_path_bounds_large_groups():
    assert (
        checkpoint_module_path(
            "language_model.model.layers.17.mlp.experts.250.down_proj.weight"
        )
        == "language_model.model.layers.17"
    )
    assert (
        checkpoint_module_path("vision_tower.encoder.blocks.63.wqkv.weight")
        == "vision_tower.encoder.blocks.63"
    )
    assert checkpoint_module_path("mm_projector.proj.0.weight") == "mm_projector"


def test_clone_module_proxy_replaces_only_selected_path():
    model = _ToyModel()
    proxy, shadow = clone_module_proxy(model, "language_model.model.layers.1")

    assert shadow is proxy.language_model.model.layers[1]
    assert shadow is not model.language_model.model.layers[1]
    assert proxy.language_model.model.layers[0] is model.language_model.model.layers[0]
    assert proxy.language_model.model.layers[2] is model.language_model.model.layers[2]

    with torch.no_grad():
        shadow.weight.fill_(42)
    assert not torch.equal(shadow.weight, model.language_model.model.layers[1].weight)
