import torch

from sglang.srt.weight_sync.runtime_state import (
    checkpoint_module_path,
    clone_module_proxy,
    clone_module_tensors,
    runtime_module_path,
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


def test_runtime_module_path_includes_derived_state_but_excludes_static_state():
    assert (
        runtime_module_path(
            "language_model.model.layers.17.mlp.experts.w13_weight_scale"
        )
        == "language_model.model.layers.17"
    )
    assert runtime_module_path("language_model.model.rotary_emb.inv_freq") is None
    assert runtime_module_path("language_model.model.layers.17.rotary_emb.inv_freq") == (
        "language_model.model.layers.17"
    )


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


def test_clone_module_tensors_preserves_aliases_and_shares_non_tensor_state():
    module = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.arange(8.0))
    parameter.weight_loader = "sentinel"
    module.register_parameter("weight", parameter)
    module.alias = parameter[2:6]
    module.runtime_object = object()

    cloned = clone_module_tensors(module)

    assert cloned.weight is not module.weight
    assert cloned.weight.data_ptr() != module.weight.data_ptr()
    assert cloned.weight.weight_loader == "sentinel"
    assert cloned.alias.untyped_storage().data_ptr() == cloned.weight.data_ptr()
    assert cloned.alias.storage_offset() == 2
    assert cloned.runtime_object is module.runtime_object
    with torch.no_grad():
        cloned.weight.add_(10)
    assert torch.equal(module.weight, torch.arange(8.0))


def test_clone_module_tensors_preserves_sglang_parameter_subclass():
    from sglang.srt.layers.parameter import ModelWeightParameter

    parameter = ModelWeightParameter(
        data=torch.empty(2, 2),
        output_dim=1,
        input_dim=1,
        weight_loader=lambda _: None,
    )
    module = torch.nn.Module()
    module.register_parameter("weight", parameter)

    cloned = clone_module_tensors(module)

    assert type(cloned.weight) is type(parameter)
    assert cloned.weight.output_dim == parameter.output_dim
    assert cloned.weight.input_dim == parameter.input_dim
    assert cloned.weight.weight_loader is parameter.weight_loader
