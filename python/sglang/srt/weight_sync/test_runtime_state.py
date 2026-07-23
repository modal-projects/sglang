import concurrent.futures
import gc
import weakref

import torch
from sglang.srt.models.utils import WeightsMapper

from sglang.srt.weight_sync.runtime_state import (
    HostImageArena,
    PreparedRuntimeState,
    RuntimeModuleGroup,
    RuntimeStorageSegment,
    RuntimeStateImage,
    _parallel_cpu_storage_copy,
    build_host_load_proxy,
    build_runtime_module_groups,
    checkpoint_module_path,
    clone_module_proxy,
    clone_module_tensors,
    grouped_mmap_weights_iterator,
    map_checkpoint_names_to_runtime_groups,
    module_at_path,
    ordered_mmap_weights_iterator,
    release_module_tensors,
    replace_proxy_module,
    runtime_group_finalization_order,
    runtime_group_image_ranges,
    runtime_module_path,
    streaming_mmap_weights_iterator,
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
    assert runtime_module_path(
        "language_model.model.layers.17.rotary_emb.inv_freq"
    ) == ("language_model.model.layers.17")


def test_runtime_module_groups_derive_bounded_model_frontier():
    model = _ToyModel()
    groups = build_runtime_module_groups(
        model,
        max_group_bytes=model.language_model.model.layers[0].weight.nbytes,
        device_type="cpu",
    )

    assert groups == [
        RuntimeModuleGroup(
            path=f"language_model.model.layers.{index}",
            nbytes=model.language_model.model.layers[index].weight.nbytes,
        )
        for index in range(3)
    ]


def test_checkpoint_names_map_to_runtime_groups_without_model_patterns():
    model = _ToyModel()
    groups = [
        RuntimeModuleGroup(
            path=f"language_model.model.layers.{index}",
            nbytes=model.language_model.model.layers[index].weight.nbytes,
        )
        for index in range(3)
    ]

    mapping = map_checkpoint_names_to_runtime_groups(
        model,
        [
            "language_model.model.layers.0.weight",
            "language_model.model.layers.2.weight",
        ],
        groups,
    )

    assert mapping == {
        "language_model.model.layers.0.weight": "language_model.model.layers.0",
        "language_model.model.layers.2.weight": "language_model.model.layers.2",
    }


def test_checkpoint_names_use_model_mapper_and_skip_explicit_ignores():
    model = _ToyModel()
    model.hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "unused.": None,
        }
    )
    groups = [
        RuntimeModuleGroup(
            path=f"language_model.model.layers.{index}",
            nbytes=model.language_model.model.layers[index].weight.nbytes,
        )
        for index in range(3)
    ]

    mapping = map_checkpoint_names_to_runtime_groups(
        model,
        [
            "language_model.layers.1.weight",
            "unused.rotary_cache",
        ],
        groups,
    )

    assert mapping == {
        "language_model.layers.1.weight": "language_model.model.layers.1",
        "unused.rotary_cache": None,
    }


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


def test_clone_module_tensors_can_copy_to_explicit_device():
    module = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.arange(8.0))
    module.register_parameter("weight", parameter)
    module.alias = parameter[2:6]

    cloned = clone_module_tensors(
        module,
        target_device=torch.device("cpu"),
    )

    assert cloned.weight.device.type == "cpu"
    assert cloned.weight.data_ptr() != module.weight.data_ptr()
    assert cloned.alias.untyped_storage().data_ptr() == cloned.weight.data_ptr()
    assert torch.equal(cloned.weight, module.weight)


def test_clone_module_tensors_can_defer_unique_storage_copies():
    module = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.arange(8.0))
    module.register_parameter("weight", parameter)
    module.alias = parameter[2:6]
    pending = []

    def defer(destination, source, non_blocking):
        destination.fill_(255)
        pending.append((destination, source, non_blocking))

    cloned = clone_module_tensors(
        module,
        target_device=torch.device("cpu"),
        storage_copier=defer,
    )

    assert len(pending) == 1
    assert pending[0][2] is False
    assert not torch.equal(cloned.weight, module.weight)
    pending[0][0].copy_(pending[0][1])
    assert torch.equal(cloned.weight, module.weight)
    assert cloned.alias.untyped_storage().data_ptr() == cloned.weight.data_ptr()


def test_parallel_cpu_storage_copy_splits_and_copies_ranges():
    source = torch.arange(2 * 1024 * 1024, dtype=torch.int64).view(torch.uint8)
    destination = torch.empty_like(source)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        wall_s = _parallel_cpu_storage_copy(
            [(destination, source)],
            executor=executor,
            max_workers=4,
            chunk_bytes=1 << 20,
        )

    assert wall_s >= 0
    assert torch.equal(destination, source)


def test_release_module_tensors_allows_young_parameter_cycles_to_collect():
    module = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.arange(8.0))
    parameter.retained_cycle = parameter
    module.register_parameter("weight", parameter)
    parameter_ref = weakref.ref(parameter)
    del parameter

    release_module_tensors(module)
    gc.collect(0)

    assert parameter_ref() is None


def test_clone_module_tensors_uses_disjoint_backing_ranges_and_reclones():
    module = torch.nn.Module()
    module.register_parameter(
        "first",
        torch.nn.Parameter(torch.arange(8.0)),
    )
    module.register_parameter(
        "second",
        torch.nn.Parameter(torch.arange(8.0) + 20),
    )
    module.alias = module.first[2:6]
    backing = torch.empty(3 * 4096, dtype=torch.uint8)
    arena = HostImageArena.from_range(backing, 0, backing.numel())

    cloned = clone_module_tensors(
        module,
        target_device=torch.device("cpu"),
        storage_allocator=arena.allocate,
    )
    recloned = clone_module_tensors(
        cloned,
        target_device=torch.device("cpu"),
    )

    assert cloned.first.data_ptr() == backing.data_ptr()
    assert cloned.second.data_ptr() == backing.data_ptr() + 4096
    assert cloned.first.untyped_storage().data_ptr() == backing.data_ptr()
    assert cloned.second.untyped_storage().data_ptr() == backing.data_ptr()
    assert cloned.alias.data_ptr() == cloned.first[2:].data_ptr()
    assert recloned.first.untyped_storage().data_ptr() != (
        recloned.second.untyped_storage().data_ptr()
    )
    assert torch.equal(recloned.first, module.first)
    assert torch.equal(recloned.second, module.second)
    assert torch.equal(recloned.alias, module.alias)


def test_runtime_group_image_ranges_include_alignment_gaps():
    tensor = torch.empty(1, dtype=torch.uint8)
    segments = [
        RuntimeStorageSegment("left.a", 0, 4, tensor),
        RuntimeStorageSegment("left.b", 4096, 8, tensor),
        RuntimeStorageSegment("right.a", 8192, 16, tensor),
    ]
    groups = [
        RuntimeModuleGroup("left", 12),
        RuntimeModuleGroup("right", 16),
    ]

    assert runtime_group_image_ranges(segments, groups) == {
        "left": (0, 4104),
        "right": (8192, 8208),
    }


def test_runtime_group_finalization_order_protects_larger_final_prefix():
    tensor = torch.empty(1, dtype=torch.uint8)
    groups = [
        RuntimeModuleGroup("first", 8),
        RuntimeModuleGroup("second", 8),
    ]
    segments = [
        RuntimeStorageSegment("first.weight", 0, 12, tensor),
        RuntimeStorageSegment("second.weight", 12, 4, tensor),
    ]
    raw_ranges = {
        "first": (0, 8),
        "second": (8, 16),
    }

    assert runtime_group_finalization_order(groups, segments, raw_ranges) == [
        "second",
        "first",
    ]


def test_host_load_proxy_owns_complete_group_frontier():
    model = _ToyModel()
    groups = build_runtime_module_groups(
        model,
        max_group_bytes=model.language_model.model.layers[0].weight.nbytes,
        device_type="cpu",
    )

    proxy, stats, raw_image_ranges = build_host_load_proxy(
        model,
        groups,
        torch.device("cpu"),
        lambda module, device: None,
    )

    assert stats["groups"] == 3
    assert raw_image_ranges == {}
    for group in groups:
        proxy_group = module_at_path(proxy, group.path)
        live_group = module_at_path(model, group.path)
        assert proxy_group is not live_group
        assert proxy_group.weight.data_ptr() != live_group.weight.data_ptr()

    replacement = torch.nn.Linear(4, 4, bias=False)
    replace_proxy_module(
        proxy,
        model,
        groups[1].path,
        replacement,
    )
    assert module_at_path(proxy, groups[1].path) is replacement
    assert module_at_path(model, groups[1].path) is not replacement


def test_host_load_proxy_uses_runtime_image_as_direct_cpu_backing():
    model = _ToyModel()
    groups = build_runtime_module_groups(
        model,
        max_group_bytes=model.language_model.model.layers[0].weight.nbytes,
        device_type="cpu",
    )
    segments = [
        RuntimeStorageSegment(
            name=f"{group.path}.weight",
            image_offset=index * 4096,
            nbytes=group.nbytes,
            device_bytes=torch.empty(group.nbytes, dtype=torch.uint8),
        )
        for index, group in enumerate(groups)
    ]
    image = torch.empty(3 * 4096, dtype=torch.uint8)

    proxy, stats, raw_image_ranges = build_host_load_proxy(
        model,
        groups,
        torch.device("cpu"),
        lambda module, device: None,
        image_bytes=image,
        segments=segments,
    )

    assert set(raw_image_ranges) == {group.path for group in groups}
    assert stats["direct_image_groups"] == 3
    assert stats["fallback_groups"] == 0
    for index, group in enumerate(groups):
        proxy_weight = module_at_path(proxy, group.path).weight
        assert proxy_weight.data_ptr() == image.data_ptr() + index * 4096
        assert torch.equal(
            proxy_weight,
            module_at_path(model, group.path).weight,
        )


def test_host_load_proxy_bounds_layout_growth_with_pageable_fallback():
    model = _ToyModel()
    group = RuntimeModuleGroup(
        path="language_model.model.layers.0",
        nbytes=model.language_model.model.layers[0].weight.nbytes,
    )
    segment = RuntimeStorageSegment(
        name=f"{group.path}.weight",
        image_offset=0,
        nbytes=group.nbytes - 1,
        device_bytes=torch.empty(1, dtype=torch.uint8),
    )
    image = torch.empty(group.nbytes - 1, dtype=torch.uint8)

    proxy, stats, raw_image_ranges = build_host_load_proxy(
        model,
        [group],
        torch.device("cpu"),
        lambda module, device: None,
        image_bytes=image,
        segments=[segment],
    )

    assert raw_image_ranges == {}
    assert stats["direct_image_groups"] == 0
    assert stats["fallback_groups"] == 1
    assert module_at_path(proxy, group.path).weight.data_ptr() != image.data_ptr()


def test_release_module_tensors_does_not_mutate_live_module():
    live = torch.nn.Module()
    live.register_parameter("weight", torch.nn.Parameter(torch.arange(8.0)))
    live.aliases = {"slice": live.weight[2:6]}
    cloned = clone_module_tensors(live)

    release_module_tensors(cloned)

    assert cloned.weight is None
    assert cloned.aliases == {"slice": None}
    assert torch.equal(live.weight, torch.arange(8.0))
    assert torch.equal(live.aliases["slice"], torch.arange(2.0, 6.0))


def test_ordered_mmap_weights_iterator_merges_checkpoint_shards(tmp_path):
    import json

    from safetensors.torch import save_file

    save_file(
        {"language_model.model.layers.1.b": torch.tensor([12])},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "language_model.model.layers.0.a": torch.tensor([1]),
            "language_model.model.layers.1.a": torch.tensor([11]),
        },
        tmp_path / "model-00002-of-00002.safetensors",
    )
    index = {
        "weight_map": {
            "language_model.model.layers.1.b": "model-00001-of-00002.safetensors",
            "language_model.model.layers.0.a": "model-00002-of-00002.safetensors",
            "language_model.model.layers.1.a": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    items = list(ordered_mmap_weights_iterator(str(tmp_path)))

    assert [name for name, _ in items] == [
        "language_model.model.layers.0.a",
        "language_model.model.layers.1.a",
        "language_model.model.layers.1.b",
    ]
    assert [tensor.item() for _, tensor in items] == [1, 11, 12]


def test_streaming_mmap_weights_iterator_reads_each_shard_once(tmp_path):
    import json

    from safetensors.torch import save_file

    save_file(
        {"second": torch.tensor([2])},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    save_file(
        {"first": torch.tensor([1])},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    index = {
        "weight_map": {
            "second": "model-00002-of-00002.safetensors",
            "first": "model-00001-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    items = list(streaming_mmap_weights_iterator(str(tmp_path)))

    assert [(name, tensor.item()) for name, tensor in items] == [
        ("first", 1),
        ("second", 2),
    ]


def test_grouped_mmap_weights_iterator_yields_complete_runtime_frontier(tmp_path):
    import json

    from safetensors.torch import save_file

    filename = "model.safetensors"
    save_file(
        {
            "language_model.model.layers.0.weight": torch.tensor([1]),
            "language_model.model.layers.2.weight": torch.tensor([3]),
        },
        tmp_path / filename,
    )
    index = {
        "weight_map": {
            "language_model.model.layers.0.weight": filename,
            "language_model.model.layers.2.weight": filename,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    model = _ToyModel()
    groups = [
        RuntimeModuleGroup(
            path=f"language_model.model.layers.{index}",
            nbytes=model.language_model.model.layers[index].weight.nbytes,
        )
        for index in range(3)
    ]

    grouped = [
        (path, [(name, tensor.item()) for name, tensor in weights])
        for path, weights in grouped_mmap_weights_iterator(
            str(tmp_path),
            model,
            groups,
        )
    ]

    assert grouped == [
        (
            "language_model.model.layers.0",
            [("language_model.model.layers.0.weight", 1)],
        ),
        ("language_model.model.layers.1", []),
        (
            "language_model.model.layers.2",
            [("language_model.model.layers.2.weight", 3)],
        ),
    ]


def test_parallel_memcpy_copies_disjoint_cpu_ranges():
    first = torch.arange(4096, dtype=torch.uint8)
    second = torch.arange(255, -1, -1, dtype=torch.uint8)
    destination = torch.zeros(8192, dtype=torch.uint8)

    elapsed = PreparedRuntimeState._parallel_memcpy(
        destination.data_ptr(),
        [
            (0, first.data_ptr(), first.numel()),
            (5000, second.data_ptr(), second.numel()),
        ],
        max_workers=2,
    )

    assert elapsed >= 0
    assert torch.equal(destination[:4096], first)
    assert torch.equal(destination[5000:5256], second)


def test_allocate_image_consumes_preallocated_buffer():
    state = object.__new__(PreparedRuntimeState)
    state.image_nbytes = 32
    state._full_pinned = True
    preallocated = torch.empty(32, dtype=torch.uint8)
    state._preallocated_image_bytes = preallocated

    image = state.allocate_image("v1")

    assert image.identity == "v1"
    assert image.bytes is preallocated
    assert state._preallocated_image_bytes is None


def test_allocate_image_falls_back_when_full_pin_fails(monkeypatch):
    state = object.__new__(PreparedRuntimeState)
    state.image_nbytes = 32
    state._full_pinned = True
    state._preallocated_image_bytes = None
    real_empty = torch.empty

    def empty(*args, pin_memory=False, **kwargs):
        if pin_memory:
            raise RuntimeError("pin limit")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    image = state.allocate_image("v1")

    assert image.identity == "v1"
    assert image.bytes.numel() == 32
    assert not state._full_pinned


def test_begin_preparation_reuses_rollback_image():
    state = object.__new__(PreparedRuntimeState)
    state.active = None
    state.prepared = RuntimeStateImage(
        bytes=torch.empty(16, dtype=torch.uint8),
        identity="old",
    )
    state._staged_image = object()
    state._staged_tail = [object()]
    state._gpu_stage = object()
    state._gpu_stage_image_offset = 7

    old_bytes = state.prepared.bytes
    image = state.begin_preparation("next")

    assert image.bytes is old_bytes
    assert image.identity == "next"
    assert state._staged_image is None
    assert state._staged_tail == []
    assert state._gpu_stage is None
    assert state._gpu_stage_image_offset == 0


def test_begin_preparation_does_not_overwrite_active_rollback(monkeypatch):
    state = object.__new__(PreparedRuntimeState)
    state.active = RuntimeStateImage(
        bytes=torch.empty(16, dtype=torch.uint8),
        identity="active",
    )
    state.prepared = state.active
    state._staged_image = None
    state._staged_tail = []
    state._gpu_stage = None
    state._gpu_stage_image_offset = 0
    replacement = RuntimeStateImage(
        bytes=torch.empty(16, dtype=torch.uint8),
        identity="next",
    )
    monkeypatch.setattr(state, "allocate_image", lambda identity: replacement)

    image = state.begin_preparation("next")

    assert image is replacement
    assert state.active.identity == "active"


def test_begin_preparation_single_image_reuses_active_snapshot():
    state = object.__new__(PreparedRuntimeState)
    state._single_image = True
    state.active = RuntimeStateImage(
        bytes=torch.empty(16, dtype=torch.uint8),
        identity="active",
    )
    state.prepared = None
    state._staged_image = object()
    state._staged_tail = [object()]
    state._gpu_stage = object()
    state._gpu_stage_image_offset = 7

    active_bytes = state.active.bytes
    image = state.begin_preparation("next")

    assert image.bytes is active_bytes
    assert image.identity == "next"
    assert state.active is None
    assert state._staged_image is None
    assert state._staged_tail == []
    assert state._gpu_stage is None
    assert state._gpu_stage_image_offset == 0
