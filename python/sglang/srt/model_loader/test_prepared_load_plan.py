"""CPU tests for dense prepared checkpoint dispatch replay."""

import concurrent.futures
import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn


def _load_module():
    def default_weight_loader(parameter, loaded_weight):
        parameter.data.copy_(loaded_weight)

    utils_stub = types.ModuleType("sglang.srt.model_loader.utils")
    utils_stub.should_async_load = lambda tensor: tensor.device.type == "cpu"
    weight_utils_stub = types.ModuleType("sglang.srt.model_loader.weight_utils")
    weight_utils_stub.default_weight_loader = default_weight_loader
    for name, module in (
        ("sglang", types.ModuleType("sglang")),
        ("sglang.srt", types.ModuleType("sglang.srt")),
        ("sglang.srt.model_loader", types.ModuleType("sglang.srt.model_loader")),
        ("sglang.srt.model_loader.utils", utils_stub),
        ("sglang.srt.model_loader.weight_utils", weight_utils_stub),
    ):
        sys.modules.setdefault(name, module)
    sys.modules["sglang.srt.model_loader.utils"] = utils_stub
    sys.modules["sglang.srt.model_loader.weight_utils"] = weight_utils_stub

    path = Path(__file__).with_name("prepared_load_plan.py")
    spec = importlib.util.spec_from_file_location(
        "prepared_load_plan_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepared_load_plan = _load_module()


class ToyModel(nn.Module):
    supports_prepared_load_plan = True
    prepared_load_plan_fallback_patterns = ("tail",)

    def __init__(self):
        super().__init__()

        def copy_loader(parameter, loaded_weight):
            parameter.data.copy_(loaded_weight)

        def shard_loader(parameter, loaded_weight, shard_id):
            parameter.data[shard_id].copy_(loaded_weight)

        self.plain = nn.Parameter(torch.zeros(8), requires_grad=False)
        self.plain.weight_loader = copy_loader
        self.stacked = nn.Parameter(torch.zeros(2, 8), requires_grad=False)
        self.stacked.weight_loader = shard_loader
        self.fused = nn.Parameter(torch.zeros(2, 8), requires_grad=False)
        self.fused.weight_loader = copy_loader
        self.tail = nn.Parameter(torch.zeros(8), requires_grad=False)
        self.tail.weight_loader = copy_loader
        self.norm = nn.Parameter(torch.zeros(8), requires_grad=False)
        self.tail_seen = []
        self.consumed = []

    def load_weights(self, weights):
        buffered = {}
        loaded_names = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for name, loaded_weight in weights:
                self.consumed.append(name)
                loaded_names.append(name)
                if name.startswith("stacked."):
                    shard_id = int(name.split(".")[1])
                    self.stacked.weight_loader(
                        self.stacked,
                        loaded_weight,
                        shard_id,
                    )
                elif name.startswith("part_"):
                    buffered[name] = loaded_weight
                    if len(buffered) == 2:
                        derived = torch.stack(
                            [buffered["part_a"], buffered["part_b"]]
                        )
                        futures.append(
                            executor.submit(
                                self.fused.weight_loader,
                                self.fused,
                                derived,
                            )
                        )
                else:
                    parameter = dict(self.named_parameters())[name]
                    loader = getattr(parameter, "weight_loader", None)
                    if loader is None:
                        from sglang.srt.model_loader.weight_utils import (
                            default_weight_loader,
                        )

                        default_weight_loader(parameter, loaded_weight)
                    else:
                        loader(parameter, loaded_weight)
            for future in futures:
                future.result()
        self.tail_seen.extend(name for name in loaded_names if name == "tail")


def _weights(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "plain": torch.randn(8, generator=generator),
        "stacked.0": torch.randn(8, generator=generator),
        "stacked.1": torch.randn(8, generator=generator),
        "part_a": torch.randn(8, generator=generator),
        "part_b": torch.randn(8, generator=generator),
        "tail": torch.randn(8, generator=generator),
        "norm": torch.randn(8, generator=generator),
    }


def test_record_classifies_direct_derived_and_forced_fallback():
    model = ToyModel()
    plan = prepared_load_plan.get_or_create_prepared_load_plan(model)
    stats = plan.record(model, _weights(1).items())

    assert stats["seen"] == 7
    assert "plain" in plan.entries
    assert "stacked.0" in plan.entries
    assert "norm" in plan.entries
    assert "part_a" in plan.fallback
    assert "part_b" in plan.fallback
    assert "tail" in plan.fallback
    assert set(plan.direct_copies) == {
        "plain",
        "stacked.0",
        "stacked.1",
        "norm",
    }
    assert stats["direct_copy_entries"] == 4
    assert stats["direct_copy_operations"] == 4
    assert not hasattr(model.norm, "weight_loader")
    assert set(plan.fully_overwritten_parameters) == {
        "plain",
        "stacked",
        "fused",
        "tail",
        "norm",
    }
    assert plan.fully_overwritten_parameter_names(model) == {
        "plain",
        "stacked",
        "fused",
        "tail",
        "norm",
    }


def test_record_does_not_elide_partially_written_storage():
    class PartialModel(nn.Module):
        supports_prepared_load_plan = True

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(8), requires_grad=False)

            def partial_loader(parameter, loaded_weight):
                parameter.data[:4].copy_(loaded_weight)

            self.weight.weight_loader = partial_loader

        def load_weights(self, weights):
            for _, loaded_weight in weights:
                self.weight.weight_loader(self.weight, loaded_weight)

    model = PartialModel()
    plan = prepared_load_plan.get_or_create_prepared_load_plan(model)
    stats = plan.record(model, [("weight", torch.ones(4))])

    assert stats["fully_overwritten_parameters"] == 0
    assert plan.fully_overwritten_parameter_names(model) == set()


def test_record_keeps_transformed_source_on_loader_fallback():
    class TransformedModel(nn.Module):
        supports_prepared_load_plan = True

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(8), requires_grad=False)

            def transformed_loader(parameter, loaded_weight):
                parameter.data.copy_(loaded_weight * 2)

            self.weight.weight_loader = transformed_loader

        def load_weights(self, weights):
            for _, loaded_weight in weights:
                self.weight.weight_loader(self.weight, loaded_weight)

    recorded = TransformedModel()
    plan = prepared_load_plan.get_or_create_prepared_load_plan(recorded)
    plan.record(recorded, [("weight", torch.ones(8))])
    assert "weight" in plan.entries
    assert "weight" not in plan.direct_copies

    replayed = TransformedModel()
    replayed._prepared_load_plan = plan
    stats = plan.replay(replayed, [("weight", torch.full((8,), 3.0))], max_workers=2)

    torch.testing.assert_close(replayed.weight, torch.full((8,), 6.0))
    assert stats["worker_direct_entries"] == 0


def test_record_keeps_mutated_source_on_loader_fallback():
    class MutatingModel(nn.Module):
        supports_prepared_load_plan = True

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(8), requires_grad=False)

            def mutating_loader(parameter, loaded_weight):
                loaded_weight.mul_(2)
                parameter.data.copy_(loaded_weight)

            self.weight.weight_loader = mutating_loader

        def load_weights(self, weights):
            for _, loaded_weight in weights:
                self.weight.weight_loader(self.weight, loaded_weight)

    recorded = MutatingModel()
    plan = prepared_load_plan.get_or_create_prepared_load_plan(recorded)
    plan.record(recorded, [("weight", torch.ones(8))])
    assert "weight" in plan.entries
    assert "weight" not in plan.direct_copies

    replayed = MutatingModel()
    replayed._prepared_load_plan = plan
    stats = plan.replay(
        replayed,
        [("weight", torch.full((8,), 3.0))],
        max_workers=2,
    )

    torch.testing.assert_close(replayed.weight, torch.full((8,), 6.0))
    assert stats["worker_direct_entries"] == 0


def test_direct_copy_resolves_parameter_in_shared_image_storage():
    recorded = ToyModel()
    plan = prepared_load_plan.get_or_create_prepared_load_plan(recorded)
    plan.record(recorded, _weights(1).items())

    backing = torch.empty(256, dtype=torch.uint8)
    logical_byte_offset = 128
    view = torch.empty(0, dtype=torch.float32).set_(
        backing.untyped_storage(),
        logical_byte_offset // torch.tensor([], dtype=torch.float32).element_size(),
        (8,),
        (1,),
    )
    parameter = nn.Parameter(view, requires_grad=False)
    setattr(
        parameter,
        "_sglang_prepared_logical_storage_range",
        (logical_byte_offset, 8 * parameter.element_size()),
    )
    loaded_weight = torch.arange(8, dtype=torch.float32)

    resolved = plan._resolve_direct_copies(
        {"plain": parameter},
        loaded_weight,
        plan.direct_copies["plain"],
    )

    assert resolved is not None
    destinations = [item[0] for item in resolved]
    sources = [item[1] for item in resolved]
    torch._foreach_copy_(destinations, sources)
    torch.testing.assert_close(parameter, loaded_weight)


def test_replay_consumes_full_checkpoint_and_matches_ordinary_load():
    recorded = ToyModel()
    plan = prepared_load_plan.get_or_create_prepared_load_plan(recorded)
    plan.record(recorded, _weights(1).items())

    target_weights = _weights(2)
    expected = ToyModel()
    expected.load_weights(target_weights.items())

    replayed = ToyModel()
    replayed._prepared_load_plan = plan
    stats = plan.replay(replayed, target_weights.items(), max_workers=4)

    for name, parameter in expected.named_parameters():
        torch.testing.assert_close(
            parameter,
            dict(replayed.named_parameters())[name],
        )
    assert set(replayed.consumed) == {"part_a", "part_b", "tail"}
    assert replayed.tail_seen == ["tail"]
    assert stats["hits"] == 4
    assert stats["fallback"] == 3
    assert stats["unknown"] == 0
    assert stats["source_tensors"] == 7
    assert stats["source_bytes"] == 7 * 8 * 4
    assert stats["direct_source_bytes"] == 4 * 8 * 4
    assert stats["fallback_source_bytes"] == 3 * 8 * 4
    assert stats["worker_calls"] == 4
    assert stats["worker_bytes"] == 4 * 8 * 4
    assert stats["worker_direct_entries"] == 4
    assert stats["worker_direct_operations"] == 4
    assert stats["worker_foreach_calls"] >= 1
    assert stats["worker_foreach_fallback_calls"] == 0
    assert stats["submitted_batches"] >= 1
    assert stats["source_next_s"] >= 0
    assert stats["source_next_cpu_s"] >= 0
    assert stats["source_next_minor_faults"] >= 0
    assert stats["source_next_major_faults"] >= 0
    assert stats["worker_minor_faults"] >= 0
    assert stats["worker_major_faults"] >= 0
    assert stats["drain_wait_s"] >= 0


def test_unsupported_model_uses_ordinary_loader():
    model = nn.Linear(2, 2)
    assert prepared_load_plan.get_or_create_prepared_load_plan(model) is None
