"""CPU unit tests for model_loader/load_plan.py.

Covers the invariants the implementation depends on (see the module
docstring): attribution classes (recorded entries, default-loaded params,
inline-derived calls, executor-thread fused aliases, inert names), fallback
streaming preserving name-gated post-load tails, replay equivalence, and
partial reloads with fusion + expert closures.

The module under test only needs torch + safetensors; it is loaded directly
so the tests run without the full sglang import chain (and therefore in any
CPU environment).
"""

import concurrent.futures
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn


def _load_module():
    """Import load_plan.py with its two light sglang deps stubbed, so the
    test needs neither GPUs nor the heavy sglang import chain."""
    if "load_plan_under_test" in sys.modules:
        return sys.modules["load_plan_under_test"]

    def default_weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    utils_stub = types.ModuleType("sglang.srt.model_loader.utils")
    utils_stub.should_async_load = lambda t: t.device.type == "cpu"
    weight_utils_stub = types.ModuleType("sglang.srt.model_loader.weight_utils")
    weight_utils_stub.default_weight_loader = default_weight_loader
    for name, mod in (
        ("sglang", types.ModuleType("sglang")),
        ("sglang.srt", types.ModuleType("sglang.srt")),
        ("sglang.srt.model_loader", types.ModuleType("sglang.srt.model_loader")),
        ("sglang.srt.model_loader.utils", utils_stub),
        ("sglang.srt.model_loader.weight_utils", weight_utils_stub),
    ):
        sys.modules.setdefault(name, mod)
    sys.modules["sglang.srt.model_loader.utils"] = utils_stub
    sys.modules["sglang.srt.model_loader.weight_utils"] = weight_utils_stub

    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/model_loader/load_plan.py"
    )
    spec = importlib.util.spec_from_file_location("load_plan_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["load_plan_under_test"] = module
    spec.loader.exec_module(module)
    return module


load_plan = _load_module()

NUM_EXPERTS = 4
DIM = 8


class ToyModel(nn.Module):
    """Exercises every attribution class the recorder must handle:

    - ``plain.weight``: ordinary param with a weight_loader attribute
    - ``qkv.weight``: stacked param, three checkpoint names with shard args
    - ``experts.weight``: per-expert rows loaded with expert_id kwargs
    - ``norm.weight``: loader-less param (default_weight_loader path)
    - ``fused_ab.weight``: built by buffering ``part_a`` and cat-ing on
      ``part_b``, dispatched on an executor thread (tag-less, like DeepSeek)
    - ``tail_gated.weight``: streamed via fallback pattern; the model's
      post-load tail records which names it saw
    - ``"skipped.weight"``: consumed and dropped (inert)
    """

    supports_load_plan_replay = True
    load_plan_fallback_patterns = ("tail_gated",)
    load_plan_fused_aliases = (
        ("part_a", "fused_ab"),
        ("part_b", "fused_ab"),
    )

    def __init__(self):
        super().__init__()
        def loader(param, w):
            param.data.copy_(w)

        def shard_loader(param, w, shard_id):
            param.data[shard_id * DIM : (shard_id + 1) * DIM].copy_(w)

        def expert_loader(param, w, name, shard_id=None, expert_id=None):
            param.data[expert_id].copy_(w)

        self.plain = nn.Linear(DIM, DIM, bias=False)
        self.plain.weight.weight_loader = loader
        self.qkv = nn.Parameter(torch.zeros(3 * DIM, DIM), requires_grad=False)
        self.qkv.weight_loader = shard_loader
        self.experts = nn.ParameterDict(
            {"weight": nn.Parameter(torch.zeros(NUM_EXPERTS, DIM), requires_grad=False)}
        )
        self.experts["weight"].weight_loader = expert_loader
        self.scales = nn.ParameterDict(
            {"weight": nn.Parameter(torch.zeros(NUM_EXPERTS, DIM), requires_grad=False)}
        )
        self.scales["weight"].weight_loader = expert_loader
        self.norm = nn.LayerNorm(DIM)  # loader-less params
        # ParameterDicts give realistic fqns ("fused_ab.weight"), which the
        # declared-alias rewrites depend on.
        self.fused_ab = nn.ParameterDict(
            {"weight": nn.Parameter(torch.zeros(2 * DIM, DIM), requires_grad=False)}
        )
        self.fused_ab["weight"].weight_loader = loader
        self.tail_gated = nn.ParameterDict(
            {"weight": nn.Parameter(torch.zeros(DIM), requires_grad=False)}
        )
        self.tail_gated["weight"].weight_loader = loader
        self.tail_seen: list = []

    def load_weights(self, weights):
        cached = {}
        loaded_names = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for name, w in weights:
                loaded_names.append(name)
                if name == "skipped.weight":
                    continue  # inert: consumed, no param effect
                if name in ("part_a.weight", "part_b.weight"):
                    cached[name] = w
                    if len(cached) == 2:
                        fused = torch.cat(
                            [cached["part_a.weight"], cached["part_b.weight"]]
                        )
                        # tag-less + worker thread, like DeepSeek's fusion
                        param = self.fused_ab["weight"]
                        futures.append(
                            executor.submit(param.weight_loader, param, fused)
                        )
                        cached.clear()
                    continue
                if name.startswith("qkv."):
                    shard = int(name.split(".")[1])
                    self.qkv.weight_loader(self.qkv, w, shard)
                    continue
                if name.startswith("experts."):
                    expert = int(name.split(".")[1])
                    param = self.experts["weight"]
                    param.weight_loader(param, w, name, expert_id=expert)
                    continue
                if name.startswith("scales."):
                    expert = int(name.split(".")[1])
                    param = self.scales["weight"]
                    param.weight_loader(param, w, name, expert_id=expert)
                    continue
                params = dict(self.named_parameters())
                param = params[name]
                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    from sglang.srt.model_loader.weight_utils import default_weight_loader

                    default_weight_loader(param, w)
                else:
                    loader(param, w)
            for f in futures:
                f.result()
        # name-gated post-load tail, like DeepSeek's post_load_weights
        self.tail_seen.extend(n for n in loaded_names if "tail_gated" in n)


def make_weights():
    g = torch.Generator().manual_seed(7)
    weights = {
        "plain.weight": torch.randn(DIM, DIM, generator=g),
        "norm.weight": torch.randn(DIM, generator=g),
        "norm.bias": torch.randn(DIM, generator=g),
        "part_a.weight": torch.randn(DIM, DIM, generator=g),
        "part_b.weight": torch.randn(DIM, DIM, generator=g),
        "tail_gated.weight": torch.randn(DIM, generator=g),
        "skipped.weight": torch.randn(DIM, generator=g),
    }
    for i in range(3):
        weights[f"qkv.{i}.weight"] = torch.randn(DIM, DIM, generator=g)
    for e in range(NUM_EXPERTS):
        weights[f"experts.{e}.weight"] = torch.randn(DIM, generator=g)
        weights[f"scales.{e}.weight"] = torch.randn(DIM, generator=g)
    return weights


def recorded_model():
    model = ToyModel()
    plan = load_plan.get_or_create_plan(model)
    plan.record(model, iter(make_weights().items()))
    return model, plan


def state_snapshot(model):
    return {k: v.detach().clone() for k, v in model.named_parameters()}


class RecordTest(unittest.TestCase):
    def test_attribution_classes(self):
        model, plan = recorded_model()
        # replayable entries: plain, qkv shards, experts, scales, norm (default-loaded)
        self.assertIn("plain.weight", plan.entries)
        self.assertIn("qkv.1.weight", plan.entries)
        self.assertIn("experts.2.weight", plan.entries)
        self.assertIn("norm.weight", plan.entries)
        # fallback: fused parts (worker-thread, tag-less), tail-gated (pattern), inert
        for name in ("part_a.weight", "part_b.weight", "tail_gated.weight", "skipped.weight"):
            self.assertIn(name, plan.fallback)
            self.assertNotIn(name, plan.entries)
        # tail-gated keeps its effects for module attribution despite fallback
        self.assertEqual(plan.effects["tail_gated.weight"], ["tail_gated.weight"])
        # inline-derived attribution: cat happened on a worker thread -> none;
        # fused parts resolve via declared aliases instead
        fqns = {fqn for fqn, _ in model.named_parameters()}
        self.assertEqual(plan._alias_fqn("part_b.weight", fqns), "fused_ab.weight")
        # expert index knows every tensor of each expert
        self.assertEqual(
            plan.expert_index["experts"][2], {"experts.2.weight"}
        )

    def test_default_loader_wrapper_removed_after_record(self):
        model, _ = recorded_model()
        self.assertFalse(hasattr(model.norm.weight, "weight_loader"))


class ReplayTest(unittest.TestCase):
    def test_replay_matches_fresh_load_and_preserves_tail(self):
        model, plan = recorded_model()
        reference = state_snapshot(model)
        for p in model.parameters():
            p.data.zero_()
        model.tail_seen.clear()
        stats = plan.replay(model, iter(make_weights().items()))
        for name, want in reference.items():
            torch.testing.assert_close(dict(model.named_parameters())[name], want)
        # the name-gated tail saw exactly the fallback-streamed tail names
        self.assertEqual(model.tail_seen, ["tail_gated.weight"])
        self.assertEqual(stats["plan_unknown"], 0)


class PartialTest(unittest.TestCase):
    def _checkpoint_dir(self, tmp):
        import safetensors.torch

        weights = make_weights()
        shard = "model-00001-of-00001.safetensors"
        safetensors.torch.save_file(weights, os.path.join(tmp, shard))
        index = {"weight_map": {name: shard for name in weights}}
        with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        return tmp

    def test_touched_plan_closures_and_inert(self):
        model, plan = recorded_model()
        fqns = {fqn for fqn, _ in model.named_parameters()}
        # expert touch -> whole-expert closure with expert ids in the detail
        detail, names = plan.touched_plan(["experts.1.weight"], fqns)
        self.assertEqual(detail["experts"]["weight"], {1})
        self.assertIn("experts.1.weight", names)
        # fused part touch -> both declared inputs reload
        detail, names = plan.touched_plan(["part_a.weight"], fqns)
        self.assertEqual(set(names), {"part_a.weight", "part_b.weight"})
        self.assertIn("fused_ab", detail)  # module of the fused param
        # inert touch -> skipped, empty plan
        detail, names = plan.touched_plan(["skipped.weight"], fqns)
        self.assertEqual(names, set())
        # unknown touch -> None (caller full-reloads)
        self.assertIsNone(plan.touched_plan(["never.seen"], fqns))

    def test_partial_replay_touches_only_the_touched(self):
        model, plan = recorded_model()
        reference = state_snapshot(model)
        with tempfile.TemporaryDirectory() as tmp:
            self._checkpoint_dir(tmp)
            # poison one touched and one untouched param
            model.experts["weight"].data[1].fill_(-99.0)
            model.plain.weight.data.fill_(-77.0)
            result = plan.partial_replay(model, tmp, ["experts.1.weight"])
            self.assertIsNotNone(result)
            stats, detail = result
            self.assertEqual(stats["plan"], "partial")
            params = dict(model.named_parameters())
            # touched expert restored...
            torch.testing.assert_close(
                params["experts.weight"][1], reference["experts.weight"][1]
            )
            # ...untouched param untouched (still poisoned): partial is O(delta)
            self.assertTrue(torch.all(params["plain.weight"] == -77.0))

    def test_partial_replay_fused_pair(self):
        model, plan = recorded_model()
        reference = state_snapshot(model)
        with tempfile.TemporaryDirectory() as tmp:
            self._checkpoint_dir(tmp)
            model.fused_ab["weight"].data.zero_()
            result = plan.partial_replay(model, tmp, ["part_b.weight"])
            self.assertIsNotNone(result)
            torch.testing.assert_close(
                dict(model.named_parameters())["fused_ab.weight"],
                reference["fused_ab.weight"],
            )

    def test_partial_replay_missing_index_falls_back(self):
        model, plan = recorded_model()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(plan.partial_replay(model, tmp, ["plain.weight"]))


if __name__ == "__main__":
    unittest.main()
