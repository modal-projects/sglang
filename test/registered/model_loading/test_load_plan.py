"""CPU unit tests for model_loader/load_plan.py.

Covers the invariants the implementation depends on (see the module
docstring): attribution classes (recorded entries, default-loaded params,
inline-derived calls, inert names), fallback streaming preserving name-gated
post-load tails, and replay equivalence.

The module under test only needs torch; it is loaded directly
so the tests run without the full sglang import chain (and therefore in any
CPU environment).
"""

import concurrent.futures
import importlib.util
import sys
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
        # A ParameterDict gives a realistic fqn ("fused_ab.weight") for the
        # tag-less worker-thread fusion that must fall back.
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
                    from sglang.srt.model_loader.weight_utils import (
                        default_weight_loader,
                    )

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
        for name in (
            "part_a.weight",
            "part_b.weight",
            "tail_gated.weight",
            "skipped.weight",
        ):
            self.assertIn(name, plan.fallback)
            self.assertNotIn(name, plan.entries)

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


if __name__ == "__main__":
    unittest.main()
