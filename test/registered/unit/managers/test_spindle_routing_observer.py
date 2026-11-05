"""Routing snapshots use the scheduler request's CPU prefix-length counter."""

import sys
from array import array
from types import SimpleNamespace

import pytest

from sglang.srt import spindle_routing_observer as observer
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "prefix_len,output_ids,expected_prefill,expected_decode",
    [(0, [], 8, 6), (3, [], 5, 6), (20, [], 0, 6), (3, [42, 43], 0, 4)],
)
def test_snapshot_uses_current_request_prefix_counter(
    monkeypatch, prefix_len, output_ids, expected_prefill, expected_decode
):
    with get_context().override_server_args(model_path="dummy", enable_lora=True):
        req = Req(
            rid="request",
            origin_input_text="prompt",
            origin_input_ids=array("q", range(8)),
            sampling_params=SamplingParams(max_new_tokens=6),
            lora_id="adapter-id",
        )
    assert not hasattr(req, "prefix_indices")
    req.prefix_len = prefix_len
    req.output_ids = array("q", output_ids)
    submissions = []
    publisher = SimpleNamespace(
        known={}, failures=0, ready=lambda cache: True, submit=submissions.append
    )
    tree_core = type("RustUnifiedTreeCore", (), {})()
    tree_core._binding = SimpleNamespace(routing_snapshot_delta=lambda known: (0, []))
    cache = type("UnifiedRadixCache", (), {})()
    cache.tree_core = tree_core
    cache.is_swa_enabled = False
    cache.is_mamba_enabled = True
    manager = SimpleNamespace(
        lora_refs={"adapter-id": SimpleNamespace(lora_name="adapter")},
        loras={"adapter-id": object()},
        memory_pool=SimpleNamespace(uid_to_buffer_id={"adapter-id": 0}),
    )
    scheduler = SimpleNamespace(
        enable_lora=True,
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(lora_manager=manager)),
        tree_cache=cache,
        waiting_queue=[req],
        chunked_req=None,
        pool_stats_observer=SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                full_num_used=2, full_available_size=5, full_evictable_size=3
            )
        ),
    )
    monkeypatch.setattr(observer, "ENABLED", True)
    monkeypatch.setattr(observer, "publisher", publisher)
    monkeypatch.setattr(observer, "last_snapshot", 0)
    monkeypatch.setattr(observer, "sequence", 0)
    monkeypatch.setattr(
        observer,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, dp_size=1, pp_size=1),
    )
    observer.snapshot(scheduler, None, None)
    assert len(submissions) == 1
    state = submissions[0][0]
    assert state["requests"] == [
        {
            "rid": "request",
            "adapter": "adapter",
            "prefill_tokens": expected_prefill,
            "decode_tokens": expected_decode,
        }
    ]
    assert state["kv_used_tokens"] == 2
    assert state["kv_capacity_tokens"] == 10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
