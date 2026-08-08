from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.dp_attention import get_dp_local_slice_cpu
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _forward_batch(mode: ForwardMode):
    return SimpleNamespace(
        forward_mode=mode,
        global_num_tokens_cpu=[5, 7, 11, 13],
    )


@patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=2)
def test_dp_local_slice_uses_decode_graph_stride(_):
    assert get_dp_local_slice_cpu(
        _forward_batch(ForwardMode.DECODE),
        can_run_graph=True,
        cuda_graph_batch=16,
    ) == (32, 11)


@patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=2)
def test_dp_local_slice_uses_dense_prefill_graph_layout(_):
    assert get_dp_local_slice_cpu(
        _forward_batch(ForwardMode.EXTEND),
        can_run_graph=True,
        cuda_graph_batch=None,
    ) == (12, 11)


@patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=2)
def test_dp_local_slice_ignores_stale_graph_stride_for_eager_forward(_):
    assert get_dp_local_slice_cpu(
        _forward_batch(ForwardMode.EXTEND),
        can_run_graph=False,
        cuda_graph_batch=16,
    ) == (12, 11)
