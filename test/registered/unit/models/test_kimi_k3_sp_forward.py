import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import layernorm_sp
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.prefill_graph import prefill_graph_scope
from sglang.srt.models import kimi_k3
from sglang.srt.runtime_context import get_flags, get_forward, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TP = 8
HIDDEN = 4


def kimi_k3_with_stack_rows(rows):
    """Kimi-K3 whose stack returns ``rows`` rows, like a replayed prefill graph's body."""
    model = kimi_k3.KimiK3LinearForCausalLM.__new__(kimi_k3.KimiK3LinearForCausalLM)
    nn.Module.__init__(model)
    model.model = lambda *args, **kwargs: torch.zeros(rows, HIDDEN)
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.capture_aux_hidden_states = False
    model.lm_head = None
    model.logits_processor = lambda input_ids, hidden_states, *args: hidden_states
    return model


class TestKimiK3SPForward(CustomTestCase):
    def setUp(self):
        parallel = SimpleNamespace(tp_group=SimpleNamespace(world_size=TP))
        for patcher in (
            patch.object(kimi_k3, "get_parallel", return_value=parallel),
            patch.object(
                layernorm_sp,
                "sp_exit_gather",
                side_effect=lambda rows, n: rows.repeat(TP, 1)[:n],
            ),
            get_flags().sp.override(
                enabled=True,
                model_collectives=True,
                min_tokens=4097,
                graph_min_tokens=1024,
            ),
            prefill_graph_scope(full_graph=False),
        ):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)
        self.addCleanup(reset_context)

    def forward(self, num_tokens, stack_rows):
        return kimi_k3_with_stack_rows(stack_rows).forward(
            torch.zeros(num_tokens, dtype=torch.long),
            None,
            SimpleNamespace(forward_mode=ForwardMode.EXTEND),
        )

    def test_replayed_sp_bucket_gathers_while_the_stack_flag_reads_off(self):
        """A replayed SP bucket's rows are gathered though its stack set no flag."""
        get_forward().set("sp_active", False)
        self.assertEqual(self.forward(2048, 2048 // TP).shape[0], 2048)

    def test_bucket_below_the_graph_minimum_keeps_the_stack_rows(self):
        """A bucket below the graph minimum reads the stack's rows as they are."""
        self.assertEqual(self.forward(1000, 1000).shape[0], 1000)

    def test_stack_disagreeing_with_the_forward_raises(self):
        """A full-row stack output for an SP-sized forward fails loud."""
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            self.forward(2048, 2048)


if __name__ == "__main__":
    unittest.main()
