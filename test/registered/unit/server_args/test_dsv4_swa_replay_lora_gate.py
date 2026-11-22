"""DeepSeek-V4.1 keeps decoder SWA bounded replay off with LoRA.

BUG REGRESSION. The replay runs the late layers over each request's last SWA
window of rows, while LoRA's segments describe every row of the extend, so the
first prefill longer than the window failed check_lora_token_rows (and without
that check the adapter's delta would land on the wrong rows). The replay
resolved on by default, LoRA or not.

    python -m pytest test/registered/unit/server_args/test_dsv4_swa_replay_lora_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_ENV_GATE = "sglang.srt.layers.attention.dsv4.unified_kv_gate"


def _resolved_replay(**fields):
    sa = ServerArgs(
        model_path="local-deepseek-v41",
        attention_backend="dsv4",
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    )
    sa._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41")
    )
    for name, value in fields.items():
        msgspec.Struct.__setattr__(sa, name, value)
    with patch(f"{_ENV_GATE}.is_unified_kv_triton", return_value=False):
        deepseek_v4_hook.validate_deepseek_v41_features(sa)
    return resolving_view(sa).enable_decoder_swa_bounded_replay


class TestDecoderSwaReplayLoraGate(unittest.TestCase):
    @override_platform(is_cuda=True, is_hip=False)
    def test_lora_resolves_the_replay_off(self):
        self.assertFalse(_resolved_replay(enable_lora=True))
        # --lora-paths has not been turned into enable_lora yet at this point
        self.assertFalse(_resolved_replay(enable_lora=None, lora_paths=["a=/tmp/a"]))

    @override_platform(is_cuda=True, is_hip=False)
    def test_without_lora_the_replay_still_resolves_on(self):
        """An always-true LoRA predicate would turn the replay off for every launch."""
        self.assertTrue(_resolved_replay(enable_lora=None, lora_paths=None))

    @override_platform(is_cuda=True, is_hip=False)
    def test_explicit_replay_with_lora_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined with LoRA"):
            _resolved_replay(enable_decoder_swa_bounded_replay=True, enable_lora=True)


if __name__ == "__main__":
    unittest.main()
