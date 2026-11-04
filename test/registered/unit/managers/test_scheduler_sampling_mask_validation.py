"""Reject PD sampling-mask requests before entering transfer queues."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative import spec_sampling_mask
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSchedulerSamplingMaskValidation(CustomTestCase):
    def test_disabled_pd_masks_return_bad_request_before_admission(self):
        """Neither PD role may queue mask requests without metadata buffers."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_session_radix_cache = False
                scheduler.model_config = SimpleNamespace(
                    hf_eos_token_id={1}, vocab_size=128
                )
                scheduler.disaggregation_mode = mode
                scheduler.disagg_metadata_buffers = SimpleNamespace(
                    enable_sampling_mask=False
                )
                scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
                scheduler.tokenizer = None
                scheduler.dllm_config = None
                scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
                scheduler._record_kv_event_lora_name = MagicMock()
                scheduler.spec_algorithm = SimpleNamespace(
                    is_dflash_family=lambda: False,
                    is_uno=lambda: False,
                )
                scheduler._add_request_to_queue = MagicMock()
                scheduler.output_streamer = MagicMock()
                recv_req = MagicMock(
                    session_params=None,
                    session_id=None,
                    input_embeds=None,
                    bootstrap_port=1,
                    bootstrap_room=9,
                )
                req = MagicMock(return_sampling_mask=True, return_logprob=False)
                with (
                    patch(
                        "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                        return_value=1,
                    ),
                    patch("sglang.srt.managers.scheduler.Req", return_value=req),
                    patch("sglang.srt.managers.scheduler.prepare_abort") as abort,
                ):
                    scheduler.handle_generate_request(recv_req)
                abort.assert_called_once()
                self.assertEqual(
                    abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                )
                scheduler._add_request_to_queue.assert_not_called()


class TestSpeculativeSamplingCaptureValidation(CustomTestCase):
    """Both the mask and the sampling logprobs alone need committed tokens that are
    exact samples of the captured target distribution."""

    def _validate(self, algorithm, *, min_p=0.0, thresholds=(1.0, 1.0)):
        req = SimpleNamespace(sampling_params=SimpleNamespace(min_p=min_p, top_k=10))
        spec = SimpleNamespace(
            speculative_accept_threshold_single=thresholds[0],
            speculative_accept_threshold_acc=thresholds[1],
        )
        with (
            patch.object(spec_sampling_mask, "get_spec", return_value=spec),
            patch.object(
                spec_sampling_mask,
                "is_dflash_sampling_verify_available",
                return_value=True,
            ),
        ):
            return spec_sampling_mask.validate_spec_sampling_mask_request(
                req, algorithm
            )

    def test_only_the_dflash_family_captures_the_target_distribution(self):
        for algorithm in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK):
            with self.subTest(algorithm=algorithm):
                self.assertIsNone(self._validate(algorithm))
        for algorithm in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3):
            with self.subTest(algorithm=algorithm):
                self.assertIn("outside the DFlash family", self._validate(algorithm))

    def test_dflash_capture_requires_exact_target_samples(self):
        self.assertIn("min_p", self._validate(SpeculativeAlgorithm.DFLASH, min_p=0.1))
        self.assertIn(
            "acceptance thresholds of 1.0",
            self._validate(SpeculativeAlgorithm.DFLASH, thresholds=(0.9, 1.0)),
        )


if __name__ == "__main__":
    unittest.main()
