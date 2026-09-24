"""Terminal outcome counters preserve usage and remain bounded across failures."""

import unittest
from functools import partial

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.observability import metrics_collector
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFinishedOutcomeMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            prompt_tokens_buckets=None, generation_tokens_buckets=None
        )
        self.server_args = override.install()
        self.addCleanup(override.restore)
        self.registry = CollectorRegistry()
        self.labels = {"model_name": "test-model", "priority": "", "workload": ""}

        class Collector(TokenizerMetricsCollector):
            _counter_cls = partial(Counter, registry=self.registry)
            _gauge_cls = partial(Gauge, registry=self.registry)
            _histogram_cls = partial(Histogram, registry=self.registry)

        self.collector_cls = Collector
        self.collector = Collector(server_args=self.server_args, labels=self.labels)

    def sample(self, name, labels):
        return self.registry.get_sample_value("sglang:" + name, labels)

    def test_all_outcomes_visible_before_first_request(self):
        """An outcome's first failure must not create its default-label series."""
        for outcome in (
            "success",
            "abort",
            "rejected",
            "invalid_request",
            "engine_fault",
            "other",
        ):
            for metric in (
                "finished_requests_by_outcome_total",
                "finished_prompt_tokens_by_outcome_total",
                "finished_cached_tokens_by_outcome_total",
            ):
                with self.subTest(outcome=outcome, metric=metric):
                    self.assertEqual(
                        self.sample(metric, {**self.labels, "outcome": outcome}), 0
                    )

    def test_request_labels_seed_all_outcomes_and_preserve_zero_cache(self):
        """Request-specific labels must expose zero-valued sibling outcomes."""
        labels = {**self.labels, "priority": "7", "workload": "interactive"}
        self.collector.observe_finished_outcome(labels, "engine_fault", 12, 0)
        for outcome in (
            "success",
            "abort",
            "rejected",
            "invalid_request",
            "engine_fault",
            "other",
        ):
            with self.subTest(outcome=outcome):
                outcome_labels = {**labels, "outcome": outcome}
                self.assertEqual(
                    self.sample(
                        "finished_cached_tokens_by_outcome_total", outcome_labels
                    ),
                    0,
                )
                self.assertEqual(
                    self.sample("finished_requests_by_outcome_total", outcome_labels),
                    int(outcome == "engine_fault"),
                )
        self.assertEqual(
            self.sample(
                "finished_prompt_tokens_by_outcome_total",
                {**labels, "outcome": "engine_fault"},
            ),
            12,
        )

    def test_label_values_share_the_backend_string_identity(self):
        """Raw JSON values and their string forms must select the same series."""
        labels = {**self.labels, "priority": 7, "workload": {"phases": ["interactive"]}}
        strings = {name: str(value) for name, value in labels.items()}
        self.collector.observe_finished_outcome(labels, "success", 12, 4)
        self.collector.observe_finished_outcome(strings, "success", 12, 4)
        self.assertEqual(
            self.sample(
                "finished_requests_by_outcome_total", {**strings, "outcome": "success"}
            ),
            2,
        )
        self.assertEqual(
            self.sample(
                "finished_prompt_tokens_by_outcome_total",
                {**strings, "outcome": "success"},
            ),
            24,
        )
        self.assertEqual(
            self.sample(
                "finished_requests_by_outcome_total",
                {**strings, "outcome": "engine_fault"},
            ),
            0,
        )

    def test_negative_cache_count_does_not_drop_request(self):
        """A negative cache count must not discard the terminal request count."""
        self.collector.observe_finished_outcome(self.labels, "success", 12, -4)
        outcome_labels = {**self.labels, "outcome": "success"}
        self.assertEqual(
            self.sample("finished_requests_by_outcome_total", outcome_labels), 1
        )
        self.assertEqual(
            self.sample("finished_cached_tokens_by_outcome_total", outcome_labels), 0
        )

    def test_negative_cache_count_does_not_inflate_uncached_usage(self):
        """The uncached prompt histogram must not exceed the request's prompt."""
        self.collector.observe_one_finished_request(
            self.labels, 12, 3, -4, 0.5, False, is_streaming=True
        )
        self.assertEqual(
            self.sample("uncached_prompt_tokens_histogram_sum", self.labels), 12
        )
        stream_labels = {**self.labels, "is_streaming": "true"}
        self.assertEqual(self.sample("prompt_tokens_total", stream_labels), 12)
        self.assertEqual(self.sample("num_requests_total", stream_labels), 1)

    def test_unknown_outcome_cannot_create_unbounded_series(self):
        self.collector.observe_finished_outcome(self.labels, "unexpected detail", 2, 0)
        self.assertEqual(
            self.sample(
                "finished_requests_by_outcome_total",
                {**self.labels, "outcome": "other"},
            ),
            1,
        )
        self.assertIsNone(
            self.sample(
                "finished_requests_by_outcome_total",
                {**self.labels, "outcome": "unexpected detail"},
            )
        )

    def test_outcome_label_collision_fails_before_registration(self):
        with self.assertRaisesRegex(ValueError, "label 'outcome' is reserved"):
            self.collector_cls(
                server_args=self.server_args, labels={"outcome": "custom"}
            )

    def test_finish_reason_edge_cases_remain_bounded(self):
        for reason, outcome in (
            (None, "other"),
            ({"type": "future"}, "other"),
            ({"type": "abort", "status_code": "503"}, "rejected"),
            ({"type": "abort", "status_code": "unknown"}, "abort"),
            ({"type": "abort", "status_code": 200}, "abort"),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(metrics_collector.finished_outcome(reason), outcome)


if __name__ == "__main__":
    unittest.main()
