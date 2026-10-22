"""Unit tests for migrated ServerArgs CLI metadata."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestServerArgsMigratedCliMetadata(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def test_data_parallel_aliases_keep_old_usage(self):
        for option in ("--data-parallel-size", "--dp-size"):
            with self.subTest(option=option):
                args = self.parser.parse_args(["--model", "dummy", option, "3"])
                self.assertEqual(args.dp_size, 3)
                self.assertEqual(ServerArgs.from_cli_args(args).dp_size, 3)

    def test_request_chat_template_requires_explicit_opt_in(self):
        for flags, expected in (([], False), (["--trust-request-chat-template"], True)):
            with self.subTest(flags=flags):
                args = self.parser.parse_args(["--model", "dummy", *flags])
                self.assertIs(
                    ServerArgs.from_cli_args(args).trust_request_chat_template, expected
                )

    def test_dflash_swa_cache_is_opt_in_even_with_hicache(self):
        for flags, expected in (
            ([], False),
            (["--enable-hierarchical-cache"], False),
            (
                [
                    "--enable-dflash-swa-cache",
                    "--enable-hierarchical-cache",
                    "--enable-unified-memory",
                ],
                True,
            ),
        ):
            with self.subTest(flags=flags):
                args = self.parser.parse_args(["--model", "dummy", *flags])
                self.assertIs(
                    ServerArgs.from_cli_args(args).enable_dflash_swa_cache, expected
                )

    def test_dflash_cache_flags_validate_dependencies(self):
        kwargs = dict(
            model_path="dummy",
            device="cuda",
            speculative_algorithm="dflash",
            speculative_draft_model_path="draft",
            speculative_num_draft_tokens=16,
            speculative_draft_attention_backend="fa3",
            max_running_requests=32,
            enable_dflash_swa_cache=True,
            enable_unified_memory=True,
            enable_hierarchical_cache=True,
        )
        # Only model-config I/O is external; exercise real alias resolution and
        # speculative validation, including a supported configuration.
        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=SimpleNamespace(architectures=["DFlash2DraftModel"]),
        ):
            args = ServerArgs(**kwargs)
            handle_speculative_decoding(args)
            self.assertEqual(resolution_result(args, "speculative_algorithm"), "DFLASH")
            for hicache in (False, True):
                for taps in (False, True):
                    handle_speculative_decoding(
                        ServerArgs(
                            **(
                                kwargs
                                | {
                                    "enable_hierarchical_cache": hicache,
                                    "enable_dflash_tap_cache": taps,
                                }
                            )
                        )
                    )
            for overrides, error in (
                ({"speculative_algorithm": None}, "--speculative-algorithm DFLASH"),
                ({"speculative_algorithm": "EAGLE3"}, "--speculative-algorithm DFLASH"),
                ({"enable_unified_memory": False}, "--enable-unified-memory"),
                (
                    {"enable_dflash_swa_cache": False, "enable_dflash_tap_cache": True},
                    "--enable-dflash-swa-cache",
                ),
            ):
                with (
                    self.subTest(overrides=overrides),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    handle_speculative_decoding(ServerArgs(**(kwargs | overrides)))

    def test_prefill_max_context_accepts_human_readable_values(self):
        for option in (
            "--cuda-graph-prefill-max-context",
            "--context-bucket",
        ):
            with self.subTest(option=option):
                args = self.parser.parse_args(["--model", "dummy", option, "200k"])

                self.assertEqual(
                    ServerArgs.from_cli_args(args).cuda_graph_prefill_max_context,
                    200_000,
                )

    def test_migrated_and_manual_options_parse_together(self):
        args = self.parser.parse_args(
            [
                "--model",
                "dummy",
                "--dtype",
                "bfloat16",
                "--max-total-tokens",
                "1024",
                "--prefill-delayer-forward-passes-buckets",
                "1.5",
                "2.5",
                "--data-parallel-size",
                "2",
                "--load-balance-method",
                "total_tokens",
                "--tp-size",
                "4",
            ]
        )
        server_args = ServerArgs.from_cli_args(args)

        self.assertEqual(server_args.dtype, "bfloat16")
        self.assertEqual(server_args.max_total_tokens, 1024)
        self.assertEqual(server_args.prefill_delayer_forward_passes_buckets, [1.5, 2.5])
        self.assertEqual(server_args.dp_size, 2)
        self.assertEqual(server_args.load_balance_method, "total_tokens")
        self.assertEqual(server_args.tp_size, 4)


if __name__ == "__main__":
    unittest.main()
