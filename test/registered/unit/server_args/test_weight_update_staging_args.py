import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.validation_hook import validate_weight_update_staging
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _config(**overrides):
    values = {
        "weight_update_staging": None,
        "weight_update_local_checkpoint_dir": None,
        "weight_update_max_compile_group_gb": 8.0,
        "weight_version": "0",
        "weight_cache_mode": "off",
        "delete_ckpt_after_loading": False,
        "cpu_offload_gb": 0,
        "offload_group_size": 0,
        "pp_size": 1,
        "dcp_replicate_q_proj": False,
        "enable_eplb": False,
        "enable_lora": False,
        "lora_paths": None,
        "elastic_ep_backend": None,
        "enable_elastic_expert_backup": False,
        "speculative_algorithm": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestWeightUpdateStagingArgs(CustomTestCase):
    def test_staging_backend_is_closed_over_known_physical_destinations(self):
        with self.assertRaisesRegex(ValueError, "must be 'disk' or 'cpu'"):
            validate_weight_update_staging(
                _config(weight_update_staging="remote-memory")
            )

    def test_local_checkpoint_directory_requires_staging(self):
        with self.assertRaisesRegex(ValueError, "requires --weight-update-staging"):
            validate_weight_update_staging(
                _config(weight_update_local_checkpoint_dir="/local")
            )

    def test_disk_staging_requires_a_local_checkpoint_directory(self):
        with self.assertRaisesRegex(ValueError, "requires.*local-checkpoint-dir"):
            validate_weight_update_staging(_config(weight_update_staging="disk"))

    def test_staging_requires_an_integer_initial_version(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            validate_weight_update_staging(
                _config(
                    weight_update_staging="disk",
                    weight_update_local_checkpoint_dir="/local",
                    weight_version="release-candidate",
                )
            )

    def test_cpu_staging_accepts_target_only_speculative_decoding(self):
        with patch(
            "sglang.srt.arg_groups.validation_hook.get_platform",
            return_value=SimpleNamespace(is_cuda=True),
        ):
            validate_weight_update_staging(
                _config(
                    weight_update_staging="cpu",
                    speculative_algorithm="EAGLE",
                )
            )

    def test_cpu_staging_rejects_mutable_or_nonresident_layouts(self):
        incompatible = {
            "cpu_offload_gb": 1,
            "offload_group_size": 1,
            "pp_size": 2,
            "dcp_replicate_q_proj": True,
            "enable_eplb": True,
            "enable_lora": True,
            "lora_paths": ["adapter"],
            "elastic_ep_backend": "deepep",
            "enable_elastic_expert_backup": True,
        }
        with patch(
            "sglang.srt.arg_groups.validation_hook.get_platform",
            return_value=SimpleNamespace(is_cuda=True),
        ):
            for name, value in incompatible.items():
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate_weight_update_staging(
                        _config(weight_update_staging="cpu", **{name: value})
                    )


if __name__ == "__main__":
    unittest.main()
