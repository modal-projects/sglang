import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

INKLING_MODEL = os.environ.get("INKLING_MODEL_PATH")
INKLING_MAMBA_CHUNK_SIZE = 64
INKLING_MAMBA_TRACK_INTERVAL = 256


class TestUnifiedInklingHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    kl_threshold = 0.015
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=INKLING_MAMBA_CHUNK_SIZE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=INKLING_MAMBA_TRACK_INTERVAL)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = INKLING_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--trust-remote-code",
                "--fp4-gemm-backend",
                "flashinfer_trtllm",
                "--moe-runner-backend",
                "flashinfer_trtllm_routed",
                "--attention-backend",
                "fa4",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(INKLING_MAMBA_TRACK_INTERVAL),
                "--chunked-prefill-size",
                "8192",
                "--mem-fraction-static",
                "0.85",
                "--swa-full-tokens-ratio",
                "0.25",
                "--mamba-full-memory-ratio",
                "0.1",
                "--reasoning-parser",
                "inkling",
                "--tool-call-parser",
                "inkling",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "8",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--max-total-tokens",
                "20000",
                "--max-running-requests",
                "4",
                "--disable-prefill-cuda-graph",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
