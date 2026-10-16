"""Media that cannot be fetched or decoded is a client error, not a 500."""

import asyncio
import unittest
from concurrent.futures import Future

import requests

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.media_artifacts.base import _await_media_snapshot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _failed(exc):
    future = Future()
    future.set_exception(exc)
    return future


class TestMediaSnapshotErrors(CustomTestCase):
    def _await(self, exc):
        return asyncio.run(_await_media_snapshot(_failed(exc), Modality.IMAGE, 2))

    def test_unfetchable_media_is_a_request_error(self):
        with self.assertRaisesRegex(ValueError, "IMAGE data at index 2"):
            self._await(requests.exceptions.ConnectionError("refused"))

    def test_server_fault_propagates(self):
        with self.assertRaises(MemoryError):
            self._await(MemoryError())


if __name__ == "__main__":
    unittest.main()
