import asyncio
import unittest

from sglang.srt.entrypoints.log_exclude_ids import ID_HEADERS, LogExcludeIdsMiddleware
from sglang.test.test_utils import CustomTestCase


class _RecordingApp:
    """Downstream ASGI app that records the headers it was handed."""

    def __init__(self):
        self.scope = None

    async def __call__(self, scope, receive, send):
        self.scope = scope


class TestLogExcludeIdsMiddleware(CustomTestCase):
    def setUp(self):
        self.downstream = _RecordingApp()
        self.middleware = LogExcludeIdsMiddleware(self.downstream)

    def _call(self, headers, path="/v1/chat/completions", scope_type="http"):
        scope = {"type": scope_type, "path": path, "headers": headers}
        asyncio.run(self.middleware(scope, None, None))
        return self.downstream.scope["headers"]

    def _id_header_values(self, headers):
        return {k: v for k, v in headers if k in ID_HEADERS}

    def test_loggable_request_opts_in(self):
        headers = self._call([(b"modal-log-exclude", b"false")])

        self.assertEqual(
            self._id_header_values(headers),
            {header: b"1" for header in ID_HEADERS},
        )

    def test_excluded_request_gets_no_ids(self):
        headers = self._call([(b"modal-log-exclude", b"true")])

        self.assertEqual(self._id_header_values(headers), {})

    def test_missing_header_is_treated_as_excluded(self):
        headers = self._call([(b"content-type", b"application/json")])

        self.assertEqual(self._id_header_values(headers), {})

    def test_excluded_request_cannot_opt_itself_in(self):
        headers = self._call(
            [
                (b"modal-log-exclude", b"true"),
                (b"x-sglext-return-input-ids", b"1"),
                (b"x-sglext-return-output-ids", b"1"),
            ]
        )

        self.assertEqual(self._id_header_values(headers), {})

    def test_other_headers_are_preserved(self):
        headers = self._call(
            [(b"modal-log-exclude", b"0"), (b"authorization", b"Bearer token")]
        )

        self.assertIn((b"authorization", b"Bearer token"), headers)
        self.assertIn((b"modal-log-exclude", b"0"), headers)

    def test_non_chat_path_is_untouched(self):
        original = [(b"modal-log-exclude", b"false")]
        headers = self._call(original, path="/v1/completions")

        self.assertEqual(headers, original)


if __name__ == "__main__":
    unittest.main()
