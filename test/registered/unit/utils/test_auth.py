"""Unit tests for srt/utils/auth.py — no server, no model loading."""

import json
import unittest

from sglang.srt.utils.auth import (
    AuthDecision,
    AuthLevel,
    auth_level,
    decide_request_auth,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestAuthDecision(CustomTestCase):
    def test_not_allowed_with_custom_status(self):
        decision = AuthDecision(allowed=False, error_status_code=403)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.error_status_code, 403)

    def test_frozen(self):
        decision = AuthDecision(allowed=True)
        with self.assertRaises(AttributeError):
            decision.allowed = False


class TestAuthLevel(CustomTestCase):
    def test_is_string_enum(self):
        self.assertIsInstance(AuthLevel.NORMAL, str)
        # str mixin allows direct comparison with string values
        self.assertEqual(AuthLevel.NORMAL, "normal")


class TestAuthLevelDecorator(CustomTestCase):
    def test_decorator_sets_auth_level(self):
        @auth_level(AuthLevel.ADMIN_FORCE)
        def my_endpoint():
            pass

        self.assertEqual(my_endpoint._auth_level, AuthLevel.ADMIN_FORCE)


class TestDecideRequestAuth(CustomTestCase):
    """Tests for the pure decide_request_auth function."""

    # ==================== Always-Allowed Paths ====================

    def test_options_method_always_allowed(self):
        decision = decide_request_auth(
            method="OPTIONS",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key="secret",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(decision.allowed)

    def test_health_path_always_allowed(self):
        decision = decide_request_auth(
            method="GET",
            path="/health",
            authorization_header=None,
            api_key="secret",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_health_subpath_always_allowed(self):
        decision = decide_request_auth(
            method="GET",
            path="/health_generate",
            authorization_header=None,
            api_key="secret",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_metrics_path_always_allowed(self):
        decision = decide_request_auth(
            method="GET",
            path="/metrics",
            authorization_header=None,
            api_key="secret",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    # ==================== NORMAL Auth Level ====================

    def test_normal_no_keys_configured(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_normal_with_api_key_correct(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_normal_with_api_key_wrong(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="Bearer wrong-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_normal_with_api_key_missing_header(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_normal_only_admin_key_configured(self):
        """When only admin_api_key is configured, normal endpoints allow all."""
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header=None,
            api_key=None,
            admin_api_key="admin-secret",
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    # ==================== ADMIN_FORCE Auth Level ====================

    def test_admin_force_no_admin_key_configured(self):
        """ADMIN_FORCE without admin_api_key configured returns 403."""
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.error_status_code, 403)

    def test_admin_force_correct_admin_key(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer admin-secret",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(decision.allowed)

    def test_admin_force_wrong_admin_key(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer wrong-key",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.error_status_code, 401)

    def test_admin_force_api_key_not_accepted(self):
        """ADMIN_FORCE rejects api_key, only accepts admin_api_key."""
        decision = decide_request_auth(
            method="POST",
            path="/admin/endpoint",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(decision.allowed)

    # ==================== ADMIN_OPTIONAL Auth Level ====================

    def test_admin_optional_no_keys_configured(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    def test_admin_optional_only_api_key_correct(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    def test_admin_optional_only_api_key_wrong(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer wrong-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertFalse(decision.allowed)

    def test_admin_optional_only_admin_key_correct(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer admin-secret",
            api_key=None,
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    def test_admin_optional_both_keys_requires_admin(self):
        """When both keys configured, ADMIN_OPTIONAL requires admin_api_key."""
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertFalse(decision.allowed)

    def test_admin_optional_both_keys_admin_accepted(self):
        decision = decide_request_auth(
            method="POST",
            path="/admin/optional",
            authorization_header="Bearer admin-secret",
            api_key="my-api-key",
            admin_api_key="admin-secret",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(decision.allowed)

    # ==================== Bearer Token Edge Cases ====================

    def test_malformed_authorization_header(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="NotBearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_empty_authorization_header(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(decision.allowed)

    def test_bearer_case_insensitive(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/chat/completions",
            authorization_header="BEARER my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)


class TestXApiKeyHeader(CustomTestCase):
    """G-01: `x-api-key` is Bearer-equivalent on /v1/messages* only (spec §1.2)."""

    def _decide_normal(self, *, path, x_api_key_header=None, authorization_header=None):
        return decide_request_auth(
            method="POST",
            path=path,
            authorization_header=authorization_header,
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
            x_api_key_header=x_api_key_header,
        )

    def test_x_api_key_accepted_on_messages(self):
        decision = self._decide_normal(
            path="/v1/messages", x_api_key_header="my-api-key"
        )
        self.assertTrue(decision.allowed)

    def test_x_api_key_accepted_on_count_tokens(self):
        decision = self._decide_normal(
            path="/v1/messages/count_tokens", x_api_key_header="my-api-key"
        )
        self.assertTrue(decision.allowed)

    def test_x_api_key_rejected_wrong_value(self):
        decision = self._decide_normal(path="/v1/messages", x_api_key_header="wrong-key")
        self.assertFalse(decision.allowed)

    def test_x_api_key_empty_value_rejected(self):
        decision = self._decide_normal(path="/v1/messages", x_api_key_header="")
        self.assertFalse(decision.allowed)

    def test_x_api_key_not_honored_on_openai_paths(self):
        """Legacy byte-identical behavior: x-api-key means nothing elsewhere."""
        for path in ("/v1/chat/completions", "/v1/models", "/generate"):
            decision = self._decide_normal(path=path, x_api_key_header="my-api-key")
            self.assertFalse(decision.allowed, msg=path)

    def test_x_api_key_defaults_to_none_legacy_path(self):
        """Existing callers that don't pass x_api_key_header are unaffected."""
        decision = decide_request_auth(
            method="POST",
            path="/v1/messages",
            authorization_header="Bearer my-api-key",
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(decision.allowed)

    def test_x_api_key_never_counts_as_admin_key(self):
        """ADMIN_FORCE/ADMIN_OPTIONAL admin paths stay Bearer-only."""
        for level in (AuthLevel.ADMIN_FORCE, AuthLevel.ADMIN_OPTIONAL):
            decision = decide_request_auth(
                method="POST",
                path="/v1/messages",
                authorization_header=None,
                api_key="my-api-key",
                admin_api_key="admin-secret",
                auth_level=level,
                x_api_key_header="admin-secret",
            )
            self.assertFalse(decision.allowed, msg=level)

    def test_admin_optional_api_key_accepts_x_api_key(self):
        decision = decide_request_auth(
            method="POST",
            path="/v1/messages",
            authorization_header=None,
            api_key="my-api-key",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_OPTIONAL,
            x_api_key_header="my-api-key",
        )
        self.assertTrue(decision.allowed)


class TestApiKeyMiddlewareAnthropicEnvelope(unittest.TestCase):
    """G-01: middleware-level checks of envelope shape + credential acceptance."""

    def _make_app(self, api_key="user-secret", admin_api_key=None):
        from fastapi import FastAPI

        from sglang.srt.utils.auth import add_api_key_middleware, auth_level

        app = FastAPI()

        @app.post("/v1/messages")
        async def _messages():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def _chat():
            return {"ok": True}

        @app.post("/v1/messages/admin_probe")
        @auth_level(AuthLevel.ADMIN_FORCE)
        async def _admin_probe():
            return {"ok": True}

        add_api_key_middleware(app, api_key=api_key, admin_api_key=admin_api_key)
        return app

    def _client(self, app):
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_x_api_key_accepted(self):
        resp = self._client(self._make_app()).post(
            "/v1/messages", headers={"x-api-key": "user-secret"}, json={}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})

    def test_bearer_still_accepted(self):
        resp = self._client(self._make_app()).post(
            "/v1/messages", headers={"Authorization": "Bearer user-secret"}, json={}
        )
        self.assertEqual(resp.status_code, 200)

    def test_401_anthropic_envelope(self):
        resp = self._client(self._make_app()).post("/v1/messages", json={})
        self.assertEqual(resp.status_code, 401)
        body = resp.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "authentication_error")
        self.assertNotIn("Unauthorized", json.dumps(body))
        self.assertRegex(body["request_id"], r"^req_[0-9a-f]{32}$")
        self.assertEqual(resp.headers["request-id"], body["request_id"])

    def test_401_wrong_x_api_key_envelope(self):
        resp = self._client(self._make_app()).post(
            "/v1/messages", headers={"x-api-key": "nope"}, json={}
        )
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"]["type"], "authentication_error")

    def test_403_anthropic_envelope(self):
        # ADMIN_FORCE route with no admin_api_key configured → 403 permission_error.
        resp = self._client(self._make_app()).post(
            "/v1/messages/admin_probe",
            headers={"x-api-key": "user-secret"},
            json={},
        )
        self.assertEqual(resp.status_code, 403)
        body = resp.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "permission_error")

    def test_legacy_envelope_byte_identical_for_other_paths(self):
        resp = self._client(self._make_app()).post("/v1/chat/completions", json={})
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.content, b'{"error":"Unauthorized"}')

    def test_no_api_key_configured_is_noop(self):
        app = self._make_app(api_key=None, admin_api_key=None)
        resp = self._client(app).post("/v1/messages", json={})
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
