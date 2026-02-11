import os
import tempfile
import time
import unittest
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402


class SecurityAuthAndHardeningTests(unittest.TestCase):
    def setUp(self):
        self._old_db_path = app.CFG.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        app.CFG.DB_PATH = str(Path(self._tmpdir.name) / "test.sqlite3")
        app.init_db()
        self.client = TestClient(app.APP)

    def tearDown(self):
        app.CFG.DB_PATH = self._old_db_path
        app.APP.dependency_overrides.clear()
        self._tmpdir.cleanup()

    def _insert_user(self, telegram_id: int, role: str = "admin", active: int = 1):
        now = "2026-02-01T00:00:00+00:00"
        return app.db_exec_returning_id(
            """
            INSERT INTO users (telegram_id, name, role, active, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (telegram_id, f"U{telegram_id}", role, active, now),
        )

    def test_verify_session_token_rejects_tampered_signature(self):
        token = app.make_session_token({"telegram_id": 111, "exp": int(time.time()) + 600}, app.CFG.SESSION_SECRET)
        body, _sig = token.split(".", 1)
        tampered = f"{body}.AAAA"

        with self.assertRaises(HTTPException) as ctx:
            app.verify_session_token(tampered, app.CFG.SESSION_SECRET)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid token signature", str(ctx.exception.detail))

    def test_verify_session_token_rejects_expired_token(self):
        token = app.make_session_token({"telegram_id": 222, "exp": int(time.time()) - 1}, app.CFG.SESSION_SECRET)

        with self.assertRaises(HTTPException) as ctx:
            app.verify_session_token(token, app.CFG.SESSION_SECRET)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Token expired", str(ctx.exception.detail))

    def test_verify_session_token_rejects_invalid_base64_signature(self):
        token = app.make_session_token({"telegram_id": 223, "exp": int(time.time()) + 600}, app.CFG.SESSION_SECRET)
        body, _sig = token.split(".", 1)
        broken = f"{body}.###"

        with self.assertRaises(HTTPException) as ctx:
            app.verify_session_token(broken, app.CFG.SESSION_SECRET)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid token signature", str(ctx.exception.detail))

    def test_verify_session_token_rejects_non_json_payload(self):
        broken_payload = app._b64url_encode(b"not-json")
        sig = app._b64url_encode(
            app.hmac.new(
                app.CFG.SESSION_SECRET.encode("utf-8"),
                broken_payload.encode("utf-8"),
                app.hashlib.sha256,
            ).digest()
        )
        token = f"{broken_payload}.{sig}"

        with self.assertRaises(HTTPException) as ctx:
            app.verify_session_token(token, app.CFG.SESSION_SECRET)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid token payload", str(ctx.exception.detail))

    def test_api_me_requires_bearer_token(self):
        resp = self.client.get("/api/me")
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Missing Authorization Bearer token", resp.text)

    def test_api_me_denies_inactive_user_even_with_valid_signature(self):
        self._insert_user(telegram_id=333, role="admin", active=0)
        token = app.make_session_token({"telegram_id": 333, "exp": int(time.time()) + 600}, app.CFG.SESSION_SECRET)

        resp = self.client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 403)
        self.assertIn("inactive", resp.text.lower())

    def test_backup_download_blocks_path_traversal(self):
        app.APP.dependency_overrides[app.get_current_user] = lambda: {
            "id": 1,
            "telegram_id": 1,
            "name": "admin",
            "role": "admin",
            "active": 1,
        }

        resp = self.client.get("/api/backups/../secret/download")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("Not Found", resp.text)

    def test_backup_restore_rejects_oversized_upload(self):
        app.APP.dependency_overrides[app.get_current_user] = lambda: {
            "id": 1,
            "telegram_id": 1,
            "name": "admin",
            "role": "admin",
            "active": 1,
        }

        payload = b"x" * (app.MAX_BACKUP_UPLOAD_BYTES + 1)
        resp = self.client.post(
            "/api/backups/restore",
            files={"file": ("too-big.bin", payload, "application/octet-stream")},
        )
        self.assertEqual(resp.status_code, 413)
        self.assertIn("too large", resp.text.lower())

    def test_cors_wildcard_origin_not_allowed_when_credentials_enabled(self):
        cors = None
        for middleware in app.APP.user_middleware:
            if middleware.cls is app.CORSMiddleware:
                cors = middleware
                break

        self.assertIsNotNone(cors)
        allow_origins = list(cors.kwargs.get("allow_origins", []))
        self.assertNotIn("*", allow_origins)


if __name__ == "__main__":
    unittest.main()
