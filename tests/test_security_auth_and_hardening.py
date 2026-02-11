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


    def test_verify_session_token_rejects_invalid_format(self):
        with self.assertRaises(HTTPException) as ctx:
            app.verify_session_token("just-one-part", app.CFG.SESSION_SECRET)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Invalid token format", str(ctx.exception.detail))

    def test_api_me_rejects_corrupted_token_payload(self):
        # valid bearer shape but invalid payload/signature relationship
        bad_token = "not-base64.not-base64"
        resp = self.client.get("/api/me", headers={"Authorization": f"Bearer {bad_token}"})
        self.assertEqual(resp.status_code, 401)

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

    def test_restore_backup_sanitizes_uploaded_filename(self):
        app.APP.dependency_overrides[app.get_current_user] = lambda: {
            "id": 1,
            "telegram_id": 1,
            "name": "admin",
            "role": "admin",
            "active": 1,
        }

        escaped_target = Path("/tmp/tch_restore_escape_test.sqlite3")
        if escaped_target.exists():
            escaped_target.unlink()

        files = {"file": (str(escaped_target), b"not-a-sqlite", "application/octet-stream")}
        resp = self.client.post("/api/backups/restore", files=files)
        self.assertIn(resp.status_code, (400, 422))
        self.assertFalse(escaped_target.exists())



if __name__ == "__main__":
    unittest.main()
