import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402


class AdminFullSystemTestApiTests(unittest.TestCase):
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

    def _user(self, role: str):
        return {
            "id": 1,
            "telegram_id": 100,
            "name": "Tester",
            "role": role,
            "active": 1,
        }

    def test_admin_can_run_full_system_test(self):
        app.APP.dependency_overrides[app.get_current_user] = lambda: self._user("admin")

        res = self.client.post("/api/admin/system/full-test", json={})
        self.assertEqual(res.status_code, 200)
        payload = res.json()

        self.assertIn(payload["status"], ("ok", "warn", "fail"))
        self.assertGreaterEqual(len(payload["checks"]), 5)
        check_names = {item["name"] for item in payload["checks"]}
        self.assertTrue({"database", "settings", "scheduler", "core_data", "storage"}.issubset(check_names))

    def test_non_admin_cannot_run_full_system_test(self):
        app.APP.dependency_overrides[app.get_current_user] = lambda: self._user("accountant")

        res = self.client.post("/api/admin/system/full-test", json={})
        self.assertEqual(res.status_code, 403)


if __name__ == "__main__":
    unittest.main()
