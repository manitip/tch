import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402


class AdminDiagnosticsApiTests(unittest.TestCase):
    def setUp(self):
        self._old_db_path = app.CFG.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        app.CFG.DB_PATH = str(Path(self._tmpdir.name) / "test.sqlite3")
        app.init_db()
        self.client = TestClient(app.APP)
        app.APP.dependency_overrides[app.get_current_user] = lambda: {
            "id": 1,
            "telegram_id": 100,
            "name": "Admin",
            "role": "admin",
            "active": 1,
        }

    def tearDown(self):
        app.CFG.DB_PATH = self._old_db_path
        app.APP.dependency_overrides.clear()
        self._tmpdir.cleanup()

    def test_run_and_list_diagnostics(self):
        res = self.client.post(
            "/api/admin/diagnostics/run",
            json={"suite": "quick", "mode": "safe", "options": {"timeout_sec": 30}},
        )
        self.assertEqual(res.status_code, 200)
        run_id = res.json()["run_id"]

        for _ in range(20):
            one = self.client.get(f"/api/admin/diagnostics/runs/{run_id}")
            self.assertEqual(one.status_code, 200)
            if one.json().get("status") != "running":
                break

        listed = self.client.get("/api/admin/diagnostics/runs?limit=5")
        self.assertEqual(listed.status_code, 200)
        ids = [item["id"] for item in listed.json().get("items", [])]
        self.assertIn(run_id, ids)

        downloaded = self.client.get(f"/api/admin/diagnostics/runs/{run_id}/download?format=json")
        self.assertEqual(downloaded.status_code, 200)
        self.assertIn("steps", downloaded.json())


if __name__ == "__main__":
    unittest.main()
