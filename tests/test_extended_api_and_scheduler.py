import datetime as dt
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402
import cashflow_models as cf_models  # noqa: E402


class BaseDbTestCase(unittest.TestCase):
    def setUp(self):
        self._old_db_path = app.CFG.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        app.CFG.DB_PATH = str(Path(self._tmpdir.name) / "test.sqlite3")
        app.init_db()

    def tearDown(self):
        app.CFG.DB_PATH = self._old_db_path
        app.APP.dependency_overrides.clear()
        self._tmpdir.cleanup()

    def _insert_user(self, telegram_id: int, role: str, name: str = "User") -> dict:
        now = "2026-02-01T00:00:00+00:00"
        user_id = app.db_exec_returning_id(
            """
            INSERT INTO users (telegram_id, name, role, active, created_at)
            VALUES (?, ?, ?, 1, ?);
            """,
            (telegram_id, name, role, now),
        )
        return {
            "id": user_id,
            "telegram_id": telegram_id,
            "name": name,
            "role": role,
            "active": 1,
        }

    def _insert_month_and_expense(self) -> int:
        now = "2026-02-01T00:00:00+00:00"
        month_id = app.db_exec_returning_id(
            """
            INSERT INTO months (year, month, monthly_min_needed, start_balance, sundays_override, created_at, updated_at)
            VALUES (?, ?, ?, ?, NULL, ?, ?);
            """,
            (2026, 2, 1000.0, 0.0, now, now),
        )
        return app.db_exec_returning_id(
            """
            INSERT INTO expenses (
                month_id, expense_date, category, title, qty, unit_amount, total,
                comment, is_system, account, created_at, updated_at
            ) VALUES (?, '2026-02-10', 'Операционные', 'Тестовый расход', 1, 10, 10, NULL, 0, 'main', ?, ?);
            """,
            (month_id, now, now),
        )


class AttachmentsApiTests(BaseDbTestCase):
    def setUp(self):
        super().setUp()
        self._old_attachments_dir = app.ATTACHMENTS_DIR
        app.ATTACHMENTS_DIR = Path(self._tmpdir.name) / "uploads" / "receipts"
        self.client = TestClient(app.APP)

    def tearDown(self):
        app.ATTACHMENTS_DIR = self._old_attachments_dir
        super().tearDown()

    def test_attachment_upload_list_download_delete_flow(self):
        admin = self._insert_user(101, "admin", "Admin")
        expense_id = self._insert_month_and_expense()

        app.APP.dependency_overrides[app.get_current_user] = lambda: admin

        files = {"file": ("receipt.png", b"\x89PNG\r\n\x1a\n123456", "image/png")}
        up = self.client.post(f"/api/expenses/{expense_id}/attachments", files=files)
        self.assertEqual(up.status_code, 200)
        attachment_id = up.json()["attachment"]["id"]

        listed = self.client.get(f"/api/expenses/{expense_id}/attachments")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["items"]), 1)

        fetched = self.client.get(f"/api/attachments/{attachment_id}?inline=0")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.headers["content-type"], "image/png")
        self.assertIn("receipt.png", fetched.headers["content-disposition"])

        deleted = self.client.delete(f"/api/attachments/{attachment_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json(), {"ok": True})

        listed_after = self.client.get(f"/api/expenses/{expense_id}/attachments")
        self.assertEqual(listed_after.status_code, 200)
        self.assertEqual(listed_after.json()["items"], [])

    def test_attachment_upload_rejects_unsupported_mime(self):
        admin = self._insert_user(102, "admin", "Admin")
        expense_id = self._insert_month_and_expense()

        app.APP.dependency_overrides[app.get_current_user] = lambda: admin
        files = {"file": ("bad.txt", b"not allowed", "text/plain")}

        resp = self.client.post(f"/api/expenses/{expense_id}/attachments", files=files)
        self.assertEqual(resp.status_code, 415)
        self.assertIn("Unsupported file type", resp.text)


class CashflowRoutesAccessTests(BaseDbTestCase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(app.APP)
        self._old_users_json_path = app.CFG.USERS_JSON_PATH
        users_path = Path(self._tmpdir.name) / "users.json"
        users_path.write_text(
            json.dumps(
                [
                    {"telegram_id": 1, "name": "Admin", "role": "admin", "active": True},
                    {"telegram_id": 2, "name": "Signer", "role": "cash_signer", "active": True},
                    {"telegram_id": 3, "name": "Viewer", "role": "viewer", "active": True},
                ]
            ),
            encoding="utf-8",
        )
        app.CFG.USERS_JSON_PATH = str(users_path)
        app.ALLOWLIST_CACHE = {}
        app.ALLOWLIST_MTIME = None

        app.refresh_allowlist_if_needed()
        cfg = cf_models.CashflowConfig(
            base_dir=Path(self._tmpdir.name),
            db_path=Path(app.CFG.DB_PATH),
            users_json_path=users_path,
            uploads_dir=Path(self._tmpdir.name) / "uploads",
            timezone="Europe/Warsaw",
        )
        with app.db_connect() as conn:
            cf_models.init_cashflow_db(conn)
            self.request_id = cf_models.create_cash_request(
                conn,
                cfg,
                account="main",
                op_type="collect",
                amount=100.0,
                created_by_telegram_id=1,
            )

    def tearDown(self):
        app.CFG.USERS_JSON_PATH = self._old_users_json_path
        super().tearDown()

    def _auth_headers(self, telegram_id: int) -> dict:
        token = app.make_session_token(
            {"telegram_id": telegram_id, "exp": int(time.time()) + 3600},
            app.CFG.SESSION_SECRET,
        )
        return {"Authorization": f"Bearer {token}"}

    def test_cashflow_request_detail_denies_non_participant(self):
        resp = self.client.get(
            f"/api/cashflow/requests/{self.request_id}",
            headers=self._auth_headers(3),
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Not a participant", resp.text)

    def test_cashflow_request_detail_allows_participant_signer(self):
        resp = self.client.get(
            f"/api/cashflow/requests/{self.request_id}",
            headers=self._auth_headers(2),
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["request"]["id"], self.request_id)
        participant_ids = [p["telegram_id"] for p in body["participants"]]
        self.assertIn(2, participant_ids)


class SchedulerEdgeCaseTests(unittest.TestCase):
    def test_parse_hhmm_invalid_fallbacks_to_default(self):
        self.assertEqual(app.parse_hhmm("bad", 18, 0), (18, 0))
        self.assertEqual(app.parse_hhmm("24:61", 21, 15), (21, 15))

    def test_reschedule_jobs_without_daily_expenses_adds_only_required_jobs(self):
        fake_scheduler = Mock()
        fake_settings = {
            "timezone": "Europe/Warsaw",
            "sunday_report_time": "99:99",  # invalid -> fallback 18:00
            "month_report_time": "21:45",
            "daily_expenses_enabled": 0,
        }

        with patch.object(app, "scheduler", fake_scheduler), patch.object(app, "get_settings", return_value=fake_settings):
            app.reschedule_jobs()

        self.assertEqual(fake_scheduler.remove_job.call_count, 3)
        self.assertEqual(fake_scheduler.add_job.call_count, 2)
        sunday_call = fake_scheduler.add_job.call_args_list[0]
        trigger_repr = str(sunday_call.kwargs["trigger"])
        self.assertIn("day_of_week='sun'", trigger_repr)
        self.assertIn("hour='18'", trigger_repr)


class AdminReportScheduleTests(BaseDbTestCase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(app.APP)

    def test_admin_settings_apply_custom_report_time_to_scheduler(self):
        admin = self._insert_user(777, "admin", "Admin")
        app.APP.dependency_overrides[app.get_current_user] = lambda: admin

        payload = {
            "sunday_report_time": "07:35",
            "month_report_time": "22:10",
            "daily_expenses_enabled": True,
            "timezone": "Europe/Warsaw",
        }

        fake_scheduler = Mock()
        with patch.object(app, "scheduler", fake_scheduler):
            resp = self.client.put("/api/settings", json=payload)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(fake_scheduler.remove_job.call_count, 3)
        self.assertEqual(fake_scheduler.add_job.call_count, 3)

        jobs = {call.kwargs["id"]: call.kwargs for call in fake_scheduler.add_job.call_args_list}
        sunday_trigger = jobs["job_sunday_report"]["trigger"]
        daily_trigger = jobs["job_daily_expenses"]["trigger"]

        now = dt.datetime(2026, 2, 2, 6, 0, tzinfo=dt.timezone.utc)
        next_sunday_run = sunday_trigger.get_next_fire_time(None, now)
        next_daily_run = daily_trigger.get_next_fire_time(None, now)

        self.assertEqual(next_sunday_run.weekday(), 6)  # Sunday
        self.assertEqual((next_sunday_run.hour, next_sunday_run.minute), (7, 35))
        self.assertEqual((next_daily_run.hour, next_daily_run.minute), (22, 10))

        def _consume_task(coro):
            coro.close()
            return Mock()

        with patch.object(app, "run_sunday_report_job", new=AsyncMock()) as sunday_job, \
             patch.object(app, "run_daily_expenses_job", new=AsyncMock()) as daily_job, \
             patch.object(app.asyncio, "create_task", side_effect=_consume_task) as create_task:
            jobs["job_sunday_report"]["func"]()
            jobs["job_daily_expenses"]["func"]()

        self.assertEqual(create_task.call_count, 2)
        self.assertEqual(sunday_job.call_count, 1)
        self.assertEqual(daily_job.call_count, 1)

        settings_resp = self.client.get("/api/settings")
        self.assertEqual(settings_resp.status_code, 200)
        updated = settings_resp.json()
        self.assertEqual(updated["sunday_report_time"], "07:35")
        self.assertEqual(updated["month_report_time"], "22:10")
        self.assertTrue(updated["daily_expenses_enabled"])


if __name__ == "__main__":
    unittest.main()
