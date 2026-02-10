import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402


class SummarySubaccountsIsolationTests(unittest.TestCase):
    def setUp(self):
        self._old_db_path = app.CFG.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        app.CFG.DB_PATH = str(Path(self._tmpdir.name) / "test.sqlite3")
        app.init_db()

    def tearDown(self):
        app.CFG.DB_PATH = self._old_db_path
        self._tmpdir.cleanup()

    def test_main_month_calculations_ignore_subaccounts(self):
        now = "2026-02-01T00:00:00+00:00"
        month_id = app.db_exec_returning_id(
            """
            INSERT INTO months (year, month, monthly_min_needed, start_balance, sundays_override, created_at, updated_at)
            VALUES (?, ?, ?, ?, NULL, ?, ?);
            """,
            (2026, 2, 1000.0, 50.0, now, now),
        )

        app.db_exec(
            """
            INSERT INTO services (
                month_id, service_date, idx, cashless, cash, total,
                weekly_min_needed, mnsps_status, pvs_ratio, income_type, account,
                created_at, updated_at
            ) VALUES
                (?, '2026-02-01', 1, 0, 100, 100, 0, 'Не собрана', 0, 'donation', 'main', ?, ?),
                (?, '2026-02-08', 2, 200, 0, 200, 0, 'Не собрана', 0, 'donation', 'main', ?, ?),
                (?, '2026-02-01', 1, 999, 0, 999, 0, 'Не собрана', 0, 'donation', 'praise', ?, ?),
                (?, '2026-02-08', 2, 0, 777, 777, 0, 'Не собрана', 0, 'donation', 'alpha', ?, ?);
            """,
            (month_id, now, now, month_id, now, now, month_id, now, now, month_id, now, now),
        )

        app.db_exec(
            """
            INSERT INTO expenses (
                month_id, expense_date, category, title, qty, unit_amount, total,
                comment, is_system, account, created_at, updated_at
            ) VALUES
                (?, '2026-02-10', 'Операционные', 'Расход main', 1, 30, 30, NULL, 0, 'main', ?, ?),
                (?, '2026-02-10', 'Операционные', 'Расход praise', 1, 500, 500, NULL, 0, 'praise', ?, ?);
            """,
            (month_id, now, now, month_id, now, now),
        )

        summary = app.compute_month_summary(month_id, ensure_tithe=False)

        self.assertEqual(summary["month_income_sum"], 300.0)
        self.assertEqual(summary["month_expenses_sum"], 30.0)
        self.assertEqual(summary["month_balance"], 270.0)
        self.assertEqual(summary["fact_balance"], 320.0)
        self.assertEqual(summary["avg_sunday"], 150.0)


if __name__ == "__main__":
    unittest.main()
