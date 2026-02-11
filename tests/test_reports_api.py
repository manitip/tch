import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402


class ReportsApiTests(unittest.TestCase):
    def test_api_report_sunday_sends_text_and_png_bundle(self):
        fake_settings = {"timezone": "Europe/Warsaw"}
        recipients = [101, 202]
        bundle_mock = AsyncMock()

        async def _run():
            with patch.object(app, "get_settings", return_value=fake_settings), \
                 patch.object(app, "list_report_recipients", return_value=recipients), \
                 patch.object(app, "send_sunday_reports_bundle", bundle_mock):
                result = await app.api_report_sunday({"id": 1, "role": "admin"})
                self.assertEqual(result, {"ok": True})
                bundle_mock.assert_awaited_once()
                called_today, called_recipients = bundle_mock.await_args.args[:2]
                self.assertIsInstance(called_today, app.dt.date)
                self.assertEqual(called_recipients, recipients)
                self.assertTrue(bundle_mock.await_args.kwargs["raise_on_error"])

        asyncio.run(_run())

    def test_sunday_scheduler_job_does_not_send_duplicate_png(self):
        fake_settings = {"timezone": "Europe/Warsaw"}
        recipients = [101, 202]
        bundle_mock = AsyncMock()
        png_send_mock = AsyncMock()

        async def _execute_job(_name, c):
            return await c

        async def _run():
            with patch.object(app, "run_job_with_logging", AsyncMock(side_effect=_execute_job)), \
                 patch.object(app, "get_settings", return_value=fake_settings), \
                 patch.object(app, "list_report_recipients", return_value=recipients), \
                 patch.object(app, "send_sunday_reports_bundle", bundle_mock), \
                 patch.object(app, "send_report_png_to_recipients", png_send_mock):
                await app.run_sunday_report_job()
                bundle_mock.assert_awaited_once()
                png_send_mock.assert_not_awaited()

        asyncio.run(_run())

    def test_report_builders_do_not_attach_inline_buttons(self):
        today = app.dt.date(2026, 1, 25)
        sunday_text, sunday_kb = app.build_sunday_report_text(today)
        expenses_text, expenses_kb = app.build_month_expenses_report_text(today)

        self.assertIn("Отчёт по пожертвованиям", sunday_text)
        self.assertIsNone(sunday_kb)
        self.assertIn("Расходы", expenses_text)
        self.assertIsNone(expenses_kb)


if __name__ == "__main__":
    unittest.main()
