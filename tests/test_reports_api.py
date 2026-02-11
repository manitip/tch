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
                self.assertFalse(bundle_mock.await_args.kwargs["raise_on_error"])

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
