import os
import unittest

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402

PIL_AVAILABLE = True
try:
    app.require_pillow()
except Exception:
    PIL_AVAILABLE = False


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for PNG rendering tests")
class ReportPngRenderTests(unittest.TestCase):
    def test_render_png_with_long_plan_labels_and_open(self):
        month_row = {"month": 2, "year": 2026}
        summary = {
            "month_income_sum": 2222.0,
            "month_expenses_sum": 222.2,
            "month_balance": 1999.8,
            "fact_balance": 1999.8,
            "monthly_min_needed": 2222.0,
            "monthly_completion": 1.0,
            "sddr": 2222.0,
            "psdpm": 0.0,
            "avg_sunday": 2222.0,
            "weekly_min_needed": 1111.0,
        }
        services = [
            {
                "service_date": "2026-02-01",
                "total": 1111.0,
                "mnsps_status": "Не собрана",
                "weekly_min_needed": 1111.0,
            },
            {
                "service_date": "2026-02-08",
                "total": 1400.0,
                "mnsps_status": "Собрана",
                "weekly_min_needed": 1111.0,
            },
        ]

        png = app.render_month_report_png(
            month_row=month_row,
            summary=summary,
            services=services,
            top_categories=[{"category": "Десятина", "sum": 222.2}],
            expenses=[
                {
                    "expense_date": "2026-02-28",
                    "category": "Десятина",
                    "title": "10% объединение",
                    "total": 222.2,
                }
            ],
            subaccount_services=[],
            subaccount_expenses=[],
            preset="landscape",
            pixel_ratio=2,
        )

        self.assertGreater(len(png), 5000)
        Image, _, _ = app.require_pillow()
        with Image.open(app.io.BytesIO(png)) as img:
            img.load()
            self.assertEqual(img.format, "PNG")

    def test_service_bar_color_uses_min_threshold(self):
        month_row = {"month": 2, "year": 2026}
        summary = {
            "month_income_sum": 2100.0,
            "month_expenses_sum": 0.0,
            "month_balance": 2100.0,
            "fact_balance": 2100.0,
            "monthly_min_needed": 4000.0,
            "monthly_completion": 0.5,
            "sddr": 0.0,
            "psdpm": 0.0,
            "avg_sunday": 0.0,
            "weekly_min_needed": 1000.0,
        }
        services = [
            {
                "service_date": "2026-02-01",
                "total": 1000.0,
                "mnsps_status": "Не собрана",
                "weekly_min_needed": 1000.0,
            },
            {
                "service_date": "2026-02-08",
                "total": 900.0,
                "mnsps_status": "Собрана",
                "weekly_min_needed": 1000.0,
            },
        ]

        png = app.render_month_report_png(
            month_row=month_row,
            summary=summary,
            services=services,
            top_categories=[],
            expenses=[],
            subaccount_services=[],
            subaccount_expenses=[],
            preset="landscape",
            pixel_ratio=2,
        )
        Image, _, _ = app.require_pillow()
        with Image.open(app.io.BytesIO(png)) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = w / 1600.0
            margin = int(44 * scale)
            gap = int(20 * scale)
            kpi_y = margin + int(64 * scale)
            kpi_h = int(148 * scale)
            section_y = kpi_y + kpi_h + int(30 * scale)
            left_w = int(w * 0.40)
            right_w = w - margin * 2 - gap - left_w
            right_x = margin + left_w + gap
            content_h = h - section_y - int(36 * scale) - margin
            list_card_h = min(int(260 * scale), max(int(180 * scale), int(content_h * 0.30)))
            top_h = max(content_h - list_card_h - gap, int(280 * scale))
            chart_pad = int(24 * scale)
            chart_x0 = right_x + chart_pad
            chart_y0 = section_y + int(56 * scale)
            chart_w = right_w - chart_pad * 2
            sub_table_h = min(int(150 * scale), int(top_h * 0.34))
            chart_h = top_h - sub_table_h - int(92 * scale)
            base_y = chart_y0 + chart_h - int(18 * scale)
            plot_h = chart_h - int(44 * scale)
            bar_gap = max(6, int(10 * scale))
            n = len(services)
            bar_w = max(10, int((chart_w - bar_gap * (n + 1)) / n))

            max_total = 1000.0
            first_h = int((1000.0 / max_total) * plot_h)
            second_h = int((900.0 / max_total) * plot_h)
            first_x = chart_x0 + bar_gap
            second_x = chart_x0 + bar_gap + (bar_w + bar_gap)

            first_pixel = img.getpixel((first_x + bar_w // 2, base_y - first_h // 2))
            second_pixel = img.getpixel((second_x + bar_w // 2, base_y - second_h // 2))

            self.assertEqual(first_pixel, (16, 185, 129))
            self.assertEqual(second_pixel, (239, 68, 68))


if __name__ == "__main__":
    unittest.main()
