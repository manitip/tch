import os
import unittest

os.environ.setdefault("BOT_TOKEN", "test-token")

import app  # noqa: E402


class MainMenuKeyboardTests(unittest.TestCase):
    def test_admin_uses_only_persistent_bottom_menu(self):
        kb = app.main_menu_kb("admin")
        self.assertEqual(kb.inline_keyboard, [])

    def test_cash_signer_uses_only_persistent_bottom_menu(self):
        kb = app.main_menu_kb("cash_signer")
        self.assertEqual(kb.inline_keyboard, [])

    def test_accountant_keeps_chat_menu_buttons(self):
        kb = app.main_menu_kb("accountant")
        self.assertGreater(len(kb.inline_keyboard), 0)


if __name__ == "__main__":
    unittest.main()
