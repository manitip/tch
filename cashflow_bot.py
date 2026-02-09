"""cashflow_bot.py

Уведомления в Telegram для запросов подписания наличных.

Этот модуль НЕ запускает бота и не делает polling.
Он предоставляет функции, которые можно вызывать из FastAPI роутов.

Интеграция (пример):

    # в app.py после создания bot = Bot(...)
    import cashflow_bot
    cashflow_bot.set_bot(bot)

    # в cashflow_routes.py можно звать notify_* через BackgroundTasks

"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo


_BOT: Optional[Bot] = None


@dataclass(frozen=True)
class CashflowBotConfig:
    """Параметры ссылок для WebApp."""

    app_url: str
    webapp_url: str = ""
    cashapp_path: str = "/cashapp"

    @property
    def cashapp_url(self) -> str:
        base = (self.app_url or "").rstrip("/")
        path = (self.cashapp_path or "/cashapp")
        if not path.startswith("/"):
            path = "/" + path
        return base + path


def load_bot_config() -> CashflowBotConfig:
    # В app.py уже есть APP_URL в .env; берём то же.
    return CashflowBotConfig(
        app_url=os.getenv("APP_URL", "http://localhost:8000").strip(),
        webapp_url=os.getenv("WEBAPP_URL", "").strip(),
    )


def set_bot(bot: Bot) -> None:
    global _BOT
    _BOT = bot


def get_bot() -> Bot:
    if _BOT is None:
        raise RuntimeError("Bot is not set. Call cashflow_bot.set_bot(bot) from app.py")
    return _BOT


def require_https_webapp_url(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None
    if parsed.scheme.lower() != "https":
        return None
    return url


def _cashapp_kb(cfg: CashflowBotConfig, request_id: int) -> Optional[InlineKeyboardMarkup]:
    url = _cashapp_url(cfg, request_id)
    if not url:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Открыть и подтвердить", web_app=WebAppInfo(url=url))]]
    )

def _cashapp_url(cfg: CashflowBotConfig, request_id: int) -> Optional[str]:
    primary = require_https_webapp_url(f"{cfg.cashapp_url}?rid={int(request_id)}")
    if primary:
        return primary
    webapp_url = require_https_webapp_url(cfg.webapp_url)
    if not webapp_url:
        return None
    parsed = urllib.parse.urlparse(webapp_url)
    path = cfg.cashapp_path or "/cashapp"
    if not path.startswith("/"):
        path = "/" + path
    base = parsed._replace(path=path, params="", query="", fragment="")
    cashapp_base = urllib.parse.urlunparse(base)
    return require_https_webapp_url(f"{cashapp_base}?rid={int(request_id)}")

def format_account(account: str) -> str:
    a = (account or "").lower()
    return {"main": "Основной", "praise": "Прославление", "alpha": "Альфа курс"}.get(a, account)


def format_op(op_type: str) -> str:
    t = (op_type or "").lower()
    return {"collect": "Сбор наличных", "withdraw": "Изъятие наличных"}.get(t, op_type)


async def safe_send_message(chat_id: int, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> None:
    bot = get_bot()
    try:
        await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
    except (TelegramBadRequest, TelegramForbiddenError):
        # пользователь мог заблокировать бота / не стартанул чат
        return
    except Exception:
        return


async def notify_signers_new_request(
    *,
    request_id: int,
    account: str,
    op_type: str,
    amount: float,
    signer_ids: Iterable[int],
    is_retry: bool = False,
    admin_comment: Optional[str] = None,
) -> None:
    cfg = load_bot_config()
    head = "Повторное подтверждение" if is_retry else "Требуется подтверждение"
    text = (
        f"<b>{head}</b>\n"
        f"Операция: <b>{format_op(op_type)}</b>\n"
        f"Счёт: <b>{format_account(account)}</b>\n"
        f"Сумма: <b>{amount:.2f}</b>\n"
    )
    if admin_comment:
        text += f"Комментарий администратора: <i>{admin_comment}</i>\n"

    kb = _cashapp_kb(cfg, request_id)
    for tid in signer_ids:
        await safe_send_message(int(tid), text, kb)


async def notify_admin_about_decision(
    *,
    admin_id: int,
    request_id: int,
    account: str,
    op_type: str,
    amount: float,
    signer_name: str,
    decision: str,
    reason: Optional[str] = None,
) -> None:
    cfg = load_bot_config()
    decision_text = "подписал" if decision == "SIGNED" else "отказал"
    text = (
        f"<b>Действие подписанта</b>\n"
        f"{signer_name} {decision_text}.\n"
        f"Операция: <b>{format_op(op_type)}</b>\n"
        f"Счёт: <b>{format_account(account)}</b>\n"
        f"Сумма: <b>{amount:.2f}</b>\n"
    )
    if decision != "SIGNED" and reason:
        text += f"Причина отказа: <i>{reason}</i>\n"

    kb = _cashapp_kb(cfg, request_id)
    await safe_send_message(int(admin_id), text, kb)


async def notify_initiator_final(
    *,
    initiator_id: int,
    request_id: int,
    account: str,
    op_type: str,
    amount: float,
    status: str,
) -> None:
    cfg = load_bot_config()
    text = (
        f"<b>Запрос подтверждения завершён</b>\n"
        f"Статус: <b>{status}</b>\n"
        f"Операция: <b>{format_op(op_type)}</b>\n"
        f"Счёт: <b>{format_account(account)}</b>\n"
        f"Сумма: <b>{amount:.2f}</b>\n"
    )
    kb = _cashapp_kb(cfg, request_id)
    await safe_send_message(int(initiator_id), text, kb)
