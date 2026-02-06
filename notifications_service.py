from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import sqlite3


def create_notification(
    *,
    user_id: int,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> int:
    from app import db_exec_returning_id, iso_now, CFG
    now = iso_now(CFG.tzinfo())
    payload_json = json.dumps(payload or {}, ensure_ascii=False)
    return db_exec_returning_id(
        """
        INSERT INTO notifications (user_id, type, payload_json, is_read, created_at)
        VALUES (?, ?, ?, 0, ?);
        """,
        (int(user_id), str(event_type), payload_json, now),
    )


def notify_users(
    user_ids: Iterable[int],
    *,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> List[int]:
    ids: List[int] = []
    for uid in user_ids:
        try:
            ids.append(create_notification(user_id=int(uid), event_type=event_type, payload=payload))
        except Exception:
            continue
    return ids


def list_notifications(user_id: int, limit: int = 50, offset: int = 0) -> List[sqlite3.Row]:
    from app import db_fetchall
    return db_fetchall(
        """
        SELECT *
        FROM notifications
        WHERE user_id=?
        ORDER BY created_at DESC, id DESC
        LIMIT ? OFFSET ?;
        """,
        (int(user_id), int(limit), int(offset)),
    )


def mark_notification_read(user_id: int, notification_id: int) -> None:
    from app import db_exec
    db_exec(
        """
        UPDATE notifications
        SET is_read=1
        WHERE id=? AND user_id=?;
        """,
        (int(notification_id), int(user_id)),
    )


def mark_all_read(user_id: int) -> None:
    from app import db_exec
    db_exec(
        "UPDATE notifications SET is_read=1 WHERE user_id=?;",
        (int(user_id),),
    )
