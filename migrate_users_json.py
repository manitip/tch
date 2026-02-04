#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_users_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"users.json not found at {path}")
    data = json.loads(path.read_text("utf-8"))
    if isinstance(data, dict):
        out: List[Dict[str, Any]] = []
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            value = dict(value)
            value.setdefault("telegram_id", key)
            out.append(value)
        return out
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError("users.json must be a list or dict")


def ensure_users_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER UNIQUE NOT NULL,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            login TEXT,
            password_hash TEXT,
            name TEXT,
            team TEXT,
            role TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            cash_scopes TEXT,
            cash_ops TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT
        );
        """
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users);").fetchall()}
    missing = {
        "username": "ALTER TABLE users ADD COLUMN username TEXT;",
        "first_name": "ALTER TABLE users ADD COLUMN first_name TEXT;",
        "last_name": "ALTER TABLE users ADD COLUMN last_name TEXT;",
        "login": "ALTER TABLE users ADD COLUMN login TEXT;",
        "password_hash": "ALTER TABLE users ADD COLUMN password_hash TEXT;",
        "name": "ALTER TABLE users ADD COLUMN name TEXT;",
        "team": "ALTER TABLE users ADD COLUMN team TEXT;",
        "role": "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'viewer';",
        "is_active": "ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1;",
        "cash_scopes": "ALTER TABLE users ADD COLUMN cash_scopes TEXT;",
        "cash_ops": "ALTER TABLE users ADD COLUMN cash_ops TEXT;",
        "created_at": "ALTER TABLE users ADD COLUMN created_at TEXT;",
        "updated_at": "ALTER TABLE users ADD COLUMN updated_at TEXT;",
    }
    for col, stmt in missing.items():
        if col not in cols:
            conn.execute(stmt)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_login ON users(login);")
    conn.commit()


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _serialize_list(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return value.strip() or None
    return json.dumps([value], ensure_ascii=False)


def migrate_users(users: List[Dict[str, Any]], db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_users_schema(conn)
    now = iso_now()
    for item in users:
        telegram_id = int(item.get("telegram_id") or 0)
        if not telegram_id:
            continue
        role = _normalize_text(item.get("role")) or "viewer"
        is_active = 1 if item.get("active", True) else 0
        name = _normalize_text(item.get("name"))
        username = _normalize_text(item.get("username"))
        first_name = _normalize_text(item.get("first_name"))
        last_name = _normalize_text(item.get("last_name"))
        cash_ops = _serialize_list(item.get("cash_ops"))
        cash_scopes = _serialize_list(item.get("cash_scopes"))

        existing = conn.execute(
            "SELECT * FROM users WHERE telegram_id=?;",
            (telegram_id,),
        ).fetchone()
        if not existing:
            conn.execute(
                """
                INSERT INTO users (
                    telegram_id, username, first_name, last_name, login, password_hash,
                    name, team, role, is_active, cash_scopes, cash_ops, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    telegram_id,
                    username,
                    first_name,
                    last_name,
                    f"tg_{telegram_id}",
                    None,
                    name,
                    None,
                    role,
                    is_active,
                    cash_scopes,
                    cash_ops,
                    now,
                    now,
                ),
            )
        else:
            updates: List[str] = ["role=?", "is_active=?", "updated_at=?"]
            params: List[Any] = [role, is_active, now]
            if name:
                updates.append("name=?")
                params.append(name)
            if username:
                updates.append("username=?")
                params.append(username)
            if first_name:
                updates.append("first_name=?")
                params.append(first_name)
            if last_name:
                updates.append("last_name=?")
                params.append(last_name)
            if cash_ops is not None:
                updates.append("cash_ops=?")
                params.append(cash_ops)
            if cash_scopes is not None:
                updates.append("cash_scopes=?")
                params.append(cash_scopes)
            params.append(telegram_id)
            conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE telegram_id=?;",
                params,
            )
    conn.commit()
    conn.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    db_path = Path(os.getenv("DB_PATH", str(base_dir / "db.sqlite3")))
    if not db_path.is_absolute():
        db_path = base_dir / db_path
    users_path = Path(os.getenv("USERS_JSON", str(base_dir / "users.json")))
    if not users_path.is_absolute():
        users_path = base_dir / users_path
    users = load_users_json(users_path)
    migrate_users(users, db_path)
    print(f"Migrated {len(users)} users into {db_path}")


if __name__ == "__main__":
    main()