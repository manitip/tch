from __future__ import annotations

import os
import sqlite3
from typing import Dict, List


DB_PATH = os.getenv("DB_PATH", "db.sqlite3")


USERS_SCHEMA = """
CREATE TABLE users_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER UNIQUE,
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
    email TEXT,
    last_login_at TEXT,
    password_changed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
"""


def table_info(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(f"PRAGMA table_info({table});").fetchall()


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column for r in table_info(conn, table))


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    ).fetchone()
    return bool(row)

def column_not_null(conn: sqlite3.Connection, table: str, column: str) -> bool:
    for row in table_info(conn, table):
        if row["name"] == column:
            return bool(row["notnull"])
    return False


def rebuild_users_table(conn: sqlite3.Connection) -> None:
    print("Rebuilding users table to make telegram_id nullable and add new columns...")
    conn.executescript(USERS_SCHEMA)
    cols = [r["name"] for r in table_info(conn, "users") if r["name"] != "id"]
    keep_cols = [c for c in cols if column_exists(conn, "users_new", c)]
    col_list = ", ".join(["id"] + keep_cols)
    conn.execute(
        f"INSERT INTO users_new ({col_list}) SELECT {col_list} FROM users;"
    )
    conn.execute("DROP TABLE users;")
    conn.execute("ALTER TABLE users_new RENAME TO users;")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_login ON users(login);")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);")


def migrate_cashflow(conn: sqlite3.Connection) -> None:
    print("Migrating cashflow telegram_id -> user_id...")
    if not table_exists(conn, "cash_requests"):
        print("cash_requests table not found, skipping cashflow migration.")
        return
    if not column_exists(conn, "cash_requests", "created_by_user_id"):
        conn.execute("ALTER TABLE cash_requests ADD COLUMN created_by_user_id INTEGER NULL;")
    if not column_exists(conn, "cash_requests", "admin_user_id"):
        conn.execute("ALTER TABLE cash_requests ADD COLUMN admin_user_id INTEGER NULL;")
    if not column_exists(conn, "cash_request_participants", "user_id"):
        conn.execute("ALTER TABLE cash_request_participants ADD COLUMN user_id INTEGER NULL;")
    if not column_exists(conn, "cash_signatures", "user_id"):
        conn.execute("ALTER TABLE cash_signatures ADD COLUMN user_id INTEGER NULL;")

    mapping: Dict[int, int] = {}
    for row in conn.execute("SELECT id, telegram_id FROM users WHERE telegram_id IS NOT NULL;").fetchall():
        mapping[int(row["telegram_id"])] = int(row["id"])

    has_created_tid = column_exists(conn, "cash_requests", "created_by_telegram_id")
    has_admin_tid = column_exists(conn, "cash_requests", "admin_telegram_id")
    has_part_tid = column_exists(conn, "cash_request_participants", "telegram_id")
    has_sig_tid = column_exists(conn, "cash_signatures", "telegram_id")

    for tid, uid in mapping.items():
        if has_created_tid:
            conn.execute(
                "UPDATE cash_requests SET created_by_user_id=? WHERE created_by_user_id IS NULL AND created_by_telegram_id=?;",
                (uid, tid),
            )
        if has_admin_tid:
            conn.execute(
                "UPDATE cash_requests SET admin_user_id=? WHERE admin_user_id IS NULL AND admin_telegram_id=?;",
                (uid, tid),
            )
        if has_part_tid:
            conn.execute(
                "UPDATE cash_request_participants SET user_id=? WHERE user_id IS NULL AND telegram_id=?;",
                (uid, tid),
            )
        if has_sig_tid:
            conn.execute(
                "UPDATE cash_signatures SET user_id=? WHERE user_id IS NULL AND telegram_id=?;",
                (uid, tid),
            )


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if column_not_null(conn, "users", "telegram_id"):
            rebuild_users_table(conn)
        else:
            if not column_exists(conn, "users", "email"):
                conn.execute("ALTER TABLE users ADD COLUMN email TEXT;")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);")
            if not column_exists(conn, "users", "last_login_at"):
                conn.execute("ALTER TABLE users ADD COLUMN last_login_at TEXT;")
            if not column_exists(conn, "users", "password_changed_at"):
                conn.execute("ALTER TABLE users ADD COLUMN password_changed_at TEXT;")
        migrate_cashflow(conn)
        conn.commit()
        print("Migration completed.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
