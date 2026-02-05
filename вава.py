import sqlite3
from datetime import datetime

DB_PATH = "db.sqlite3"  # путь к БД

ADMIN = {
    "telegram_id": 1677321068,
    "username": "Nikitaiterman",
    "first_name": "Никита",
    "last_name": "Итерман",
    "name": "Итерман Н.С.",
    "role": "admin",
}

def iso_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    now = iso_now()

    # Проверяем, есть ли уже пользователь
    existing = conn.execute(
        "SELECT * FROM users WHERE telegram_id = ?;",
        (ADMIN["telegram_id"],)
    ).fetchone()

    if existing:
        print("⚠️ Админ уже существует — обновляем роль и активность")
        conn.execute(
            """
            UPDATE users
            SET
                role = ?,
                is_active = 1,
                updated_at = ?
            WHERE telegram_id = ?;
            """,
            (ADMIN["role"], now, ADMIN["telegram_id"]),
        )
    else:
        print("✅ Создаём нового администратора")
        conn.execute(
            """
            INSERT INTO users (
                telegram_id,
                username,
                first_name,
                last_name,
                login,
                name,
                role,
                is_active,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                ADMIN["telegram_id"],
                ADMIN["username"],
                ADMIN["first_name"],
                ADMIN["last_name"],
                f"tg_{ADMIN['telegram_id']}",
                ADMIN["name"],
                ADMIN["role"],
                1,
                now,
                now,
            ),
        )

    conn.commit()
    conn.close()
    print("🎉 Готово")

if __name__ == "__main__":
    main()
