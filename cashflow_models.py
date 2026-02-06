"""cashflow_models.py

Модуль данных/логики для подтверждения наличных (сбор/изъятие) с подписями.

Дизайн:
- 3 счёта: main, praise, alpha
- 2 типа операций: collect (сбор/внесение), withdraw (изъятие)
- Участники подтверждения берутся из users.json (роль cash_signer) + обязательный admin.
- Подписант может подписать (вирт. подпись PNG) или отказать (причина).
- Админ может отправить повторно на подпись (attempt=2) отказавшим.
- Второй отказ фиксируется как финальный "ОТКАЗ" (участие закрыто).
- Админ обязан подписать (вирт. подпись) перед финализацией.

Интеграция:
- В app.py нужно вызвать init_cashflow_db() внутри init_db().
- В routes при создании/обновлении нужно звать функции этого модуля.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ACCOUNTS: Tuple[str, ...] = ("main", "praise", "alpha")
OP_TYPES: Tuple[str, ...] = ("collect", "withdraw")


@dataclasses.dataclass(frozen=True)
class CashflowConfig:
    base_dir: Path
    db_path: Path
    uploads_dir: Path
    timezone: str = "Europe/Warsaw"


def load_cashflow_config(base_dir: Optional[Path] = None) -> CashflowConfig:
    """Загружает конфиг, совместимый с app.py (.env)."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    db_path = Path(os.getenv("DB_PATH", "db.sqlite3").strip() or "db.sqlite3")
    if not db_path.is_absolute():
        db_path = base_dir / db_path



    uploads_dir = base_dir / "uploads" / "cashflow"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    tz = os.getenv("TIMEZONE", "Europe/Warsaw").strip() or "Europe/Warsaw"

    return CashflowConfig(
        base_dir=base_dir,
        db_path=db_path,
        uploads_dir=uploads_dir,
        timezone=tz,
    )


def db_connect(cfg: CashflowConfig) -> sqlite3.Connection:
    conn = sqlite3.connect(str(cfg.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(str(r[1]) == column for r in rows)


def iso_now() -> str:
    # В проекте уже есть TZ-логика, но для модуля достаточно ISO UTC.
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def init_cashflow_db(conn: sqlite3.Connection) -> None:
    """Создаёт таблицы cashflow, если их нет."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cash_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account TEXT NOT NULL,
            op_type TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            created_by_user_id INTEGER NULL,
            admin_user_id INTEGER NOT NULL,
            admin_comment TEXT NULL,
            source_kind TEXT NULL,
            source_id INTEGER NULL,
            source_payload TEXT NULL,            
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cash_requests_status
            ON cash_requests(status, account, op_type);
        CREATE INDEX IF NOT EXISTS idx_cash_requests_source
            ON cash_requests(source_kind, source_id);

        CREATE TABLE IF NOT EXISTS cash_request_participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            name_snapshot TEXT NOT NULL,
            role_snapshot TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(request_id, user_id),
            FOREIGN KEY (request_id) REFERENCES cash_requests(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS cash_signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            attempt INTEGER NOT NULL,
            decision TEXT NOT NULL,
            refuse_reason TEXT NULL,
            signature_path TEXT NULL,
            signed_at TEXT NOT NULL,
            UNIQUE(request_id, user_id, attempt),
            FOREIGN KEY (request_id) REFERENCES cash_requests(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_cash_signatures_req
            ON cash_signatures(request_id);
        """
    )
    if not _table_has_column(conn, "cash_requests", "source_payload"):
        conn.execute("ALTER TABLE cash_requests ADD COLUMN source_payload TEXT NULL;")
        conn.commit()
    if not _table_has_column(conn, "cash_requests", "created_by_user_id"):
        conn.execute("ALTER TABLE cash_requests ADD COLUMN created_by_user_id INTEGER NULL;")
        conn.commit()
    if not _table_has_column(conn, "cash_requests", "admin_user_id"):
        conn.execute("ALTER TABLE cash_requests ADD COLUMN admin_user_id INTEGER NULL;")
        conn.commit()
    if not _table_has_column(conn, "cash_request_participants", "user_id"):
        conn.execute("ALTER TABLE cash_request_participants ADD COLUMN user_id INTEGER NULL;")
        conn.commit()
    if not _table_has_column(conn, "cash_signatures", "user_id"):
        conn.execute("ALTER TABLE cash_signatures ADD COLUMN user_id INTEGER NULL;")
        conn.commit()
    conn.commit()




def _normalize_account(account: str) -> str:
    a = (account or "").strip().lower()
    if a not in ACCOUNTS:
        raise ValueError(f"Invalid account: {account}")
    return a


def _normalize_op_type(op_type: str) -> str:
    t = (op_type or "").strip().lower()
    if t not in OP_TYPES:
        raise ValueError(f"Invalid op_type: {op_type}")
    return t


def _user_is_active(row: Dict[str, Any]) -> bool:
    if "is_active" in row:
        value = row.get("is_active")
        return bool(value == 1 or value is True)
    if "active" in row:
        value = row.get("active")
        return bool(value == 1 or value is True)
    return True


def _user_display_name(row: Dict[str, Any]) -> str:
    name = str(row.get("name") or "").strip()
    if name:
        return name
    first = str(row.get("first_name") or "").strip()
    last = str(row.get("last_name") or "").strip()
    if first or last:
        return " ".join(part for part in (first, last) if part).strip()
    username = str(row.get("username") or "").strip()
    if username:
        return username
    login = str(row.get("login") or "").strip()
    if login:
        return login
    return str(row.get("id") or "unknown")


def _load_users_by_role(conn: sqlite3.Connection, role: str) -> Dict[int, Dict[str, Any]]:
    rows = conn.execute("SELECT * FROM users WHERE role=?;", (str(role),)).fetchall()
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        data = dict(row)
        if not _user_is_active(data):
            continue
        uid = int(data.get("id") or 0)
        if uid:
            out[uid] = data
    return out


def _split_scope_values(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [p.strip().lower() for p in re.split(r"[,\s]+", str(raw)) if p.strip()]


def _user_allows_cash(user_row: Dict[str, Any], account: str, op_type: str) -> bool:
    scopes = set(_split_scope_values(user_row.get("cash_scopes")))
    ops = set(_split_scope_values(user_row.get("cash_ops")))
    if scopes and account not in scopes:
        return False
    if ops and op_type not in ops:
        return False
    return True


def pick_primary_admin(conn: sqlite3.Connection) -> int:
    admins = _load_users_by_role(conn, "admin")
    if not admins:
        admins = _load_users_by_role(conn, "owner")
    if not admins:
        raise RuntimeError("No active admin in users table")
    return int(sorted(admins)[0])


def pick_cash_signers(conn: sqlite3.Connection, account: str, op_type: str) -> List[int]:
    """Возвращает список user_id подписантов.

    Для подписей наличных по суммам не ограничиваем по cash_scopes/cash_ops,
    чтобы активные подписанты всегда могли подписывать заявки.
    """
    _normalize_account(account)
    _normalize_op_type(op_type)
    signers = _load_users_by_role(conn, "cash_signer")
    allowed = [
        uid for uid, row in signers.items() if _user_allows_cash(row, account, op_type)
    ]
    return sorted(allowed)


def create_cash_request(
    conn: sqlite3.Connection,
    cfg: CashflowConfig,
    *,
    account: str,
    op_type: str,
    amount: float,
    created_by_user_id: Optional[int],
    source_kind: Optional[str] = None,
    source_id: Optional[int] = None,
    source_payload: Optional[Dict[str, Any]] = None,
) -> int:
    """Создаёт запрос на подпись + фиксирует состав участников."""
    account_n = _normalize_account(account)
    op_type_n = _normalize_op_type(op_type)
    if amount is None or float(amount) <= 0:
        raise ValueError("amount must be > 0")

    admin_uid = pick_primary_admin(conn)
    signers = pick_cash_signers(conn, account_n, op_type_n)
    # admin обязан подписывать, но может быть и в signers — дедуп.
    participants = list(dict.fromkeys(signers + [admin_uid]))

    now = iso_now()
    payload_json = None
    if source_payload is not None:
        payload_json = json.dumps(source_payload, ensure_ascii=False, sort_keys=True)
    cur = conn.execute(
        """
        INSERT INTO cash_requests (
          account, op_type, amount, status, attempt,
          created_by_user_id, admin_user_id,
          source_kind, source_id, source_payload,
          created_at, updated_at
        ) VALUES (?, ?, ?, 'PENDING_SIGNERS', 1, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            account_n,
            op_type_n,
            float(amount),
            int(created_by_user_id) if created_by_user_id else None,
            int(admin_uid),
            str(source_kind) if source_kind else None,
            int(source_id) if source_id is not None else None,
            payload_json,
            now,
            now,
        ),
    )
    request_id = int(cur.lastrowid)

    users_map: Dict[int, Dict[str, Any]] = {}
    users_map.update(_load_users_by_role(conn, "admin"))
    users_map.update(_load_users_by_role(conn, "owner"))
    users_map.update(_load_users_by_role(conn, "cash_signer"))
    for uid in participants:
        u = users_map.get(int(uid), {})
        name = _user_display_name(u)
        role = str(u.get("role") or "unknown")
        is_admin = 1 if int(uid) == int(admin_uid) else 0
        conn.execute(
            """
            INSERT OR IGNORE INTO cash_request_participants
              (request_id, user_id, name_snapshot, role_snapshot, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (request_id, int(uid), name, role, is_admin, now),
        )

    conn.commit()
    return request_id


def _fetchone(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchone()


def _fetchall(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return list(cur.fetchall())


def get_cash_request(conn: sqlite3.Connection, request_id: int) -> sqlite3.Row:
    row = _fetchone(conn, "SELECT * FROM cash_requests WHERE id=?;", (int(request_id),))
    if not row:
        raise KeyError("cash_request not found")
    return row


def list_cash_requests(
    conn: sqlite3.Connection,
    *,
    account: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[sqlite3.Row]:
    where: List[str] = []
    params: List[Any] = []
    if account:
        where.append("account=?")
        params.append(_normalize_account(account))
    if status:
        where.append("status=?")
        params.append(str(status))
    w = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM cash_requests{w} ORDER BY id DESC LIMIT ? OFFSET ?;"
    params.extend([int(limit), int(offset)])
    return _fetchall(conn, sql, params)


def list_my_cash_requests(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    account: Optional[str] = None,
    only_open: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> List[sqlite3.Row]:
    where: List[str] = ["p.user_id=?"]
    params: List[Any] = [int(user_id)]
    if account:
        where.append("LOWER(r.account)=?")
        params.append(_normalize_account(account))
    if only_open:
        where.append("r.status IN ('PENDING_SIGNERS','PENDING_ADMIN')")
    w = " AND ".join(where)
    sql = (
        "SELECT r.* FROM cash_requests r "
        "JOIN cash_request_participants p ON p.request_id=r.id "
        f"WHERE {w} "
        "ORDER BY r.id DESC LIMIT ? OFFSET ?;"
    )
    params.extend([int(limit), int(offset)])
    return _fetchall(conn, sql, params)


def get_request_participants(conn: sqlite3.Connection, request_id: int) -> List[sqlite3.Row]:
    return _fetchall(
        conn,
        "SELECT * FROM cash_request_participants WHERE request_id=? ORDER BY is_admin ASC, id ASC;",
        (int(request_id),),
    )


def get_request_signatures(conn: sqlite3.Connection, request_id: int) -> List[sqlite3.Row]:
    return _fetchall(
        conn,
        "SELECT * FROM cash_signatures WHERE request_id=? ORDER BY attempt ASC, id ASC;",
        (int(request_id),),
    )


def _effective_participant_state(
    sig_attempt1: Optional[sqlite3.Row],
    sig_attempt2: Optional[sqlite3.Row],
) -> Tuple[str, Optional[str]]:
    """Возвращает (state, detail) для UI/логики."""
    if sig_attempt2:
        if sig_attempt2["decision"] == "SIGNED":
            return "SIGNED", None
        # attempt2 REFUSED -> финальный отказ
        reason = sig_attempt2["refuse_reason"]
        return "REFUSED_FINAL", str(reason) if reason else None

    if sig_attempt1:
        if sig_attempt1["decision"] == "SIGNED":
            return "SIGNED", None
        reason = sig_attempt1["refuse_reason"]
        return "REFUSED_NEEDS_RETRY", str(reason) if reason else None

    return "PENDING", None


def build_request_view(conn: sqlite3.Connection, request_id: int) -> Dict[str, Any]:
    r = get_cash_request(conn, request_id)
    participants = get_request_participants(conn, request_id)
    sigs = get_request_signatures(conn, request_id)

    sig_map: Dict[Tuple[int, int], sqlite3.Row] = {}
    for s in sigs:
        sig_map[(int(s["user_id"]), int(s["attempt"]))] = s

    part_views: List[Dict[str, Any]] = []
    for p in participants:
        uid = int(p["user_id"])
        s1 = sig_map.get((uid, 1))
        s2 = sig_map.get((uid, 2))
        state, detail = _effective_participant_state(s1, s2)
        part_views.append(
            {
                "user_id": uid,
                "name": p["name_snapshot"],
                "role": p["role_snapshot"],
                "is_admin": bool(int(p["is_admin"]) == 1),
                "state": state,
                "detail": detail,
                "attempt1": dict(s1) if s1 else None,
                "attempt2": dict(s2) if s2 else None,
            }
        )

    return {"request": dict(r), "participants": part_views}


def _ensure_participant(conn: sqlite3.Connection, request_id: int, user_id: int) -> sqlite3.Row:
    p = _fetchone(
        conn,
        "SELECT * FROM cash_request_participants WHERE request_id=? AND user_id=?;",
        (int(request_id), int(user_id)),
    )
    if not p:
        raise PermissionError("User is not a participant")
    return p


def _decode_data_url_png(data_url: str) -> bytes:
    if not data_url:
        raise ValueError("signature is required")
    m = re.match(r"^data:image/png;base64,(.+)$", data_url.strip())
    if not m:
        # допускаем "голый" base64
        b64 = data_url.strip()
    else:
        b64 = m.group(1)
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 signature") from e


def _save_signature_png(cfg: CashflowConfig, request_id: int, user_id: int, attempt: int, png_bytes: bytes) -> str:
    req_dir = cfg.uploads_dir / f"req_{int(request_id)}"
    req_dir.mkdir(parents=True, exist_ok=True)
    name = f"sig_{int(user_id)}_a{int(attempt)}_{uuid.uuid4().hex}.png"
    path = req_dir / name
    path.write_bytes(png_bytes)
    # храним относительный путь от uploads_dir
    rel = str(path.relative_to(cfg.uploads_dir))
    return rel


def record_signature(
    conn: sqlite3.Connection,
    cfg: CashflowConfig,
    *,
    request_id: int,
    user_id: int,
    signature_data_url: str,
    as_admin: bool = False,
) -> None:
    """Записывает подпись пользователя (SIGNED)."""
    r = get_cash_request(conn, request_id)
    if r["status"] in ("FINAL", "CANCELLED"):
        raise ValueError("Request is closed")
    p = _ensure_participant(conn, request_id, user_id)
    if as_admin and int(p["is_admin"]) != 1:
        raise PermissionError("Not admin participant")

    attempt = int(r["attempt"])
    # если уже есть запись на этой попытке — запрещаем
    existing = _fetchone(
        conn,
        "SELECT * FROM cash_signatures WHERE request_id=? AND user_id=? AND attempt=?;",
        (int(request_id), int(user_id), int(attempt)),
    )
    if existing:
        raise ValueError("Already decided on this attempt")

    png = _decode_data_url_png(signature_data_url)
    png = normalize_signature_png_bytes(png)
    rel_path = _save_signature_png(cfg, request_id, user_id, attempt, png)
    now = iso_now()
    conn.execute(
        """
        INSERT INTO cash_signatures (request_id, user_id, attempt, decision, refuse_reason, signature_path, signed_at)
        VALUES (?, ?, ?, 'SIGNED', NULL, ?, ?);
        """,
        (int(request_id), int(user_id), int(attempt), rel_path, now),
    )
    conn.execute(
        "UPDATE cash_requests SET updated_at=? WHERE id=?;",
        (now, int(request_id)),
    )
    recompute_request_status(conn, request_id)
    conn.commit()


def record_refusal(
    conn: sqlite3.Connection,
    *,
    request_id: int,
    user_id: int,
    reason: str,
) -> None:
    """Записывает отказ пользователя (REFUSED)."""
    r = get_cash_request(conn, request_id)
    if r["status"] in ("FINAL", "CANCELLED"):
        raise ValueError("Request is closed")
    p = _ensure_participant(conn, request_id, user_id)
    if int(p["is_admin"]) == 1:
        raise PermissionError("Admin cannot refuse")
    attempt = int(r["attempt"])
    if not reason or not str(reason).strip():
        raise ValueError("Refuse reason is required")

    existing = _fetchone(
        conn,
        "SELECT * FROM cash_signatures WHERE request_id=? AND user_id=? AND attempt=?;",
        (int(request_id), int(user_id), int(attempt)),
    )
    if existing:
        raise ValueError("Already decided on this attempt")

    now = iso_now()
    conn.execute(
        """
        INSERT INTO cash_signatures (request_id, user_id, attempt, decision, refuse_reason, signature_path, signed_at)
        VALUES (?, ?, ?, 'REFUSED', ?, NULL, ?);
        """,
        (int(request_id), int(user_id), int(attempt), str(reason).strip(), now),
    )
    conn.execute(
        "UPDATE cash_requests SET updated_at=? WHERE id=?;",
        (now, int(request_id)),
    )
    recompute_request_status(conn, request_id)
    conn.commit()


def resend_for_refusals(
    conn: sqlite3.Connection,
    *,
    request_id: int,
    admin_user_id: int,
    target_user_ids: Optional[List[int]] = None,
    admin_comment: Optional[str] = None,
) -> List[int]:
    """Переводит запрос на attempt=2 и возвращает список подписантов, кому нужно отправить уведомление."""
    r = get_cash_request(conn, request_id)
    if int(r["admin_user_id"]) != int(admin_user_id):
        # допускаем что админов несколько, но ответственным назначен один
        raise PermissionError("Only primary admin can resend")
    if r["status"] in ("FINAL", "CANCELLED"):
        raise ValueError("Request is closed")

    # вычисляем отказавших на attempt1
    sigs1 = _fetchall(
        conn,
        "SELECT user_id FROM cash_signatures WHERE request_id=? AND attempt=1 AND decision='REFUSED';",
        (int(request_id),),
    )
    refused1 = {int(s["user_id"]) for s in sigs1}
    if not refused1:
        raise ValueError("No refusals on attempt 1")

    if target_user_ids is None or len(target_user_ids) == 0:
        targets = sorted(refused1)
    else:
        targets = sorted(set(int(x) for x in target_user_ids) & refused1)
        if not targets:
            raise ValueError("No valid targets among refused signers")

    now = iso_now()
    conn.execute(
        "UPDATE cash_requests SET attempt=2, admin_comment=?, updated_at=? WHERE id=?;",
        (str(admin_comment).strip() if admin_comment else None, now, int(request_id)),
    )
    conn.commit()
    return targets


def cancel_request(conn: sqlite3.Connection, *, request_id: int, admin_user_id: int, comment: Optional[str] = None) -> None:
    r = get_cash_request(conn, request_id)
    if int(r["admin_user_id"]) != int(admin_user_id):
        raise PermissionError("Only primary admin can cancel")
    if r["status"] in ("FINAL", "CANCELLED"):
        return
    now = iso_now()
    conn.execute(
        "UPDATE cash_requests SET status='CANCELLED', admin_comment=?, updated_at=? WHERE id=?;",
        (str(comment).strip() if comment else r["admin_comment"], now, int(request_id)),
    )
    conn.commit()


def recompute_request_status(conn: sqlite3.Connection, request_id: int) -> None:
    """Пересчитывает статус запроса по текущим подписям."""
    r = get_cash_request(conn, request_id)
    if r["status"] in ("FINAL", "CANCELLED"):
        return

    participants = get_request_participants(conn, request_id)
    sigs = get_request_signatures(conn, request_id)
    sig_map: Dict[Tuple[int, int], sqlite3.Row] = {(int(s["user_id"]), int(s["attempt"])): s for s in sigs}

    all_non_admin_done = True
    admin_signed = False
    for p in participants:
        uid = int(p["user_id"])
        is_admin = int(p["is_admin"]) == 1
        s1 = sig_map.get((uid, 1))
        s2 = sig_map.get((uid, 2))
        state, _detail = _effective_participant_state(s1, s2)

        if is_admin:
            # админ считается подписавшим только если есть SIGNED (любая попытка)
            if state == "SIGNED":
                admin_signed = True
            continue

        if state == "PENDING":
            all_non_admin_done = False
        elif state == "REFUSED_NEEDS_RETRY":
            # отказ на попытке 1 не закрывает участие
            all_non_admin_done = False
        else:
            # SIGNED или REFUSED_FINAL
            pass

    new_status = r["status"]
    if all_non_admin_done and not admin_signed:
        new_status = "PENDING_ADMIN"
    elif all_non_admin_done and admin_signed:
        new_status = "FINAL"
    else:
        new_status = "PENDING_SIGNERS"

    if new_status != r["status"]:
        now = iso_now()
        conn.execute(
            "UPDATE cash_requests SET status=?, updated_at=? WHERE id=?;",
            (new_status, now, int(request_id)),
        )


def get_signature_file_path(cfg: CashflowConfig, signature_path: str) -> Path:
    # signature_path хранится относительным от cfg.uploads_dir
    p = (cfg.uploads_dir / signature_path).resolve()
    # safety: запрет выхода за uploads_dir
    if cfg.uploads_dir.resolve() not in p.parents and p != cfg.uploads_dir.resolve():
        raise ValueError("Invalid signature path")
    return p


def list_withdraw_act_rows(
    conn: sqlite3.Connection,
    *,
    account: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Возвращает строки акта изъятия/сбора наличных (по участникам) + статус подписи.

    signature_value:
      - "SIGNED" -> есть подпись (signature_path может быть None, если файл потерян)
      - "ОТКАЗ: причина"
      - "Ожидает подписи"
    signature_path:
      - относительный путь PNG (attempt=2 если есть, иначе attempt=1)
    participant_user_id:
      - user_id участника (нужно для загрузки PNG подписи через API)

    ВАЖНО:
      - date возвращаем ТОЛЬКО дату YYYY-MM-DD (без времени)
      - amount возвращаем числом, округлённым до 2 знаков
    """

    def _to_iso_lower_bound(v: str) -> str:
        """YYYY-MM-DD -> YYYY-MM-DDT00:00:00Z, ISO оставляем как есть (подправляем Z при необходимости)."""
        s = (v or "").strip()
        if not s:
            return s
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return f"{s}T00:00:00Z"
        # если вдруг без Z, но ISO — оставим как есть
        return s

    def _to_iso_upper_bound(v: str) -> str:
        """YYYY-MM-DD -> YYYY-MM-DDT23:59:59Z, ISO оставляем как есть."""
        s = (v or "").strip()
        if not s:
            return s
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return f"{s}T23:59:59Z"
        return s

    def _date_only(created_at: Any) -> str:
        """Из created_at (ISO) делаем YYYY-MM-DD."""
        raw = str(created_at or "").strip()
        if not raw:
            return ""
        # быстрый путь
        if len(raw) >= 10:
            d10 = raw[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d10):
                return d10
        # попытка распарсить
        try:
            s = raw.replace("Z", "+00:00")
            return dt.datetime.fromisoformat(s).date().isoformat()
        except Exception:
            return raw[:10] if len(raw) >= 10 else raw

    params: List[Any] = []
    where = "WHERE 1=1 "

    if account:
        where += " AND r.account = ?"
        params.append(_normalize_account(account))

    if date_from:
        where += " AND r.created_at >= ?"
        params.append(_to_iso_lower_bound(str(date_from)))

    if date_to:
        where += " AND r.created_at <= ?"
        params.append(_to_iso_upper_bound(str(date_to)))

    rows = conn.execute(
        f"""
        SELECT
            r.id AS request_id,
            r.account,
            r.op_type,
            r.amount,
            r.created_at,
            p.user_id AS participant_user_id,
            p.name_snapshot,
            p.role_snapshot,
            p.is_admin,
            s1.decision AS sig1_decision,
            s1.refuse_reason AS sig1_reason,
            s1.signature_path AS sig1_path,
            s2.decision AS sig2_decision,
            s2.refuse_reason AS sig2_reason,
            s2.signature_path AS sig2_path
        FROM cash_requests r
        JOIN cash_request_participants p ON p.request_id = r.id
        LEFT JOIN cash_signatures s1
          ON s1.request_id = r.id AND s1.user_id = p.user_id AND s1.attempt = 1
        LEFT JOIN cash_signatures s2
          ON s2.request_id = r.id AND s2.user_id = p.user_id AND s2.attempt = 2
        {where}
        ORDER BY r.created_at DESC, r.id DESC, p.is_admin ASC, p.id ASC
        """,
        params,
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for row in rows:
        # attempt=2 важнее, если есть
        sig_dec = row["sig2_decision"] or row["sig1_decision"]
        sig_reason = row["sig2_reason"] or row["sig1_reason"]
        sig_path = row["sig2_path"] or row["sig1_path"]

        if sig_dec == "REFUSED":
            signature_value = f"ОТКАЗ: {sig_reason or 'без причины'}"
        elif sig_dec == "SIGNED":
            signature_value = "SIGNED"
        else:
            signature_value = "Ожидает подписи"

        amount_val = round(float(row["amount"] or 0), 2)

        out.append(
            {
                "request_id": int(row["request_id"]),
                "participant_user_id": int(row["participant_user_id"]),
                "account": row["account"],
                "op_type": row["op_type"],
                "amount": amount_val,                 # число, округлено до 2 знаков
                "date": _date_only(row["created_at"]),# ТОЛЬКО YYYY-MM-DD
                "fio": row["name_snapshot"],
                "user_type": row["role_snapshot"],
                "signature_value": signature_value,
                "signature_path": sig_path,
            }
        )
    return out


def normalize_signature_png_bytes(png_bytes: bytes) -> bytes:
    """
    Нормализует PNG подписи для вставки в Excel:
    - убирает прозрачные поля (crop по альфе)
    - добавляет небольшой padding
    - композитит на белый фон (Excel иногда капризничает с прозрачностью)
    Если Pillow недоступен/PNG битый — возвращает исходные bytes.
    """
    try:
        import io as _io
        from PIL import Image  # pip install pillow
    except Exception:
        return png_bytes

    try:
        im = Image.open(_io.BytesIO(png_bytes))
        im.load()
    except Exception:
        return png_bytes

    try:
        if im.mode != "RGBA":
            im = im.convert("RGBA")


        alpha = im.split()[-1]
        alpha_extrema = alpha.getextrema()
        if alpha_extrema == (255, 255):
            # Канвас без прозрачности: строим маску по "небелым" пикселям
            rgb = im.convert("RGB")
            mask = Image.new("L", im.size, 0)
            px = rgb.load()
            mask_px = mask.load()
            for y in range(im.size[1]):
                for x in range(im.size[0]):
                    r, g, b = px[x, y]
                    if r < 250 or g < 250 or b < 250:
                        mask_px[x, y] = 255
            alpha = mask

        # crop по альфе/маске (убираем пустые поля)
        bbox = alpha.getbbox()
        if bbox:
            im = im.crop(bbox)
            alpha = alpha.crop(bbox)

        # padding
        pad = 8
        w, h = im.size
        canvas = Image.new("RGBA", (w + pad * 2, h + pad * 2), (255, 255, 255, 0))
        canvas.paste(im, (pad, pad), im)
        im = canvas
        alpha = alpha.crop((0, 0, w, h))
        padded_alpha = Image.new("L", (w + pad * 2, h + pad * 2), 0)
        padded_alpha.paste(alpha, (pad, pad))
        alpha = padded_alpha

        # нормализуем цвет штрихов в синий для контраста
        blue = Image.new("RGBA", im.size, (0, 74, 173, 0))
        blue.putalpha(alpha)

        # композит на белый фон (RGB)
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(blue, mask=alpha)

        out = _io.BytesIO()
        bg.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes
