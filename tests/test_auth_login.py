import importlib

import pytest
from fastapi.testclient import TestClient


def load_app(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("DB_PATH", str(db_path))
    app_module = importlib.import_module("app")
    importlib.reload(app_module)
    return app_module


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    return load_app(tmp_path, monkeypatch)


@pytest.fixture()
def client(app_module):
    return TestClient(app_module.APP)


@pytest.fixture()
def user_credentials(app_module):
    login = "tester"
    password = "secret123"
    now = app_module.iso_now(app_module.CFG.tzinfo())
    password_hash = app_module.hash_password(password)
    app_module.db_exec(
        """
        INSERT INTO users (login, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (login, password_hash, "admin", 1, now),
    )
    return login, password


def test_login_succeeds_without_existing_session(client, user_credentials):
    login, password = user_credentials
    res = client.post("/api/auth/login", json={"login": login, "password": password})
    assert res.status_code == 200
    assert res.cookies.get("session")
    assert res.cookies.get("csrf_token")


def test_login_with_existing_session_includes_csrf(client, user_credentials):
    login, password = user_credentials
    first = client.post("/api/auth/login", json={"login": login, "password": password})
    assert first.status_code == 200
    csrf_token = first.cookies.get("csrf_token")
    assert csrf_token

    res = client.post(
        "/api/auth/login",
        json={"login": login, "password": password},
        headers={"X-CSRF-Token": csrf_token},
        cookies=first.cookies,
    )
    assert res.status_code == 200


def test_login_with_existing_session_without_csrf_still_works(client, user_credentials):
    login, password = user_credentials
    first = client.post("/api/auth/login", json={"login": login, "password": password})
    assert first.status_code == 200

    res = client.post(
        "/api/auth/login",
        json={"login": login, "password": password},
        cookies={"session": first.cookies.get("session")},
    )
    assert res.status_code == 200


def test_register_creates_admin_for_first_user(client):
    res = client.post("/api/auth/register", json={"login": "owner", "password": "secret12"})
    assert res.status_code == 201
    data = res.json()
    assert data["user"]["role"] == "admin"
    assert res.cookies.get("session")


def test_register_creates_viewer_when_users_exist(client, user_credentials):
    res = client.post("/api/auth/register", json={"login": "viewer1", "password": "secret12"})
    assert res.status_code == 201
    data = res.json()
    assert data["user"]["role"] == "viewer"
