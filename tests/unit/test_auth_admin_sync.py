"""auth.ensure_admin_from_env: the admin account is created and kept in sync from
ADMIN_USERNAME / ADMIN_PASSWORD (.env) every time the app or the API starts."""
import pytest

import auth
from fake_db import FakeConn

USERNAME = "root_admin"
PASSWORD = "longenough1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)


@pytest.fixture
def admin_env(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", USERNAME)
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)


@pytest.fixture
def recorded(monkeypatch):
    """Replaces the two write helpers so the sync's DECISIONS can be asserted."""
    calls = []
    monkeypatch.setattr(auth, "update_user", lambda conn, username, **fields: calls.append(("update_user", username, fields)) or True)
    monkeypatch.setattr(auth, "set_password", lambda conn, username, password: calls.append(("set_password", username, password)) or True)
    return calls


def existing_account(password=PASSWORD):
    """A database that already has the admin row (its stored password hash)."""
    return FakeConn(responses=[("SELECT password_hash", [(auth.hash_password(password),)])])


# ---------- nothing configured: nothing happens ----------

def test_with_nothing_in_env_the_sync_does_nothing():
    conn = FakeConn()

    auth.ensure_admin_from_env(conn)

    assert conn.executed == []


def test_with_only_a_username_or_only_a_password_it_does_nothing(monkeypatch):
    conn = FakeConn()

    monkeypatch.setenv("ADMIN_USERNAME", USERNAME)
    auth.ensure_admin_from_env(conn)
    monkeypatch.delenv("ADMIN_USERNAME")
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    auth.ensure_admin_from_env(conn)

    assert conn.executed == []


def test_empty_strings_count_as_not_configured(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "")
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    conn = FakeConn()

    auth.ensure_admin_from_env(conn)

    assert conn.executed == []


# ---------- the account does not exist yet: it is created as an admin ----------

def test_a_missing_account_is_created_as_an_admin_with_a_hashed_password(admin_env):
    conn = FakeConn()  # no row comes back for the lookup: the account does not exist

    auth.ensure_admin_from_env(conn)

    _, username, stored_hash, role = conn.find("INSERT INTO")[0][1]
    assert username == USERNAME and role == "admin"
    assert stored_hash.startswith("scrypt$") and PASSWORD not in stored_hash
    assert auth.verify_password(PASSWORD, stored_hash) is True
    assert conn.commits == 1


def test_the_lookup_is_by_the_configured_username(admin_env):
    conn = FakeConn()

    auth.ensure_admin_from_env(conn)

    sql, params = conn.find("SELECT password_hash")[0]
    assert params == (USERNAME,) and "WHERE username = %s" in sql


def test_an_invalid_username_in_env_is_refused_and_nothing_is_stored(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "no")  # under the 3-character minimum
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    conn = FakeConn()

    with pytest.raises(auth.AuthError):
        auth.ensure_admin_from_env(conn)

    assert conn.find("INSERT INTO") == []


def test_a_too_short_password_in_env_is_refused_and_nothing_is_stored(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", USERNAME)
    monkeypatch.setenv("ADMIN_PASSWORD", "short")
    conn = FakeConn()

    with pytest.raises(auth.AuthError):
        auth.ensure_admin_from_env(conn)

    assert conn.find("INSERT INTO") == []


# ---------- the account exists: it is forced back to what .env says ----------

def test_an_existing_account_is_forced_to_admin_and_active(admin_env, recorded):
    auth.ensure_admin_from_env(existing_account())

    assert ("update_user", USERNAME, {"role": "admin", "is_active": True}) in recorded


def test_a_matching_password_is_left_alone_so_logins_are_not_revoked(admin_env, recorded):
    auth.ensure_admin_from_env(existing_account(PASSWORD))

    assert [call[0] for call in recorded] == ["update_user"]  # no set_password


def test_a_changed_password_in_env_is_synced_to_the_database(admin_env, recorded):
    auth.ensure_admin_from_env(existing_account("the-old-password"))

    assert ("set_password", USERNAME, PASSWORD) in recorded


def test_the_role_and_active_flag_are_restored_before_the_password_check(admin_env, recorded):
    auth.ensure_admin_from_env(existing_account("the-old-password"))

    assert [call[0] for call in recorded] == ["update_user", "set_password"]


def test_running_the_sync_twice_changes_nothing_the_second_time(admin_env, recorded):
    conn = existing_account(PASSWORD)

    auth.ensure_admin_from_env(conn)
    auth.ensure_admin_from_env(conn)

    assert [call[0] for call in recorded] == ["update_user", "update_user"]  # never a password rewrite


def test_a_demoted_or_deactivated_admin_is_restored_on_the_next_start(admin_env, recorded):
    # the stored row could say role='user' / is_active=false: the sync does not look, it just enforces
    auth.ensure_admin_from_env(existing_account())

    _, _, fields = recorded[0]
    assert fields == {"role": "admin", "is_active": True}


# ---------- secrets are never printed ----------

def test_the_password_is_never_printed_or_logged(admin_env, recorded, capsys):
    auth.ensure_admin_from_env(existing_account("the-old-password"))
    auth.ensure_admin_from_env(FakeConn())

    captured = capsys.readouterr()
    assert PASSWORD not in captured.out and PASSWORD not in captured.err
