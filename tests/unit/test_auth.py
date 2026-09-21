import pytest

import auth
from fake_db import FakeConn


def test_hash_password_format_and_verification():
    stored = auth.hash_password("correct horse")

    assert stored.startswith("scrypt$")
    assert "correct horse" not in stored
    assert auth.verify_password("correct horse", stored) is True
    assert auth.verify_password("wrong horse", stored) is False


def test_same_password_hashes_differently_because_of_salt():
    assert auth.hash_password("samepassword") != auth.hash_password("samepassword")


def test_short_password_rejected():
    with pytest.raises(auth.AuthError):
        auth.hash_password("short")


def test_verify_password_handles_malformed_and_unknown_schemes():
    assert auth.verify_password("x", "not-a-hash") is False
    assert auth.verify_password("x", "bcrypt$1$2$3$aa$bb") is False
    assert auth.verify_password("x", "scrypt$notanumber$8$1$aa$bb") is False


def test_hash_token_is_deterministic_and_not_the_token():
    assert auth.hash_token("abc") == auth.hash_token("abc")
    assert auth.hash_token("abc") != "abc"
    assert len(auth.hash_token("abc")) == 64


def test_lockout_after_repeated_failures_then_expires():
    name = "lockout-user"
    auth.clear_failed_logins(name)
    for i in range(auth.MAX_FAILED_LOGINS):
        auth.check_lockout(name, now=100.0 + i)
        auth.record_failed_login(name, now=100.0 + i)

    with pytest.raises(auth.LockedOut) as excinfo:
        auth.check_lockout(name, now=110.0)
    assert excinfo.value.retry_after > 0

    auth.check_lockout(name, now=100.0 + auth.LOCKOUT_SECONDS + 10)  # window passed: no raise
    auth.clear_failed_logins(name)


def test_clear_failed_logins_lifts_lockout():
    name = "clear-user"
    for _ in range(auth.MAX_FAILED_LOGINS):
        auth.record_failed_login(name, now=1.0)
    auth.clear_failed_logins(name)
    auth.check_lockout(name, now=2.0)


def test_create_user_validates_before_touching_the_database():
    conn = FakeConn()
    with pytest.raises(auth.AuthError):
        auth.create_user(conn, "no spaces allowed", "longenough1")
    with pytest.raises(auth.AuthError):
        auth.create_user(conn, "ab", "longenough1")
    with pytest.raises(auth.AuthError):
        auth.create_user(conn, "gooduser", "longenough1", role="superuser")
    with pytest.raises(auth.AuthError):
        auth.create_user(conn, "gooduser", "short")
    assert conn.executed == []


def test_create_user_stores_only_a_hash():
    conn = FakeConn()

    user = auth.create_user(conn, "alice", "supersecret1")

    assert user["username"] == "alice" and user["role"] == "user"
    _, params = conn.find("INSERT INTO")[0]
    assert params[1] == "alice"
    assert params[2].startswith("scrypt$") and params[2] != "supersecret1"
    assert conn.commits == 1


def _user_row(password="supersecret1", active=True):
    return ("uid-1", auth.hash_password(password), active)


def test_authenticate_success_returns_token_and_stores_only_its_hash():
    auth.clear_failed_logins("bob")
    conn = FakeConn(responses=[("users WHERE username", [_user_row()])])

    token = auth.authenticate(conn, "bob", "supersecret1")

    assert len(token) > 20
    _, params = conn.find("INSERT INTO")[0]
    assert params[0] == auth.hash_token(token)
    assert token not in params
    assert conn.commits == 1


def test_authenticate_wrong_password_is_generic_and_counts_a_failure():
    name = "carol"
    auth.clear_failed_logins(name)
    conn = FakeConn(responses=[("users WHERE username", [_user_row()])])

    with pytest.raises(auth.AuthError, match="Invalid username or password"):
        auth.authenticate(conn, name, "wrongpassword")

    assert conn.find("INSERT INTO") == []
    auth.clear_failed_logins(name)


def test_authenticate_unknown_and_inactive_users_fail_the_same_way():
    auth.clear_failed_logins("ghost")
    auth.clear_failed_logins("dave")

    with pytest.raises(auth.AuthError, match="Invalid username or password"):
        auth.authenticate(FakeConn(), "ghost", "supersecret1")

    inactive = FakeConn(responses=[("users WHERE username", [_user_row(active=False)])])
    with pytest.raises(auth.AuthError, match="Invalid username or password"):
        auth.authenticate(inactive, "dave", "supersecret1")

    auth.clear_failed_logins("ghost")
    auth.clear_failed_logins("dave")


def test_authenticate_is_blocked_while_locked_out():
    name = "locked-out"
    for _ in range(auth.MAX_FAILED_LOGINS):
        auth.record_failed_login(name)
    conn = FakeConn(responses=[("users WHERE username", [_user_row()])])

    with pytest.raises(auth.LockedOut):
        auth.authenticate(conn, name, "supersecret1")

    assert conn.executed == []  # refused before any database work
    auth.clear_failed_logins(name)


def test_get_user_by_token_none_for_empty_token_without_db_call():
    conn = FakeConn()
    assert auth.get_user_by_token(conn, "") is None
    assert auth.get_user_by_token(conn, None) is None
    assert conn.executed == []


def test_get_user_by_token_looks_up_by_hash():
    row = ("uid-1", "alice", "user", True, None, 5)
    conn = FakeConn(responses=[("auth_sessions a", [row])])

    user = auth.get_user_by_token(conn, "the-token")

    assert user == {
        "user_id": "uid-1", "username": "alice", "role": "user", "is_active": True,
        "daily_request_limit": None, "daily_token_limit": 5,
    }
    _, params = conn.executed[0]
    assert params == (auth.hash_token("the-token"),)


def test_get_user_by_token_none_when_not_found():
    assert auth.get_user_by_token(FakeConn(), "expired-or-revoked") is None


def test_update_user_rejects_unknown_fields_and_bad_roles():
    conn = FakeConn()
    with pytest.raises(auth.AuthError):
        auth.update_user(conn, "alice", password_hash="x")
    with pytest.raises(auth.AuthError):
        auth.update_user(conn, "alice", role="root")
    assert conn.executed == []


def test_update_user_with_no_fields_does_nothing():
    conn = FakeConn()
    assert auth.update_user(conn, "alice") is False
    assert conn.executed == []


def test_deactivating_a_user_revokes_their_tokens():
    conn = FakeConn(responses=[("users SET", [("uid-9",)])])

    assert auth.update_user(conn, "alice", is_active=False) is True

    revoke = conn.find("auth_sessions SET revoked")
    assert revoke and revoke[0][1] == ("uid-9",)


def test_update_user_unknown_user_returns_false():
    conn = FakeConn()
    assert auth.update_user(conn, "nobody", daily_request_limit=5) is False
    assert conn.rollbacks == 1


def test_update_user_can_reset_limits_to_default_with_none():
    conn = FakeConn(responses=[("users SET", [("uid-1",)])])
    auth.update_user(conn, "alice", daily_request_limit=None)
    _, params = conn.find("users SET")[0]
    assert params == (None, "alice")


def test_set_password_revokes_all_sessions():
    conn = FakeConn(responses=[("users SET password_hash", [("uid-3",)])])

    assert auth.set_password(conn, "alice", "brandnewpass1") is True

    assert conn.find("auth_sessions SET revoked")[0][1] == ("uid-3",)
    assert conn.find("users SET password_hash")[0][1][0].startswith("scrypt$")


def test_registration_is_on_by_default_and_can_be_switched_off(monkeypatch):
    monkeypatch.delenv("ALLOW_REGISTRATION", raising=False)
    assert auth.registration_enabled() is True

    monkeypatch.setenv("ALLOW_REGISTRATION", "1")
    assert auth.registration_enabled() is True

    monkeypatch.setenv("ALLOW_REGISTRATION", "0")
    assert auth.registration_enabled() is False


def test_is_admin_and_require_admin():
    assert auth.is_admin({"role": "admin"}) is True
    assert auth.is_admin({"role": "user"}) is False
    assert auth.is_admin(None) is False
    auth.require_admin({"role": "admin"})
    with pytest.raises(auth.PermissionDenied):
        auth.require_admin({"role": "user"})
