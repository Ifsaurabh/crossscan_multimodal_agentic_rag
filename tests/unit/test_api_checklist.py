"""FastAPI tests organised by the testing checklist (the basics live in test_api.py).

  A routing / HTTP methods      B input validation         C authentication / authorization
  D response schema             E error handling           F database interactions
  G startup (lifespan)          H admin quota display      I edge cases

Everything external is faked: the database is fake_db.FakeConn (or a fake pool),
the chatbot / graph is stubbed, and no model or network is ever touched.
"""
import sys
import threading
import types

import psycopg
import pytest
from fastapi.testclient import TestClient

import api
import auth
import quotas
from fake_db import FakeConn
from test_api import ADMIN, USER, as_user, chat_result, client  # noqa: F401  (client is a pytest fixture)


class FakeBorrow:
    """Stands in for db.connection(): lends one FakeConn and records how each borrow ended."""

    def __init__(self, fail_on_enter=None):
        self.conn = FakeConn()
        self.entered = 0
        self.exit_exceptions = []
        self.fail_on_enter = fail_on_enter

    def __call__(self):
        return self

    def __enter__(self):
        if self.fail_on_enter:
            raise self.fail_on_enter
        self.entered += 1
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.exit_exceptions.append(exc_type)
        return False


def use_conn(conn):
    """Make every request use this connection instead of the fixture's default."""
    def override():
        yield conn
    api.app.dependency_overrides[api.db_conn] = override


def quiet_client():
    """Like the normal client, but a crash inside the app comes back as a 500 response
    (what a real caller sees) instead of being re-raised into the test."""
    return TestClient(api.app, raise_server_exceptions=False)


def stub_chat(monkeypatch, **overrides):
    monkeypatch.setattr(api.chatbot, "handle_message", lambda *a, **k: chat_result(**overrides))


# ---------- A. routing / HTTP methods ----------

@pytest.mark.parametrize("method,path", [
    ("get", "/chat"), ("delete", "/chat"), ("post", "/health"), ("post", "/me"),
    ("put", "/memories"), ("post", "/stats"), ("get", "/auth/login"), ("post", "/admin/online-metrics"),
])
def test_the_wrong_http_method_is_405(client, method, path):
    as_user(ADMIN)
    assert getattr(client, method)(path).status_code == 405


def test_an_unknown_route_is_404(client):
    assert client.get("/no-such-route").status_code == 404


def test_registration_returns_201_and_delete_returns_204_with_no_body(client, monkeypatch):
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": {"username": u, "role": role})
    monkeypatch.setattr(api.chat_store, "delete_session", lambda conn, uid, sid: True)
    as_user(USER)

    assert client.post("/auth/register", json={"username": "newuser", "password": "longenough1"}).status_code == 201
    deleted = client.delete("/chat/sessions/s1")
    assert deleted.status_code == 204 and deleted.content == b""


# ---------- B. input validation ----------

@pytest.mark.parametrize("payload", [
    {"message": 123}, {"message": None}, {"message": ["hi"]}, {"message": {"a": 1}},
    {"message": "hi", "session_id": ["x"]}, {"message": "hi", "session_id": 5},
])
def test_chat_rejects_wrong_data_types(client, payload):
    as_user(USER)
    assert client.post("/chat", json=payload).status_code == 422


def test_chat_message_length_boundaries(client, monkeypatch):
    as_user(USER)
    stub_chat(monkeypatch)

    assert client.post("/chat", json={"message": "x"}).status_code == 200            # shortest allowed
    assert client.post("/chat", json={"message": "x" * 2000}).status_code == 200     # longest allowed
    assert client.post("/chat", json={"message": "x" * 2001}).status_code == 422     # one over
    assert client.post("/chat", json={"message": "x" * 1_000_000}).status_code == 422


def test_a_whitespace_only_message_is_refused_by_the_real_chat_logic(client):
    as_user(USER)  # no chatbot stub: the real handle_message runs and rejects it before any model call

    response = client.post("/chat", json={"message": "     "})

    assert response.status_code == 422 and "empty" in response.json()["detail"].lower()


@pytest.mark.parametrize("payload", [
    {}, {"username": "alice"}, {"password": "x"}, {"username": "", "password": "x"},
    {"username": "alice", "password": ""}, {"username": "a" * 65, "password": "x"},
    {"username": "alice", "password": "x" * 257}, {"username": 5, "password": "x"},
])
def test_login_rejects_bad_payloads(client, payload):
    assert client.post("/auth/login", json=payload).status_code == 422


def test_login_field_length_boundaries_are_accepted(client, monkeypatch):
    monkeypatch.setattr(api.auth, "authenticate", lambda conn, u, p: "tok")

    assert client.post("/auth/login", json={"username": "a" * 64, "password": "x" * 256}).status_code == 200
    assert client.post("/auth/login", json={"username": "a", "password": "x"}).status_code == 200


@pytest.mark.parametrize("payload", [
    {"username": "ab", "password": "longenough1"},           # username 1 under the minimum
    {"username": "a" * 33, "password": "longenough1"},       # username 1 over the maximum
    {"username": "newuser", "password": "1234567"},          # password 1 under the minimum
    {"username": "newuser", "password": "x" * 257},          # password 1 over the maximum
    {"username": "newuser"}, {"password": "longenough1"}, {},
])
def test_registration_rejects_out_of_range_or_missing_fields(client, monkeypatch, payload):
    monkeypatch.delenv("ALLOW_REGISTRATION", raising=False)
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": {"username": u, "role": role})

    assert client.post("/auth/register", json=payload).status_code == 422


@pytest.mark.parametrize("username,password", [
    ("abc", "12345678"), ("a" * 32, "12345678"), ("newuser", "x" * 256),   # each exactly on a boundary
])
def test_registration_accepts_values_exactly_on_the_boundaries(client, monkeypatch, username, password):
    monkeypatch.delenv("ALLOW_REGISTRATION", raising=False)
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": {"username": u, "role": role})

    assert client.post("/auth/register", json={"username": username, "password": password}).status_code == 201


@pytest.mark.parametrize("payload", [{"rating": "up"}, {}, {"rating": None}, {"rating": 1, "comment": 5}])
def test_feedback_rejects_bad_payloads(client, payload):
    as_user(USER)
    assert client.post("/chat/messages/5/feedback", json=payload).status_code == 422


def test_feedback_comment_boundary_and_a_non_numeric_message_id(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(api.feedback, "submit_feedback", lambda *a, **k: None)

    assert client.post("/chat/messages/5/feedback", json={"rating": 1, "comment": "x" * 500}).status_code == 200
    assert client.post("/chat/messages/abc/feedback", json={"rating": 1}).status_code == 422


def test_a_non_numeric_memory_id_is_422_not_a_crash(client):
    as_user(USER)
    assert client.delete("/memories/abc").status_code == 422


def test_admin_create_user_rejects_bad_payloads(client):
    as_user(ADMIN)

    assert client.post("/admin/users", json={"username": "ab", "password": "longenough1"}).status_code == 422
    assert client.post("/admin/users", json={"username": "newadmin", "password": "short"}).status_code == 422
    assert client.post("/admin/users", json={"password": "longenough1"}).status_code == 422


def test_admin_create_user_business_rules_come_back_as_400(client):
    as_user(ADMIN)  # the REAL auth.create_user runs: it checks the role and the username characters

    bad_role = client.post("/admin/users", json={"username": "newadmin", "password": "longenough1", "role": "superuser"})
    bad_name = client.post("/admin/users", json={"username": "bad name", "password": "longenough1"})

    assert bad_role.status_code == 400 and "Role must be one of" in bad_role.json()["detail"]
    assert bad_name.status_code == 400 and "Username" in bad_name.json()["detail"]


def test_admin_update_user_validation_and_boundaries(client, monkeypatch):
    as_user(ADMIN)
    captured = {}
    monkeypatch.setattr(api.auth, "update_user", lambda conn, username, **fields: captured.update(fields) or True)

    assert client.patch("/admin/users/bob", json={"daily_request_limit": -1}).status_code == 422
    assert client.patch("/admin/users/bob", json={"daily_token_limit": "abc"}).status_code == 422
    assert client.patch("/admin/users/bob", json={"is_active": "maybe"}).status_code == 422
    assert client.patch("/admin/users/bob", json={"daily_request_limit": 0}).status_code == 200
    assert captured == {"daily_request_limit": 0}  # 0 is a real limit ("no messages"), not "unset"


def test_admin_update_with_an_invalid_role_is_a_400_from_the_real_rules(client):
    as_user(ADMIN)

    response = client.patch("/admin/users/bob", json={"role": "superuser"})

    assert response.status_code == 400


def test_online_metrics_window_is_clamped_at_both_ends_and_must_be_a_number(client, monkeypatch):
    as_user(ADMIN)
    seen = []
    monkeypatch.setattr(api.online_report, "summary", lambda conn, days: seen.append(days) or {"days": days})
    monkeypatch.setattr(api.online_report, "alerts", lambda report: [])
    monkeypatch.setattr(api.online_report, "review_queue", lambda conn, days: [])

    client.get("/admin/online-metrics?days=0")
    client.get("/admin/online-metrics?days=-5")
    client.get("/admin/online-metrics?days=30")

    assert seen == [1, 1, 30]
    assert client.get("/admin/online-metrics?days=abc").status_code == 422


# ---------- C. authentication / authorization (the REAL token check, not the stub) ----------

def test_a_valid_bearer_token_reaches_the_endpoint(client, monkeypatch):
    seen = []
    monkeypatch.setattr(api.auth, "get_user_by_token", lambda conn, token: seen.append(token) or USER)
    monkeypatch.setattr(api.chat_store, "list_sessions", lambda conn, uid: [])

    response = client.get("/chat/sessions", headers={"Authorization": "Bearer good-token"})

    assert response.status_code == 200 and seen == ["good-token"]


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer unknown-or-expired-or-revoked"},
    {"Authorization": "Basic YWxpY2U6cHc="},   # wrong scheme
    {"Authorization": "Bearer "},              # empty token
    {},                                        # no credentials at all
])
def test_bad_or_missing_credentials_are_401_with_a_bearer_challenge(client, monkeypatch, headers):
    monkeypatch.setattr(api.auth, "get_user_by_token", lambda conn, token: None)

    response = client.get("/me", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("method,path,body", [
    ("post", "/admin/users", {"username": "newadmin", "password": "longenough1"}),
    ("patch", "/admin/users/bob", {"is_active": False}),
    ("get", "/admin/online-metrics", None), ("get", "/stats", None),
    ("post", "/auth/logout", None), ("delete", "/memories", None),
    ("delete", "/chat/sessions/s1", None), ("post", "/chat/messages/5/feedback", {"rating": 1}),
])
def test_every_protected_endpoint_refuses_missing_credentials(client, method, path, body):
    response = getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)
    assert response.status_code == 401


def test_a_regular_user_is_403_on_every_admin_endpoint(client):
    as_user(USER)

    assert client.get("/admin/online-metrics").status_code == 403
    assert client.get("/stats").status_code == 403
    assert client.get("/admin/users").status_code == 403


def test_wrong_username_and_wrong_password_give_the_same_answer(client, monkeypatch):
    def refuse(conn, u, p):
        raise auth.AuthError("Invalid username or password.")

    monkeypatch.setattr(api.auth, "authenticate", refuse)

    unknown = client.post("/auth/login", json={"username": "nobody", "password": "whatever1"})
    wrong = client.post("/auth/login", json={"username": "alice", "password": "wrongpass1"})

    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json() == {"detail": "Invalid username or password."}  # does not reveal which


def test_logout_without_credentials_does_not_revoke_anything(client, monkeypatch):
    revoked = []
    monkeypatch.setattr(api.auth, "revoke_token", lambda conn, token: revoked.append(token))

    assert client.post("/auth/logout").status_code == 401
    assert revoked == []


def test_memories_are_only_ever_touched_for_the_logged_in_user(client, monkeypatch):
    as_user(USER)
    calls = []
    monkeypatch.setattr(api.memory, "list_memories", lambda conn, uid: calls.append(("list", uid)) or [])
    monkeypatch.setattr(api.memory, "forget", lambda conn, uid, mid: calls.append(("forget", uid, mid)) or True)
    monkeypatch.setattr(api.memory, "forget_all", lambda conn, uid: calls.append(("forget_all", uid)) or 0)

    client.get("/memories")
    client.delete("/memories/7")
    client.delete("/memories")

    assert calls == [("list", "u1"), ("forget", "u1", 7), ("forget_all", "u1")]


# ---------- D. response schema ----------

def test_the_chat_response_has_exactly_the_documented_fields_and_types(client, monkeypatch):
    as_user(USER)
    stub_chat(monkeypatch)

    body = client.post("/chat", json={"message": "hi"}).json()

    assert set(body) == set(api.ChatResponse.model_fields)  # nothing missing, nothing extra (no "usage")
    assert isinstance(body["answer"], str) and isinstance(body["blocked"], bool)
    assert isinstance(body["cache_hit"], bool) and isinstance(body["is_command"], bool)
    assert isinstance(body["guardrail_flags"], list) and isinstance(body["sources"], list)
    assert isinstance(body["images"], list) and isinstance(body["latency_s"], (int, float))
    assert body["message_id"] is None and body["block_reason"] is None


def test_the_chat_response_carries_the_message_id_used_to_rate_it(client, monkeypatch):
    as_user(USER)
    stub_chat(monkeypatch, message_id=41)

    assert client.post("/chat", json={"message": "hi"}).json()["message_id"] == 41


def test_the_login_response_shape(client, monkeypatch):
    monkeypatch.setattr(api.auth, "authenticate", lambda conn, u, p: "tok123")

    body = client.post("/auth/login", json={"username": "alice", "password": "x"}).json()

    assert set(body) == {"token", "token_type", "expires_in_hours"}
    assert body["token_type"] == "bearer" and isinstance(body["expires_in_hours"], int)


def test_the_registration_response_never_contains_a_password_or_hash(client, monkeypatch):
    monkeypatch.delenv("ALLOW_REGISTRATION", raising=False)
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": {"username": u, "role": role})

    body = client.post("/auth/register", json={"username": "newuser", "password": "longenough1"}).json()

    assert set(body) == {"username", "role"}


def test_error_bodies_have_a_predictable_shape(client, monkeypatch):
    as_user(USER)

    def limited(*args, **kwargs):
        raise quotas.QuotaExceeded("daily_request_limit")

    monkeypatch.setattr(api.chatbot, "handle_message", limited)
    quota_error = client.post("/chat", json={"message": "hi"}).json()
    validation_error = client.post("/chat", json={"message": ""}).json()

    assert set(quota_error["detail"]) == {"reason", "message"}
    assert quota_error["detail"]["reason"] == "daily_request_limit"
    assert isinstance(validation_error["detail"], list)
    assert {"loc", "msg", "type"} <= set(validation_error["detail"][0])


# ---------- E. error handling ----------

def test_an_unexpected_crash_is_a_500_that_leaks_nothing(client, monkeypatch):
    as_user(USER)

    def crash(*args, **kwargs):
        raise RuntimeError("secret-internal-detail postgresql://user:pw@host/db")

    monkeypatch.setattr(api.chatbot, "handle_message", crash)

    response = quiet_client().post("/chat", json={"message": "hi"})

    assert response.status_code == 500
    assert "secret-internal-detail" not in response.text and "postgresql" not in response.text


@pytest.mark.parametrize("bad_result", [
    lambda: {k: v for k, v in chat_result().items() if k != "answer"},   # required field missing
    lambda: chat_result(answer=None),                                    # wrong type
    lambda: chat_result(sources="not-a-list"),                           # wrong type
])
def test_a_malformed_backend_result_is_a_500_not_a_broken_200(client, monkeypatch, bad_result):
    as_user(USER)
    monkeypatch.setattr(api.chatbot, "handle_message", lambda *a, **k: bad_result())

    assert quiet_client().post("/chat", json={"message": "hi"}).status_code == 500


def test_a_backend_timeout_is_a_500_response_not_a_hang(client, monkeypatch):
    as_user(USER)

    def slow(*args, **kwargs):
        raise TimeoutError("backend timed out")

    monkeypatch.setattr(api.chatbot, "handle_message", slow)

    assert quiet_client().post("/chat", json={"message": "hi"}).status_code == 500


def test_a_missing_optional_session_id_starts_a_new_chat(client, monkeypatch):
    as_user(USER)
    seen = []
    monkeypatch.setattr(
        api.chatbot, "handle_message",
        lambda user, sid, text, graph=None, conn=None: seen.append(sid) or chat_result(),
    )

    client.post("/chat", json={"message": "hi"})
    client.post("/chat", json={"message": "hi", "session_id": None})

    assert seen == [None, None]


def test_a_regular_user_cannot_promote_themselves_through_the_update_endpoint(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(api.auth, "update_user", lambda conn, username, **fields: pytest.fail("must not be reached"))

    assert client.patch("/admin/users/alice", json={"role": "admin"}).status_code == 403


# ---------- F. database interactions ----------

def test_each_request_borrows_one_connection_and_hands_it_back(client, monkeypatch):
    api.app.dependency_overrides.pop(api.db_conn)  # use the REAL dependency, with a fake pool behind it
    pool = FakeBorrow()
    monkeypatch.setattr(api, "connection", pool)
    monkeypatch.setattr(api.auth, "authenticate", lambda conn, u, p: "tok")

    client.post("/auth/login", json={"username": "alice", "password": "x"})
    client.post("/auth/login", json={"username": "alice", "password": "x"})

    assert pool.entered == 2 and pool.exit_exceptions == [None, None]


def test_the_connection_dependency_returns_the_connection_cleanly_on_success(monkeypatch):
    pool = FakeBorrow()
    monkeypatch.setattr(api, "connection", pool)

    generator = api.db_conn()
    assert next(generator) is pool.conn
    with pytest.raises(StopIteration):
        next(generator)

    assert pool.exit_exceptions == [None]


def test_the_connection_dependency_passes_a_request_failure_to_the_pool_so_it_rolls_back(monkeypatch):
    pool = FakeBorrow()
    monkeypatch.setattr(api, "connection", pool)

    generator = api.db_conn()
    next(generator)
    with pytest.raises(RuntimeError):
        generator.throw(RuntimeError("the request failed halfway"))

    assert pool.exit_exceptions == [RuntimeError]


def test_the_database_being_unreachable_is_a_500_with_no_details(client, monkeypatch):
    api.app.dependency_overrides.pop(api.db_conn)
    monkeypatch.setattr(api, "connection", FakeBorrow(fail_on_enter=RuntimeError("pool exhausted at host db.internal")))

    response = quiet_client().post("/auth/login", json={"username": "alice", "password": "x"})

    assert response.status_code == 500 and "db.internal" not in response.text


def test_a_query_failing_midway_is_a_500(client):
    class BrokenConn(FakeConn):
        def execute(self, sql, params=None):
            raise psycopg.OperationalError("connection lost")

    use_conn(BrokenConn())
    as_user(USER)

    assert quiet_client().get("/chat/sessions").status_code == 500


def test_deleting_a_chat_is_scoped_to_the_user_and_committed(client):
    as_user(USER)
    client.conn.responses = [("DELETE FROM", [("s1",)])]

    assert client.delete("/chat/sessions/s1").status_code == 204

    sql, params = client.conn.find("DELETE FROM")[0]
    assert "user_id = %s" in sql and params == ("s1", "u1")
    assert client.conn.commits == 1


def test_deleting_a_chat_that_is_not_yours_or_does_not_exist_is_404(client):
    as_user(USER)  # the fake database returns no row: nothing matched this user + session

    assert client.delete("/chat/sessions/someone-elses").status_code == 404


def test_listing_memories_reads_only_this_users_rows(client):
    as_user(USER)
    client.conn.responses = [("user_memories", [(1, "I study lung CT", "2026-09-25")])]

    body = client.get("/memories").json()

    assert body == [{"memory_id": 1, "content": "I study lung CT", "created_at": "2026-09-25"}]
    assert client.conn.find("user_memories")[0][1] == ("u1",)


def test_creating_a_user_stores_a_hash_never_the_password_and_commits(client):
    as_user(ADMIN)

    response = client.post("/admin/users", json={"username": "newadmin", "password": "longenough1", "role": "admin"})

    assert response.status_code == 201
    sql, params = client.conn.find("INSERT INTO")[0]
    _, username, stored, role = params
    assert username == "newadmin" and role == "admin"
    assert stored.startswith("scrypt$") and "longenough1" not in stored
    assert client.conn.commits == 1


def test_registration_always_stores_the_user_role_even_if_admin_is_requested(client, monkeypatch):
    monkeypatch.delenv("ALLOW_REGISTRATION", raising=False)

    client.post("/auth/register", json={"username": "sneaky", "password": "longenough1", "role": "admin"})

    assert client.conn.find("INSERT INTO")[0][1][3] == "user"


def test_a_duplicate_username_rolls_the_transaction_back_and_is_a_400(client, monkeypatch):
    monkeypatch.delenv("ALLOW_REGISTRATION", raising=False)

    class DuplicateConn(FakeConn):
        def execute(self, sql, params=None):
            if "INSERT INTO" in sql:
                raise psycopg.errors.UniqueViolation("duplicate key value")
            return super().execute(sql, params)

    conn = DuplicateConn()
    use_conn(conn)

    response = client.post("/auth/register", json={"username": "taken", "password": "longenough1"})

    assert response.status_code == 400 and "already taken" in response.json()["detail"]
    assert conn.rollbacks == 1 and conn.commits == 0


# ---------- G. startup (the lifespan added for the .env admin and the model warm-up) ----------

@pytest.fixture
def startup(monkeypatch):
    """Patches everything the startup hook touches and records what it did."""
    record = types.SimpleNamespace(
        pool=FakeBorrow(), admin_conns=[], warmed=threading.Event(), pool_closed=0, driver_closed=0,
    )
    monkeypatch.setattr(api, "connection", record.pool)
    monkeypatch.setattr(api.auth, "ensure_admin_from_env", lambda conn: record.admin_conns.append(conn))
    monkeypatch.setattr(api, "_warm_up_embedding_model", record.warmed.set)
    monkeypatch.setattr(api, "close_pool", lambda: setattr(record, "pool_closed", record.pool_closed + 1))
    monkeypatch.setattr(api, "close_driver", lambda: setattr(record, "driver_closed", record.driver_closed + 1))
    return record


def test_startup_syncs_the_env_admin_with_a_pooled_connection(startup):
    with TestClient(api.app):
        pass

    assert startup.admin_conns == [startup.pool.conn]
    assert startup.pool.exit_exceptions == [None]  # the connection went back to the pool


def test_startup_loads_the_embedding_model_in_the_background(startup):
    with TestClient(api.app):
        assert startup.warmed.wait(timeout=5)


def test_shutdown_closes_the_pool_and_the_driver_once(startup):
    with TestClient(api.app):
        assert startup.pool_closed == 0 and startup.driver_closed == 0  # still running

    assert startup.pool_closed == 1 and startup.driver_closed == 1


def test_a_failing_admin_sync_does_not_stop_the_api_from_starting(startup, monkeypatch, capsys):
    def broken(conn):
        raise RuntimeError("database not reachable")

    monkeypatch.setattr(api.auth, "ensure_admin_from_env", broken)

    with TestClient(api.app) as started:
        assert started.get("/health").json() == {"status": "ok"}
        assert startup.warmed.wait(timeout=5)  # the warm-up still happens

    assert "admin sync skipped" in capsys.readouterr().out


def test_the_warm_up_calls_the_models_loader(monkeypatch):
    calls = []
    fake_module = types.SimpleNamespace(get_embedding_model=lambda: calls.append("loaded"))
    monkeypatch.setitem(sys.modules, "retrieval_executor", fake_module)  # avoids importing the heavy real one

    api._warm_up_embedding_model()

    assert calls == ["loaded"]


def test_a_failing_warm_up_is_logged_and_never_raised(monkeypatch, capsys):
    def broken():
        raise OSError("model files missing")

    monkeypatch.setitem(sys.modules, "retrieval_executor", types.SimpleNamespace(get_embedding_model=broken))

    api._warm_up_embedding_model()  # must not raise

    assert "warm-up skipped" in capsys.readouterr().out


# ---------- H. admin quota display (admins are unlimited, regular users get 3 a day) ----------

def clear_limit_env(monkeypatch):
    for name in ("USER_DAILY_REQUEST_LIMIT", "USER_DAILY_TOKEN_LIMIT"):
        monkeypatch.delenv(name, raising=False)


def test_me_shows_an_admin_with_no_limits(client, monkeypatch):
    clear_limit_env(monkeypatch)
    as_user(ADMIN)
    client.conn.responses = [("usage_daily", [(42, 1000, 500)])]

    quota = client.get("/me").json()["quota"]

    assert quota["requests_used"] == 42 and quota["tokens_used"] == 1500  # usage is still counted
    assert quota["requests_limit"] is None and quota["requests_left"] is None
    assert quota["tokens_limit"] is None and quota["tokens_left"] is None


def test_me_shows_a_regular_user_with_three_messages_a_day(client, monkeypatch):
    clear_limit_env(monkeypatch)
    as_user(USER)
    client.conn.responses = [("usage_daily", [(1, 100, 50)])]

    quota = client.get("/me").json()["quota"]

    assert quota["requests_limit"] == 3 and quota["requests_left"] == 2
    assert quota["tokens_limit"] == 200_000 and quota["tokens_left"] == 200_000 - 150


def test_the_admin_user_list_shows_admins_unlimited_and_never_exposes_hashes(client, monkeypatch):
    clear_limit_env(monkeypatch)
    as_user(ADMIN)
    client.conn.responses = [
        ("ORDER BY created_at", [("u0", "root", "admin", True, None, None), ("u1", "alice", "user", True, None, None)]),
        ("usage_daily", [(2, 10, 10)]),
    ]

    listing = {u["username"]: u for u in client.get("/admin/users").json()}

    assert listing["root"]["quota"]["requests_limit"] is None
    assert listing["alice"]["quota"]["requests_limit"] == 3
    assert all("password" not in key for user in listing.values() for key in user)


# ---------- I. edge cases ----------

def test_unicode_and_emoji_messages_pass_through_unchanged(client, monkeypatch):
    as_user(USER)
    seen = []
    monkeypatch.setattr(
        api.chatbot, "handle_message",
        lambda user, sid, text, graph=None, conn=None: seen.append(text) or chat_result(),
    )
    message = "Qu'est-ce qu'un réseau de neurones ? 🧠 肺癌"

    client.post("/chat", json={"message": message})

    assert seen == [message]


def test_a_sql_looking_session_id_is_treated_as_plain_data(client, monkeypatch):
    as_user(USER)
    seen = []
    monkeypatch.setattr(api.chat_store, "get_session", lambda conn, uid, sid: seen.append(sid))

    response = client.get("/chat/sessions/x'%3B%20DROP%20TABLE%20users%3B--")

    assert response.status_code == 404
    assert seen == ["x'; DROP TABLE users;--"]  # passed as a value to a parameterised query, never spliced into SQL


@pytest.mark.parametrize("kwargs", [
    {"content": "not json", "headers": {"content-type": "application/json"}},
    {"content": "message=hi", "headers": {"content-type": "application/x-www-form-urlencoded"}},
    {"content": "", "headers": {"content-type": "application/json"}},
    {"json": [1, 2, 3]},
    {"json": "just a string"},
])
def test_a_body_of_the_wrong_kind_is_422(client, kwargs):
    as_user(USER)
    assert client.post("/chat", **kwargs).status_code == 422


def test_unexpected_extra_fields_are_ignored(client, monkeypatch):
    as_user(USER)
    stub_chat(monkeypatch)

    response = client.post("/chat", json={"message": "hi", "role": "admin", "is_admin": True, "user_id": "someone-else"})

    assert response.status_code == 200


def test_stats_with_no_requests_yet_is_all_zeros(client):
    as_user(ADMIN)

    stats = client.get("/stats").json()

    assert stats["requests"] == 0
    assert stats["latency_s"] == {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}


def test_percentile_handles_one_and_two_values_and_out_of_range_fractions():
    assert api.percentile([5.0], 0.5) == 5.0
    assert api.percentile([1.0, 9.0], 0.0) == 1.0
    assert api.percentile([1.0, 9.0], 1.0) == 9.0
    assert api.percentile([1.0, 9.0], 5.0) == 9.0  # never indexes past the end
