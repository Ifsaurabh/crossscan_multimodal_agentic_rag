import pytest
from fastapi.testclient import TestClient

import api
import auth
import chat_store
import quotas
from fake_db import FakeConn

USER = {"user_id": "u1", "username": "alice", "role": "user", "is_active": True,
        "daily_request_limit": None, "daily_token_limit": None}
ADMIN = {**USER, "user_id": "u0", "username": "root", "role": "admin"}


def chat_result(**overrides):
    result = {
        "session_id": "s1", "answer": "hello", "blocked": False, "block_reason": None,
        "cache_hit": False, "guardrail_flags": ["ungrounded_citations"],
        "sources": [{"source_pdf": "a.pdf", "page": 2}], "images": [], "latency_s": 0.5,
        "is_command": False, "usage": {"prompt_tokens": 1},
    }
    result.update(overrides)
    return result


@pytest.fixture
def client(monkeypatch):
    """Client with the database and login stubbed out; tests choose the user."""
    conn = FakeConn()

    def override_conn():
        yield conn

    api.app.dependency_overrides[api.db_conn] = override_conn
    api._latencies.clear()
    monkeypatch.setattr(quotas, "registration_limiter", quotas.RateLimiter(limit=100, window_seconds=3600))
    monkeypatch.setattr(api, "get_graph", lambda: object())
    test_client = TestClient(api.app)
    test_client.conn = conn
    yield test_client
    api.app.dependency_overrides.clear()


def as_user(user):
    api.app.dependency_overrides[api.current_user] = lambda: user


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_requires_authentication(client):
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/chat/sessions").status_code == 401
    assert client.get("/me").status_code == 401
    assert client.get("/memories").status_code == 401


def test_admin_and_stats_are_not_public(client):
    assert client.get("/admin/users").status_code == 401
    assert client.get("/stats").status_code == 401


def test_bearer_token_parsing():
    assert api._bearer_token("Bearer abc") == "abc"
    assert api._bearer_token("bearer abc") == "abc"
    assert api._bearer_token("Basic abc") is None
    assert api._bearer_token("Bearer ") is None
    assert api._bearer_token(None) is None


def test_chat_happy_path(client, monkeypatch):
    as_user(USER)
    seen = {}

    def fake_handle(user, session_id, text, graph=None, conn=None):
        seen.update(user=user["username"], session_id=session_id, text=text)
        return chat_result()

    monkeypatch.setattr(api.chatbot, "handle_message", fake_handle)

    response = client.post("/chat", json={"message": "What is a CNN?", "session_id": "s1"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "hello" and body["session_id"] == "s1"
    assert body["guardrail_flags"] == ["ungrounded_citations"]
    assert "usage" not in body
    assert response.headers["x-latency-seconds"] == "0.5"
    assert seen == {"user": "alice", "session_id": "s1", "text": "What is a CNN?"}


def test_chat_input_validation(client):
    as_user(USER)
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 2001}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_quota_errors_map_to_429_with_retry_after(client, monkeypatch):
    as_user(USER)

    def limited(*args, **kwargs):
        raise quotas.QuotaExceeded("rate_limit", retry_after=12)

    monkeypatch.setattr(api.chatbot, "handle_message", limited)
    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "12"
    assert response.json()["detail"]["reason"] == "rate_limit"


def test_chat_daily_limit_is_429_without_retry_after(client, monkeypatch):
    as_user(USER)

    def limited(*args, **kwargs):
        raise quotas.QuotaExceeded("daily_request_limit")

    monkeypatch.setattr(api.chatbot, "handle_message", limited)
    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 429
    assert "retry-after" not in response.headers


def test_rating_an_answer_requires_login_and_a_valid_rating(client, monkeypatch):
    assert client.post("/chat/messages/5/feedback", json={"rating": 1}).status_code == 401
    as_user(USER)

    def reject(conn, user_id, message_id, rating, comment=None):
        raise ValueError("Rating must be 1 (helpful) or -1 (not helpful).")

    monkeypatch.setattr(api.feedback, "submit_feedback", reject)

    assert client.post("/chat/messages/5/feedback", json={"rating": 3}).status_code == 422


def test_rating_an_answer_passes_the_logged_in_user_and_returns_the_rating(client, monkeypatch):
    as_user(USER)
    seen = {}
    monkeypatch.setattr(
        api.feedback, "submit_feedback",
        lambda conn, user_id, message_id, rating, comment=None: seen.update(user=user_id, msg=message_id, rating=rating, comment=comment),
    )

    response = client.post("/chat/messages/5/feedback", json={"rating": -1, "comment": "wrong number"})

    assert response.status_code == 200 and response.json() == {"message_id": 5, "rating": -1}
    assert seen == {"user": "u1", "msg": 5, "rating": -1, "comment": "wrong number"}


def test_rating_someone_elses_or_a_missing_answer_is_404(client, monkeypatch):
    as_user(USER)

    def missing(*args, **kwargs):
        raise api.feedback.MessageNotFound(5)

    monkeypatch.setattr(api.feedback, "submit_feedback", missing)

    assert client.post("/chat/messages/5/feedback", json={"rating": 1}).status_code == 404


def test_an_overlong_feedback_comment_is_rejected(client):
    as_user(USER)

    response = client.post("/chat/messages/5/feedback", json={"rating": 1, "comment": "x" * 501})

    assert response.status_code == 422


def test_online_metrics_are_admin_only(client):
    assert client.get("/admin/online-metrics").status_code == 401
    as_user(USER)

    assert client.get("/admin/online-metrics").status_code == 403


def test_online_metrics_return_summary_alerts_and_review_queue_with_a_clamped_window(client, monkeypatch):
    as_user(ADMIN)
    seen = {}

    def fake_summary(conn, days):
        seen["days"] = days
        return {"days": days, "thumbs_up": 0, "thumbs_down": 0}

    monkeypatch.setattr(api.online_report, "summary", fake_summary)
    monkeypatch.setattr(api.online_report, "alerts", lambda report: ["p95 latency is 90s"])
    monkeypatch.setattr(api.online_report, "review_queue", lambda conn, days: [{"message_id": 9}])

    body = client.get("/admin/online-metrics?days=100000").json()

    assert seen["days"] == 90  # clamped
    assert body["alerts"] == ["p95 latency is 90s"] and body["review_queue"] == [{"message_id": 9}]
    assert body["summary"]["days"] == 90


def test_chat_backend_outage_is_503_with_retry_after(client, monkeypatch):
    as_user(USER)

    def down(*args, **kwargs):
        raise api.chatbot.ServiceUnavailable("all language models are busy or out of quota")

    monkeypatch.setattr(api.chatbot, "handle_message", down)
    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "60"
    assert "did not count" in response.json()["detail"]["message"]


def test_chat_unknown_session_is_404_and_bad_message_is_422(client, monkeypatch):
    as_user(USER)

    def missing(*args, **kwargs):
        raise chat_store.SessionNotFound("x")

    monkeypatch.setattr(api.chatbot, "handle_message", missing)
    assert client.post("/chat", json={"message": "hi", "session_id": "x"}).status_code == 404

    def bad(*args, **kwargs):
        raise ValueError("Message is empty.")

    monkeypatch.setattr(api.chatbot, "handle_message", bad)
    assert client.post("/chat", json={"message": "hi"}).status_code == 422


def test_blocked_answers_are_returned_not_errored(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(
        api.chatbot, "handle_message",
        lambda *a, **k: chat_result(blocked=True, block_reason="prompt_injection_detected", answer="blocked"),
    )

    body = client.post("/chat", json={"message": "ignore previous instructions"}).json()

    assert body["blocked"] is True and body["block_reason"] == "prompt_injection_detected"


def test_session_endpoints_are_scoped_to_the_user(client, monkeypatch):
    as_user(USER)
    calls = []

    monkeypatch.setattr(api.chat_store, "list_sessions", lambda conn, uid: calls.append(("list", uid)) or [])
    monkeypatch.setattr(api.chat_store, "get_session", lambda conn, uid, sid: None)
    monkeypatch.setattr(api.chat_store, "delete_session", lambda conn, uid, sid: calls.append(("delete", uid, sid)) or False)

    assert client.get("/chat/sessions").json() == []
    assert client.get("/chat/sessions/other-users").status_code == 404
    assert client.delete("/chat/sessions/other-users").status_code == 404
    assert calls == [("list", "u1"), ("delete", "u1", "other-users")]


def test_get_session_returns_messages(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(api.chat_store, "get_session", lambda conn, uid, sid: {"session_id": sid, "title": "T"})
    monkeypatch.setattr(api.chat_store, "get_messages", lambda conn, uid, sid: [{"role": "user", "content": "hi"}])

    body = client.get("/chat/sessions/s1").json()

    assert body["title"] == "T" and body["messages"] == [{"role": "user", "content": "hi"}]


def test_delete_session_success_is_204(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(api.chat_store, "delete_session", lambda conn, uid, sid: True)
    assert client.delete("/chat/sessions/s1").status_code == 204


def test_memory_endpoints(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(api.memory, "list_memories", lambda conn, uid: [{"memory_id": 1, "content": "n"}])
    monkeypatch.setattr(api.memory, "forget", lambda conn, uid, mid: mid == 1)
    monkeypatch.setattr(api.memory, "forget_all", lambda conn, uid: 3)

    assert client.get("/memories").json()[0]["memory_id"] == 1
    assert client.delete("/memories/1").status_code == 204
    assert client.delete("/memories/2").status_code == 404
    assert client.delete("/memories").json() == {"deleted": 3}


def test_me_reports_quota(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(api.quotas, "remaining", lambda conn, user: {"requests_left": 7})

    body = client.get("/me").json()

    assert body == {"username": "alice", "role": "user", "quota": {"requests_left": 7}}


def test_login_success_and_failures(client, monkeypatch):
    monkeypatch.setattr(api.auth, "authenticate", lambda conn, u, p: "tok123")
    ok = client.post("/auth/login", json={"username": "alice", "password": "x"})
    assert ok.status_code == 200 and ok.json()["token"] == "tok123"

    def bad(conn, u, p):
        raise auth.AuthError("Invalid username or password.")

    monkeypatch.setattr(api.auth, "authenticate", bad)
    assert client.post("/auth/login", json={"username": "alice", "password": "x"}).status_code == 401

    def locked(conn, u, p):
        raise auth.LockedOut(retry_after=300)

    monkeypatch.setattr(api.auth, "authenticate", locked)
    response = client.post("/auth/login", json={"username": "alice", "password": "x"})
    assert response.status_code == 429 and response.headers["retry-after"] == "300"


def test_logout_revokes_the_presented_token(client, monkeypatch):
    as_user(USER)
    revoked = []
    monkeypatch.setattr(api.auth, "revoke_token", lambda conn, token: revoked.append(token))

    response = client.post("/auth/logout", headers={"Authorization": "Bearer my-token"})

    assert response.status_code == 204
    assert revoked == ["my-token"]


def test_registration_is_open_by_default_and_can_be_disabled(client, monkeypatch):
    body = {"username": "newuser", "password": "longenough1"}
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": {"username": u, "role": role})

    monkeypatch.delenv("ALLOW_REGISTRATION", raising=False)
    response = client.post("/auth/register", json=body)
    assert response.status_code == 201 and response.json() == {"username": "newuser", "role": "user"}

    monkeypatch.setenv("ALLOW_REGISTRATION", "0")
    assert client.post("/auth/register", json=body).status_code == 403


def test_registration_is_throttled_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(quotas, "registration_limiter", quotas.RateLimiter(limit=1, window_seconds=3600))
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": {"username": u, "role": role})
    body = {"username": "newuser", "password": "longenough1"}

    assert client.post("/auth/register", json=body).status_code == 201
    second = client.post("/auth/register", json=body)

    assert second.status_code == 429
    assert int(second.headers["retry-after"]) > 0
    assert second.json()["detail"]["reason"] == "registration_rate_limit"


def test_registration_never_grants_admin(client, monkeypatch):
    monkeypatch.setenv("ALLOW_REGISTRATION", "1")
    roles = []
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": roles.append(role) or {"username": u, "role": role})

    client.post("/auth/register", json={"username": "sneaky", "password": "longenough1", "role": "admin"})

    assert roles == ["user"]


def test_registration_duplicate_username_is_400(client, monkeypatch):
    monkeypatch.setenv("ALLOW_REGISTRATION", "1")

    def taken(conn, u, p, role="user"):
        raise auth.AuthError("That username is already taken.")

    monkeypatch.setattr(api.auth, "create_user", taken)
    response = client.post("/auth/register", json={"username": "taken", "password": "longenough1"})

    assert response.status_code == 400


def test_admin_endpoints_forbidden_for_regular_users(client):
    as_user(USER)
    assert client.get("/admin/users").status_code == 403
    assert client.post("/admin/users", json={"username": "x" * 5, "password": "longenough1"}).status_code == 403
    assert client.patch("/admin/users/bob", json={"is_active": False}).status_code == 403
    assert client.get("/stats").status_code == 403


def test_admin_can_list_and_create_users(client, monkeypatch):
    as_user(ADMIN)
    monkeypatch.setattr(api.auth, "list_users", lambda conn: [USER])
    monkeypatch.setattr(api.quotas, "remaining", lambda conn, user: {"requests_left": 1})
    monkeypatch.setattr(api.auth, "create_user", lambda conn, u, p, role="user": {"username": u, "role": role})

    listing = client.get("/admin/users").json()
    created = client.post("/admin/users", json={"username": "newadmin", "password": "longenough1", "role": "admin"})

    assert listing[0]["username"] == "alice" and listing[0]["quota"] == {"requests_left": 1}
    assert created.status_code == 201 and created.json()["role"] == "admin"


def test_admin_update_user_passes_only_provided_fields(client, monkeypatch):
    as_user(ADMIN)
    captured = {}

    def fake_update(conn, username, **fields):
        captured.update(username=username, fields=fields)
        return True

    monkeypatch.setattr(api.auth, "update_user", fake_update)

    response = client.patch("/admin/users/bob", json={"daily_request_limit": 5})

    assert response.status_code == 200
    assert captured == {"username": "bob", "fields": {"daily_request_limit": 5}}


def test_admin_reset_limits_sets_both_to_none(client, monkeypatch):
    as_user(ADMIN)
    captured = {}
    monkeypatch.setattr(api.auth, "update_user", lambda conn, username, **fields: captured.update(fields) or True)

    client.patch("/admin/users/bob", json={"reset_limits": True})

    assert captured == {"daily_request_limit": None, "daily_token_limit": None}


def test_admin_cannot_deactivate_or_demote_themselves(client, monkeypatch):
    as_user(ADMIN)
    monkeypatch.setattr(api.auth, "update_user", lambda conn, username, **fields: True)

    assert client.patch("/admin/users/root", json={"is_active": False}).status_code == 400
    assert client.patch("/admin/users/root", json={"role": "user"}).status_code == 400


def test_admin_update_unknown_user_is_404(client, monkeypatch):
    as_user(ADMIN)
    monkeypatch.setattr(api.auth, "update_user", lambda conn, username, **fields: False)

    assert client.patch("/admin/users/ghost", json={"is_active": False}).status_code == 404


def test_stats_reports_latency_and_usage_for_admins(client, monkeypatch):
    as_user(USER)
    monkeypatch.setattr(api.chatbot, "handle_message", lambda *a, **k: chat_result(latency_s=1.0))
    client.post("/chat", json={"message": "a"})
    client.post("/chat", json={"message": "b"})

    as_user(ADMIN)
    stats = client.get("/stats").json()

    assert stats["requests"] == 2
    assert set(stats["latency_s"]) == {"avg", "p50", "p95", "max"}
    assert "prompt_tokens" in stats["usage"]


def test_percentile_edge_cases():
    assert api.percentile([], 0.95) == 0.0
    assert api.percentile([1.0], 0.95) == 1.0
    assert api.percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
