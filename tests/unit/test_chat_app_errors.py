"""chat_app.py: what the USER sees when something goes wrong.

Runs the real Streamlit script inside Streamlit's AppTest harness. The database, login,
quotas and the chat call are faked, so no database, model or network is touched."""
import sys
import types
from pathlib import Path

import psycopg
import pytest
from streamlit.testing.v1 import AppTest

import auth
import chat_store
import chatbot
import db
import feedback
import memory
import online_report
import quotas
from fake_db import FakeConn

APP = str(Path(__file__).resolve().parents[2] / "src" / "chat_app.py")
USER = {"user_id": "u1", "username": "alice", "role": "user", "is_active": True,
        "daily_request_limit": None, "daily_token_limit": None}
ADMIN = {**USER, "user_id": "u0", "username": "root", "role": "admin"}

DATABASE_DOWN = "temporarily unavailable (the database could not be reached)"
UNEXPECTED = "Something went wrong while answering"


class FakeBorrow:
    """Stands in for db.connection()."""

    def __call__(self):
        return self

    def __enter__(self):
        return FakeConn()

    def __exit__(self, *args):
        return False


def start_app():
    return AppTest.from_file(APP, default_timeout=30)


def texts(elements):
    return [element.value for element in elements]


@pytest.fixture
def healthy(monkeypatch):
    """Everything the page needs, faked, for a signed-in regular user."""
    holder = types.SimpleNamespace(user=USER)
    monkeypatch.setattr(db, "connection", FakeBorrow())
    monkeypatch.setattr(auth, "ensure_admin_from_env", lambda conn: None)
    monkeypatch.setattr(auth, "get_user_by_token", lambda conn, token: holder.user if token else None)
    monkeypatch.setattr(quotas, "remaining", lambda conn, user: {
        "requests_used": 0, "requests_limit": 3, "requests_left": 3,
        "tokens_used": 0, "tokens_limit": 200_000, "tokens_left": 200_000,
    })
    monkeypatch.setattr(chat_store, "list_sessions", lambda conn, uid: [])
    monkeypatch.setattr(memory, "list_memories", lambda conn, uid: [])
    monkeypatch.setattr(feedback, "get_ratings", lambda conn, uid, ids: {})
    monkeypatch.setattr(online_report, "summary", lambda conn, days: {"days": days})
    monkeypatch.setattr(online_report, "alerts", lambda report: [])
    monkeypatch.setattr(online_report, "review_queue", lambda conn, days, limit=10: [])
    # the retrieval graph pulls in the embedding models: not needed here
    monkeypatch.setitem(sys.modules, "retrieval_graph", types.SimpleNamespace(build_graph=lambda: object()))
    return holder


def ask(healthy_state, question="What is a CNN?", token="t"):
    at = start_app()
    at.session_state["token"] = token
    at.run()
    at.chat_input[0].set_value(question).run()
    return at


# ---------- the database cannot be reached while the page loads ----------

def test_a_database_outage_at_page_load_shows_a_friendly_message_not_a_crash(monkeypatch):
    def down():
        raise psycopg.OperationalError("could not connect to server")

    monkeypatch.setattr(db, "connection", down)

    at = start_app().run()

    assert not at.exception
    assert any(DATABASE_DOWN in message for message in texts(at.warning))


def test_the_friendly_message_hides_the_technical_error(monkeypatch):
    def down():
        raise psycopg.OperationalError("could not connect to server at ep-secret-host.neon.tech")

    monkeypatch.setattr(db, "connection", down)

    at = start_app().run()

    everything = texts(at.warning) + texts(at.error) + texts(at.markdown) + texts(at.title)
    assert not any("ep-secret-host" in text for text in everything)


def test_a_database_failure_while_loading_the_sidebar_is_handled_the_same_way(healthy, monkeypatch):
    def broken(conn, user):
        raise psycopg.OperationalError("SSL connection has been closed unexpectedly")

    monkeypatch.setattr(quotas, "remaining", broken)

    at = start_app()
    at.session_state["token"] = "t"
    at.run()

    assert not at.exception
    assert any(DATABASE_DOWN in message for message in texts(at.warning))


def test_a_pool_timeout_at_page_load_is_handled_the_same_way(monkeypatch):
    import psycopg_pool

    def busy():
        raise psycopg_pool.PoolTimeout("couldn't get a connection after 15.00 sec")

    monkeypatch.setattr(db, "connection", busy)

    at = start_app().run()

    assert not at.exception and any(DATABASE_DOWN in message for message in texts(at.warning))


# ---------- the page works normally when the database is healthy ----------

def test_a_healthy_page_loads_the_login_form_when_signed_out(healthy):
    at = start_app().run()

    assert not at.exception and not at.warning and not at.error
    assert at.text_input  # the sign-in form


def test_a_healthy_page_loads_the_chat_when_signed_in(healthy):
    at = start_app()
    at.session_state["token"] = "t"

    at.run()

    assert not at.exception and not at.error and not at.warning
    assert at.chat_input


# ---------- a chat turn goes wrong ----------

def test_a_model_or_neo4j_outage_is_a_warning_with_the_not_charged_message(healthy, monkeypatch):
    def unavailable(*args, **kwargs):
        raise chatbot.ServiceUnavailable("all language models are busy or out of quota")

    monkeypatch.setattr(chatbot, "handle_message", unavailable)

    at = ask(healthy)

    assert not at.exception
    assert any("did not count against your daily limit" in message for message in texts(at.warning))


def test_a_database_outage_during_a_chat_turn_is_a_warning_with_the_not_charged_message(healthy, monkeypatch):
    def unavailable(*args, **kwargs):
        raise chatbot.ServiceUnavailable(chatbot.DATABASE_DOWN_REASON)

    monkeypatch.setattr(chatbot, "handle_message", unavailable)

    at = ask(healthy)

    assert any(chatbot.DATABASE_DOWN_REASON in message and "did not count" in message for message in texts(at.warning))


def test_an_unexpected_error_shows_a_regular_user_a_generic_message_only(healthy, monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError("secret detail: postgresql://user:pw@host/db")

    monkeypatch.setattr(chatbot, "handle_message", crash)

    at = ask(healthy)

    errors = texts(at.error)
    assert not at.exception                                     # no red Streamlit crash box
    assert any(UNEXPECTED in message for message in errors)
    assert not any("secret detail" in message or "postgresql" in message for message in errors)


def test_an_unexpected_error_shows_an_admin_the_real_details(healthy, monkeypatch):
    healthy.user = ADMIN

    def crash(*args, **kwargs):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(chatbot, "handle_message", crash)

    at = ask(healthy)

    assert any("RuntimeError: secret detail" in message for message in texts(at.error))


def test_an_unexpected_error_is_logged_for_the_operator(healthy, monkeypatch, caplog):
    def crash(*args, **kwargs):
        raise RuntimeError("the real cause")

    monkeypatch.setattr(chatbot, "handle_message", crash)

    with caplog.at_level("ERROR"):
        ask(healthy)

    assert any("chat turn failed" in record.message and record.exc_info for record in caplog.records)


def test_a_quota_refusal_is_still_a_plain_warning(healthy, monkeypatch):
    def refuse(*args, **kwargs):
        raise quotas.QuotaExceeded("daily_request_limit")

    monkeypatch.setattr(chatbot, "handle_message", refuse)

    at = ask(healthy)

    assert any("daily message limit" in message for message in texts(at.warning))
    assert not any(UNEXPECTED in message for message in texts(at.error))


def test_the_page_is_still_usable_after_an_error(healthy, monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chatbot, "handle_message", crash)

    at = ask(healthy)

    assert at.chat_input  # the user can type again
