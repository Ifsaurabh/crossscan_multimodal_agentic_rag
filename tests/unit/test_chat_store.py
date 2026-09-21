import json

import pytest

import chat_store
from fake_db import FakeConn

SESSION_ROW = ("s1", "A title", None, 0, "created", "updated")


def test_create_session_binds_the_owner():
    conn = FakeConn()

    session_id = chat_store.create_session(conn, "user-1", title="hello")

    _, params = conn.find("INSERT INTO")[0]
    assert params == (session_id, "user-1", "hello")
    assert conn.commits == 1


def test_get_session_is_scoped_by_user():
    conn = FakeConn(responses=[("chat_sessions WHERE session_id", [SESSION_ROW])])

    session = chat_store.get_session(conn, "user-1", "s1")

    assert session["session_id"] == "s1" and session["summary_upto"] == 0
    assert conn.executed[0][1] == ("s1", "user-1")


def test_get_session_returns_none_for_someone_elses_session():
    assert chat_store.get_session(FakeConn(), "user-2", "s1") is None


def test_add_message_refuses_a_session_the_user_does_not_own():
    conn = FakeConn()  # ownership lookup finds nothing

    with pytest.raises(chat_store.SessionNotFound):
        chat_store.add_message(conn, "intruder", "s1", "user", "hi")

    assert conn.find("INSERT INTO") == []


def test_add_message_inserts_and_touches_the_session():
    conn = FakeConn(responses=[
        ("chat_sessions WHERE session_id", [SESSION_ROW]),
        ("RETURNING message_id", [(42,)]),
    ])

    message_id = chat_store.add_message(conn, "user-1", "s1", "assistant", "answer", metadata={"flags": []})

    assert message_id == 42
    _, params = conn.find("INSERT INTO")[0]
    assert params[:3] == ("s1", "assistant", "answer")
    assert json.loads(params[3]) == {"flags": []}
    assert conn.find("SET updated_at")[0][1] == ("s1", "user-1")
    assert conn.commits == 1


def test_get_messages_joins_on_owner_and_maps_rows():
    rows = [(11, "user", "hi", None, "t1"), (12, "assistant", "hello", {"sources": []}, "t2")]
    conn = FakeConn(responses=[("chat_messages m", rows)])

    messages = chat_store.get_messages(conn, "user-1", "s1")

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert [m["message_id"] for m in messages] == [11, 12]  # needed to attach feedback
    assert messages[1]["metadata"] == {"sources": []}
    assert conn.executed[0][1] == ("s1", "user-1")


def test_list_sessions_scoped_and_limited():
    conn = FakeConn(responses=[("ORDER BY updated_at DESC", [("s1", "T", "u")])])

    sessions = chat_store.list_sessions(conn, "user-1", limit=10)

    assert sessions == [{"session_id": "s1", "title": "T", "updated_at": "u"}]
    assert conn.executed[0][1] == ("user-1", 10)


def test_set_title_if_empty_collapses_whitespace_and_truncates():
    conn = FakeConn()

    chat_store.set_title_if_empty(conn, "user-1", "s1", "  what   is\n a CNN?  " + "x" * 200)

    title = conn.executed[0][1][0]
    assert len(title) <= chat_store.TITLE_MAX_CHARS
    assert title.startswith("what is a CNN?")
    assert "\n" not in title


def test_set_summary_is_scoped_by_user():
    conn = FakeConn()

    chat_store.set_summary(conn, "user-1", "s1", "a summary", 6)

    assert conn.executed[0][1] == ("a summary", 6, "s1", "user-1")


def test_delete_session_reports_whether_anything_was_deleted():
    deleted = FakeConn(responses=[("DELETE", [("s1",)])])
    assert chat_store.delete_session(deleted, "user-1", "s1") is True
    assert deleted.executed[0][1] == ("s1", "user-1")

    assert chat_store.delete_session(FakeConn(), "user-2", "s1") is False
