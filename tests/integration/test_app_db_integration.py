"""Checks the application SQL against a REAL Postgres (throwaway schema).

The unit tests use a stand-in connection, which happily accepts SQL a real
database would reject (upserts, pgvector `<=>`, `make_interval`, cascades,
JSONB round-trips). These tests exist to catch exactly that. Only the model
calls and the embedding model are faked; every database statement is real.

Run:  python -m pytest tests/integration --run-integration
"""
import numpy as np
import pytest

import auth
import chat_store
import chatbot
import factories
import feedback
import memory
import online_eval
import online_report
import quotas
from chat_store import SessionNotFound

pytestmark = pytest.mark.integration

APP_TABLES = {
    "users", "auth_sessions", "chat_sessions", "chat_messages", "usage_daily", "user_memories",
    "answer_feedback", "online_eval_scores",
}


def count(conn, schema, table, where="TRUE", params=()):
    return conn.execute(f"SELECT COUNT(*) FROM {schema}.{table} WHERE {where}", params).fetchone()[0]


def unit_vector(index, dim=768):
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


# ---------- schema ----------

def test_all_app_tables_exist(conn, schema):
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (schema,)
    ).fetchall()

    assert {r[0] for r in rows} == APP_TABLES


# ---------- auth ----------

def test_create_user_stores_a_hash_and_survives_a_duplicate_username(conn, faker, schema):
    name, plain = factories.username(faker), factories.password(faker)

    auth.create_user(conn, name, plain)
    stored = conn.execute(f"SELECT password_hash FROM {schema}.users WHERE username = %s", (name,)).fetchone()[0]

    assert stored.startswith("scrypt$") and plain not in stored
    with pytest.raises(auth.AuthError, match="already taken"):
        auth.create_user(conn, name, factories.password(faker))
    # the failed INSERT was rolled back, so the connection is still usable
    assert auth.get_user(conn, name)["username"] == name


def test_login_token_lifecycle(conn, faker, make_user, schema):
    user = make_user()

    token = auth.authenticate(conn, user["username"], user["password"])

    assert auth.get_user_by_token(conn, token)["user_id"] == user["user_id"]
    # only the hash of the token is stored
    assert count(conn, schema, "auth_sessions", "token_hash = %s", (auth.hash_token(token),)) == 1
    assert count(conn, schema, "auth_sessions", "token_hash = %s", (token,)) == 0

    auth.revoke_token(conn, token)
    assert auth.get_user_by_token(conn, token) is None
    assert auth.get_user_by_token(conn, factories.password(faker, 40)) is None  # unknown token


def test_wrong_password_and_unknown_user_are_rejected(conn, faker, make_user):
    user = make_user()

    with pytest.raises(auth.AuthError, match="Invalid username or password"):
        auth.authenticate(conn, user["username"], factories.password(faker))
    with pytest.raises(auth.AuthError, match="Invalid username or password"):
        auth.authenticate(conn, factories.username(faker), factories.password(faker))

    auth.clear_failed_logins(user["username"])


def test_expired_tokens_stop_working(conn, make_user, schema):
    user = make_user()
    token = auth.authenticate(conn, user["username"], user["password"])
    conn.execute(f"UPDATE {schema}.auth_sessions SET expires_at = now() - interval '1 minute'")
    conn.commit()

    assert auth.get_user_by_token(conn, token) is None


def test_deactivating_a_user_revokes_tokens_and_blocks_login(conn, make_user):
    user = make_user()
    token = auth.authenticate(conn, user["username"], user["password"])

    assert auth.update_user(conn, user["username"], is_active=False) is True

    assert auth.get_user_by_token(conn, token) is None
    with pytest.raises(auth.AuthError):
        auth.authenticate(conn, user["username"], user["password"])
    auth.clear_failed_logins(user["username"])


def test_setting_a_password_signs_the_user_out_everywhere(conn, faker, make_user):
    user = make_user()
    token = auth.authenticate(conn, user["username"], user["password"])
    new_password = factories.password(faker)

    assert auth.set_password(conn, user["username"], new_password) is True

    assert auth.get_user_by_token(conn, token) is None
    assert auth.authenticate(conn, user["username"], new_password)


def test_user_limits_can_be_set_and_reset_to_the_role_default(conn, make_user):
    user = make_user()

    auth.update_user(conn, user["username"], daily_request_limit=1, daily_token_limit=500)
    assert auth.get_user(conn, user["username"])["daily_request_limit"] == 1

    auth.update_user(conn, user["username"], daily_request_limit=None)
    limited = auth.get_user(conn, user["username"])
    assert limited["daily_request_limit"] is None and limited["daily_token_limit"] == 500
    assert auth.update_user(conn, "no-such-user-" + user["username"], daily_request_limit=3) is False


# ---------- quotas ----------

def test_record_usage_accumulates_per_user_per_day(conn, make_user, schema):
    alice, bob = make_user(), make_user()

    quotas.record_usage(conn, alice["user_id"], 100, 20)
    quotas.record_usage(conn, alice["user_id"], 50, 5)
    quotas.record_usage(conn, bob["user_id"], 7, 1)
    conn.execute(
        f"INSERT INTO {schema}.usage_daily (user_id, day, requests) VALUES (%s, current_date - 1, 99)",
        (alice["user_id"],),
    )
    conn.commit()

    assert quotas.get_usage_today(conn, alice["user_id"]) == {"requests": 2, "prompt_tokens": 150, "output_tokens": 25}
    assert quotas.get_usage_today(conn, bob["user_id"])["requests"] == 1  # yesterday's 99 does not count today


def test_daily_request_limit_is_enforced_from_real_usage(conn, make_user):
    user = make_user()
    auth.update_user(conn, user["username"], daily_request_limit=2)
    user = auth.get_user(conn, user["username"])
    limiter = quotas.RateLimiter(limit=100)

    quotas.check_quota(conn, user, limiter=limiter)
    quotas.record_usage(conn, user["user_id"], 0, 0)
    quotas.record_usage(conn, user["user_id"], 0, 0)

    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(conn, user, limiter=limiter)
    assert excinfo.value.reason == "daily_request_limit"


def test_global_cap_counts_every_users_requests(conn, make_user, monkeypatch):
    monkeypatch.setenv("GLOBAL_DAILY_REQUEST_CAP", "3")
    busy, idle = make_user(), make_user()
    for _ in range(3):
        quotas.record_usage(conn, busy["user_id"], 0, 0)

    assert quotas.get_global_requests_today(conn) == 3
    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        quotas.check_quota(conn, idle, limiter=quotas.RateLimiter(limit=100))
    assert excinfo.value.reason == "global_daily_cap"


# ---------- chat storage ----------

def test_messages_keep_order_and_jsonb_metadata(conn, make_user, faker):
    user = make_user()
    session_id = chat_store.create_session(conn, user["user_id"])
    metadata = {"sources": [{"source_pdf": "a.pdf", "page": 2}], "flags": [], "cache_hit": False}

    chat_store.add_message(conn, user["user_id"], session_id, "user", factories.message(faker))
    chat_store.add_message(conn, user["user_id"], session_id, "assistant", factories.message(faker), metadata=metadata)
    chat_store.add_message(conn, user["user_id"], session_id, "user", factories.message(faker))
    messages = chat_store.get_messages(conn, user["user_id"], session_id)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["metadata"] == metadata  # JSONB came back as a real dict
    assert messages[0]["metadata"] is None


def test_one_user_cannot_touch_anothers_conversation(conn, make_user, faker, schema):
    owner, intruder = make_user(), make_user()
    session_id = chat_store.create_session(conn, owner["user_id"])
    chat_store.add_message(conn, owner["user_id"], session_id, "user", factories.message(faker))

    assert chat_store.get_session(conn, intruder["user_id"], session_id) is None
    assert chat_store.get_messages(conn, intruder["user_id"], session_id) == []
    with pytest.raises(SessionNotFound):
        chat_store.add_message(conn, intruder["user_id"], session_id, "user", "let me in")
    assert chat_store.delete_session(conn, intruder["user_id"], session_id) is False
    assert chat_store.list_sessions(conn, intruder["user_id"]) == []

    assert len(chat_store.get_messages(conn, owner["user_id"], session_id)) == 1  # untouched
    assert count(conn, schema, "chat_messages") == 1


def test_deleting_a_session_removes_its_messages(conn, make_user, faker, schema):
    user = make_user()
    session_id = chat_store.create_session(conn, user["user_id"])
    chat_store.add_message(conn, user["user_id"], session_id, "user", factories.message(faker))

    assert chat_store.delete_session(conn, user["user_id"], session_id) is True

    assert count(conn, schema, "chat_messages", "session_id = %s", (session_id,)) == 0


def test_title_is_only_set_once_and_summary_is_stored(conn, make_user, faker):
    user = make_user()
    session_id = chat_store.create_session(conn, user["user_id"])

    chat_store.set_title_if_empty(conn, user["user_id"], session_id, "first title")
    chat_store.set_title_if_empty(conn, user["user_id"], session_id, "second title")
    chat_store.set_summary(conn, user["user_id"], session_id, "a rolling summary", 6)
    session = chat_store.get_session(conn, user["user_id"], session_id)

    assert session["title"] == "first title"
    assert session["summary"] == "a rolling summary" and session["summary_upto"] == 6


def test_most_recently_active_session_is_listed_first(conn, make_user, faker):
    user = make_user()
    older = chat_store.create_session(conn, user["user_id"], title="older")
    newer = chat_store.create_session(conn, user["user_id"], title="newer")
    chat_store.add_message(conn, user["user_id"], older, "user", factories.message(faker))  # touches `older`

    listed = [s["session_id"] for s in chat_store.list_sessions(conn, user["user_id"])]

    assert listed == [older, newer]


# ---------- long-term memory (real pgvector) ----------

def test_recall_returns_only_close_notes_and_only_your_own(conn, make_user, faker):
    mine, other = make_user(), make_user()
    vectors = {"note about lung CT": unit_vector(0), "note about satellites": unit_vector(1)}
    embed = lambda text: vectors.get(text, unit_vector(0))  # any other text (the query) is close to the CT note

    memory.remember(conn, mine["user_id"], "note about lung CT", embed_fn=embed)
    memory.remember(conn, mine["user_id"], "note about satellites", embed_fn=embed)
    memory.remember(conn, other["user_id"], "note about lung CT", embed_fn=embed)

    assert memory.recall(conn, mine["user_id"], factories.message(faker), embed_fn=embed) == ["note about lung CT"]
    assert memory.recall(conn, other["user_id"], factories.message(faker), embed_fn=lambda t: unit_vector(5)) == []


def test_remember_redacts_emails_and_enforces_the_per_user_limit(conn, make_user, faker, schema, monkeypatch):
    user = make_user()
    address = factories.email(faker)
    embed = lambda text: unit_vector(3)

    memory.remember(conn, user["user_id"], f"I work with {address} on CT nodules", embed_fn=embed)
    stored = conn.execute(f"SELECT content FROM {schema}.user_memories").fetchone()[0]
    assert address not in stored and "[REDACTED_EMAIL]" in stored

    monkeypatch.setattr(memory, "MAX_MEMORIES_PER_USER", 2)
    memory.remember(conn, user["user_id"], "second note", embed_fn=embed)
    with pytest.raises(memory.MemoryCommandError, match="limit"):
        memory.remember(conn, user["user_id"], "third note", embed_fn=embed)


def test_forget_and_forget_all_are_scoped_to_the_user(conn, make_user):
    mine, other = make_user(), make_user()
    embed = lambda text: unit_vector(2)
    first = memory.remember(conn, mine["user_id"], "one", embed_fn=embed)
    memory.remember(conn, mine["user_id"], "two", embed_fn=embed)
    theirs = memory.remember(conn, other["user_id"], "theirs", embed_fn=embed)

    assert memory.forget(conn, other["user_id"], first) is False  # not their note
    assert memory.forget(conn, mine["user_id"], first) is True
    assert memory.forget_all(conn, mine["user_id"]) == 1
    assert [m["memory_id"] for m in memory.list_memories(conn, other["user_id"])] == [theirs]


# ---------- cascades ----------

def test_deleting_a_user_erases_all_of_their_data(conn, make_user, faker, schema):
    user = make_user()
    auth.authenticate(conn, user["username"], user["password"])
    session_id = chat_store.create_session(conn, user["user_id"])
    chat_store.add_message(conn, user["user_id"], session_id, "user", factories.message(faker))
    memory.remember(conn, user["user_id"], "a note", embed_fn=lambda t: unit_vector(4))
    quotas.record_usage(conn, user["user_id"], 1, 1)

    conn.execute(f"DELETE FROM {schema}.users WHERE user_id = %s", (user["user_id"],))
    conn.commit()

    for table in APP_TABLES - {"users", "chat_messages"}:
        assert count(conn, schema, table) == 0, table
    assert count(conn, schema, "chat_messages") == 0


# ---------- the whole chat turn ----------

def graph_result(answer="A CNN is a neural network [a.pdf, p.1]."):
    return {
        "final_answer": answer, "blocked": False, "cache_hit": False, "guardrail_flags": [],
        "sub_queries": [{"chunks": [{"source_pdf": "a.pdf", "page_start": 1}], "images": []}],
    }


def test_a_chat_turn_is_stored_redacted_carries_history_and_is_metered(conn, make_user, faker, monkeypatch):
    monkeypatch.setenv("USER_DAILY_REQUEST_LIMIT", "2")
    monkeypatch.setenv("GLOBAL_DAILY_REQUEST_CAP", "100")
    monkeypatch.setattr(quotas, "rate_limiter", quotas.RateLimiter(limit=100))
    user = make_user()
    address = factories.email(faker)
    seen = []

    def fake_invoke(graph, query, history="", notes=""):
        seen.append({"query": query, "history": history})
        return graph_result()

    first = chatbot.handle_message(user, None, f"Email me at {address} about CNNs", conn=conn, invoke=fake_invoke)
    stored = [m["content"] for m in chat_store.get_messages(conn, user["user_id"], first["session_id"])]
    assert len(stored) == 2 and all(address not in text for text in stored)
    assert seen[0]["history"] == ""

    second = chatbot.handle_message(user, first["session_id"], "and its accuracy?", conn=conn, invoke=fake_invoke)
    assert "Recent turns:" in seen[1]["history"] and "CNNs" in seen[1]["history"]
    assert second["session_id"] == first["session_id"]
    assert quotas.get_usage_today(conn, user["user_id"])["requests"] == 2

    with pytest.raises(quotas.QuotaExceeded) as excinfo:
        chatbot.handle_message(user, first["session_id"], factories.message(faker), conn=conn, invoke=fake_invoke)
    assert excinfo.value.reason == "daily_request_limit"
    assert len(seen) == 2  # the refused message never reached the model


def test_memory_commands_do_not_use_up_the_daily_allowance(conn, make_user, monkeypatch):
    monkeypatch.setenv("USER_DAILY_REQUEST_LIMIT", "2")
    user = make_user()

    reply = chatbot.handle_message(user, None, "/memories", conn=conn)

    assert reply["is_command"] is True
    assert quotas.get_usage_today(conn, user["user_id"])["requests"] == 0


# ---------- online evaluation: feedback, judge scores, health report ----------

def make_answer(conn, user, faker, metadata=None, question=None):
    """A stored question + answer pair; returns (session_id, answer_message_id)."""
    session_id = chat_store.create_session(conn, user["user_id"])
    chat_store.add_message(conn, user["user_id"], session_id, "user", question or factories.message(faker))
    answer_id = chat_store.add_message(
        conn, user["user_id"], session_id, "assistant", factories.message(faker), metadata=metadata or {},
    )
    return session_id, answer_id


def test_feedback_is_upserted_once_per_user_and_answer(conn, make_user, faker, schema):
    user = make_user()
    _, answer_id = make_answer(conn, user, faker)

    feedback.submit_feedback(conn, user["user_id"], answer_id, 1)
    feedback.submit_feedback(conn, user["user_id"], answer_id, -1, "the number was wrong")

    assert count(conn, schema, "answer_feedback") == 1  # replaced, not duplicated
    assert feedback.get_ratings(conn, user["user_id"], [answer_id, 999999]) == {answer_id: -1}
    assert conn.execute(f"SELECT comment FROM {schema}.answer_feedback").fetchone()[0] == "the number was wrong"


def test_only_the_owner_can_rate_and_only_assistant_answers(conn, make_user, faker, schema):
    owner, intruder = make_user(), make_user()
    session_id, answer_id = make_answer(conn, owner, faker)
    question_id = conn.execute(
        f"SELECT message_id FROM {schema}.chat_messages WHERE session_id = %s AND role = 'user'", (session_id,)
    ).fetchone()[0]

    with pytest.raises(feedback.MessageNotFound):
        feedback.submit_feedback(conn, intruder["user_id"], answer_id, 1)
    with pytest.raises(feedback.MessageNotFound):
        feedback.submit_feedback(conn, owner["user_id"], question_id, 1)  # a question, not an answer

    assert count(conn, schema, "answer_feedback") == 0


def test_the_rating_constraint_is_enforced_by_the_database_too(conn, make_user, faker, schema):
    import psycopg

    user = make_user()
    _, answer_id = make_answer(conn, user, faker)

    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            f"INSERT INTO {schema}.answer_feedback (message_id, user_id, rating) VALUES (%s, %s, 5)",
            (answer_id, user["user_id"]),
        )
    conn.rollback()


def test_scores_and_feedback_are_erased_with_their_conversation(conn, make_user, faker, schema):
    user = make_user()
    session_id, answer_id = make_answer(conn, user, faker)
    feedback.submit_feedback(conn, user["user_id"], answer_id, 1)
    online_eval.record_score(conn, answer_id, {"faithfulness": 0.9, "relevance": 0.9, "judge_model": "m", "reason": "r"})

    chat_store.delete_session(conn, user["user_id"], session_id)

    assert count(conn, schema, "answer_feedback") == 0 and count(conn, schema, "online_eval_scores") == 0


def test_record_score_upserts_and_accepts_a_null_faithfulness(conn, make_user, faker, schema):
    user = make_user()
    _, answer_id = make_answer(conn, user, faker)

    online_eval.record_score(conn, answer_id, {"faithfulness": None, "relevance": 0.4, "judge_model": "m1", "reason": "a"})
    online_eval.record_score(conn, answer_id, {"faithfulness": 0.8, "relevance": 0.9, "judge_model": "m2", "reason": "b"})

    row = conn.execute(f"SELECT faithfulness, relevance, judge_model FROM {schema}.online_eval_scores").fetchone()
    assert count(conn, schema, "online_eval_scores") == 1
    assert row == (pytest.approx(0.8), pytest.approx(0.9), "m2")


def test_health_summary_is_computed_from_real_rows(conn, make_user, faker):
    user = make_user()
    for latency, cache_hit, blocked in ((10.0, False, False), (20.0, True, False), (30.0, False, True), (40.0, False, False)):
        make_answer(conn, user, faker, metadata={
            "latency_s": latency, "cache_hit": cache_hit, "blocked": blocked, "prompt_tokens": 900, "output_tokens": 100,
        })
    make_answer(conn, user, faker, metadata={"sources": []})  # an old answer with no health metadata

    result = online_report.summary(conn, days=7)

    assert result["answers"] == 5
    assert result["avg_latency_s"] == 25.0  # the old answer is ignored, not counted as 0
    assert result["p95_latency_s"] == pytest.approx(38.5, abs=0.01)
    assert result["cache_hit_rate"] == 0.25 and result["blocked_rate"] == 0.25  # of the 4 answers that have the field
    assert result["avg_tokens_per_answer"] == 800  # (4 x 1000 + 0) / 5


def test_summary_counts_votes_and_judge_scores(conn, make_user, faker):
    users = [make_user() for _ in range(3)]
    answer_ids = [make_answer(conn, u, faker)[1] for u in users]
    feedback.submit_feedback(conn, users[0]["user_id"], answer_ids[0], 1)
    feedback.submit_feedback(conn, users[1]["user_id"], answer_ids[1], 1)
    feedback.submit_feedback(conn, users[2]["user_id"], answer_ids[2], -1)
    online_eval.record_score(conn, answer_ids[0], {"faithfulness": 1.0, "relevance": 0.8, "judge_model": "m", "reason": ""})
    online_eval.record_score(conn, answer_ids[1], {"faithfulness": 0.5, "relevance": 0.6, "judge_model": "m", "reason": ""})

    result = online_report.summary(conn, days=7)

    assert (result["thumbs_up"], result["thumbs_down"]) == (2, 1)
    assert result["satisfaction"] == pytest.approx(0.667, abs=0.001)
    assert result["judged_answers"] == 2
    assert (result["avg_faithfulness"], result["avg_relevance"]) == (0.75, 0.7)


def test_summary_of_an_empty_database_does_not_crash(conn):
    result = online_report.summary(conn, days=7)

    assert result["answers"] == 0 and result["satisfaction"] is None and result["avg_faithfulness"] is None


def test_review_queue_finds_low_scores_and_thumbs_down_with_the_question(conn, make_user, faker):
    user = make_user()
    _, bad_score = make_answer(conn, user, faker, question="what accuracy did the CNN get?")
    _, thumbs_down = make_answer(conn, user, faker, question="which dataset was used?")
    _, fine = make_answer(conn, user, faker, question="an easy one")
    online_eval.record_score(conn, bad_score, {"faithfulness": 0.2, "relevance": 0.9, "judge_model": "m", "reason": "invented a number"})
    online_eval.record_score(conn, fine, {"faithfulness": 0.9, "relevance": 0.9, "judge_model": "m", "reason": ""})
    feedback.submit_feedback(conn, user["user_id"], thumbs_down, -1, "wrong dataset")

    queue = online_report.review_queue(conn, days=7)

    assert {item["message_id"] for item in queue} == {bad_score, thumbs_down}  # not `fine`
    by_id = {item["message_id"]: item for item in queue}
    assert by_id[bad_score]["question"] == "what accuracy did the CNN get?"
    assert by_id[bad_score]["judge_reason"] == "invented a number"
    assert by_id[thumbs_down]["thumbs"] == -1 and by_id[thumbs_down]["user_comment"] == "wrong dataset"


def test_a_null_faithfulness_alone_does_not_hide_a_low_relevance_answer(conn, make_user, faker):
    user = make_user()
    _, answer_id = make_answer(conn, user, faker)
    online_eval.record_score(conn, answer_id, {"faithfulness": None, "relevance": 0.1, "judge_model": "m", "reason": ""})

    assert [i["message_id"] for i in online_report.review_queue(conn)] == [answer_id]  # LEAST ignores NULL


def test_a_chat_turn_stores_health_metadata_and_a_sampled_answer_is_judged(conn, make_user, faker, schema, monkeypatch):
    import db

    monkeypatch.setenv("GLOBAL_DAILY_REQUEST_CAP", "100")
    monkeypatch.setattr(quotas, "rate_limiter", quotas.RateLimiter(limit=100))
    user = make_user()
    judged = []

    def fake_judge(query, answer, contexts):
        judged.append(query)
        return {"faithfulness": 0.95, "relevance": 0.9, "judge_model": "fake-judge", "reason": "fine"}

    def sampled(message_id, query, answer, result, blocked=False):
        return online_eval.maybe_score(
            message_id, query, answer, result, blocked=blocked, background=False, rate=1.0,
            judge_fn=fake_judge, conn_factory=db.get_connection,
        )

    def fake_invoke(graph, query, history="", notes=""):
        return {
            **graph_result(),
            "sub_queries": [{"sub_query": "CNN accuracy?", "chunks": [{"parent_text": "94%"}], "images": []}],
        }

    reply = chatbot.handle_message(user, None, factories.message(faker), conn=conn, invoke=fake_invoke, online_eval_fn=sampled)

    metadata = conn.execute(
        f"SELECT metadata FROM {schema}.chat_messages WHERE message_id = %s", (reply["message_id"],)
    ).fetchone()[0]
    assert metadata["latency_s"] >= 0 and "prompt_tokens" in metadata
    assert judged == ["CNN accuracy?"]
    assert online_report.summary(conn, days=1)["judged_answers"] == 1
    assert feedback.get_ratings(conn, user["user_id"], [reply["message_id"]]) == {}
