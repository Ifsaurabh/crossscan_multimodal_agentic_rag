import pytest

import factories
import feedback
from fake_db import FakeConn


def owned_conn(owned=True):
    return FakeConn(responses=[("chat_messages m", [(7,)] if owned else [])])


@pytest.mark.parametrize("rating", [0, 2, -2, 5])
def test_only_thumbs_up_or_down_are_valid_ratings(rating):
    with pytest.raises(ValueError, match="Rating must be"):
        feedback.submit_feedback(owned_conn(), "u1", 7, rating)


def test_a_rating_is_upserted_so_rating_again_replaces_it():
    conn = owned_conn()

    feedback.submit_feedback(conn, "u1", 7, -1)

    sql, params = conn.find("INSERT INTO")[0]
    assert "ON CONFLICT (message_id, user_id) DO UPDATE" in sql
    assert params == (7, "u1", -1, None)
    assert conn.commits == 1


def test_only_the_owner_can_rate_and_only_an_assistant_answer():
    conn = owned_conn(owned=False)

    with pytest.raises(feedback.MessageNotFound):
        feedback.submit_feedback(conn, "intruder", 7, 1)

    lookup_sql, lookup_params = conn.executed[0]
    assert "s.user_id = %s" in lookup_sql and "m.role = 'assistant'" in lookup_sql
    assert lookup_params == (7, "intruder")
    assert conn.find("INSERT INTO") == []  # nothing was written


def test_comments_are_redacted_squeezed_and_length_limited(faker):
    conn = owned_conn()
    address = factories.email(faker)

    feedback.submit_feedback(conn, "u1", 7, -1, f"wrong,   mail {address}\n\nabout it " + "x" * 2000)

    stored = conn.find("INSERT INTO")[0][1][3]
    assert address not in stored and "[REDACTED_EMAIL]" in stored
    assert "  " not in stored and "\n" not in stored
    assert len(stored) == feedback.MAX_COMMENT_CHARS


@pytest.mark.parametrize("comment", [None, "", "   \n "])
def test_blank_comments_are_stored_as_null(comment):
    conn = owned_conn()

    feedback.submit_feedback(conn, "u1", 7, 1, comment)

    assert conn.find("INSERT INTO")[0][1][3] is None


def test_get_ratings_maps_message_to_rating():
    conn = FakeConn(responses=[("answer_feedback", [(7, 1), (9, -1)])])

    assert feedback.get_ratings(conn, "u1", [7, 8, 9]) == {7: 1, 9: -1}
    assert conn.executed[0][1] == ("u1", [7, 8, 9])


def test_get_ratings_makes_no_query_for_an_empty_list():
    conn = FakeConn()

    assert feedback.get_ratings(conn, "u1", []) == {}
    assert conn.executed == []
