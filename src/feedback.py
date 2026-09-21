from db import SCHEMA_NAME
from query_guardrail import redact_pii

MAX_COMMENT_CHARS = 500


class MessageNotFound(Exception):
    """The message does not exist, is not an answer, or belongs to someone
    else - deliberately indistinguishable, like chat_store.SessionNotFound."""


def submit_feedback(conn, user_id: str, message_id: int, rating: int, comment: str = None) -> None:
    """Records (or changes) the user's thumbs up (1) / down (-1) on an answer
    they received. One rating per user per answer: rating again replaces it.
    The comment is PII-redacted and length-limited before it is stored."""
    if rating not in (1, -1):
        raise ValueError("Rating must be 1 (helpful) or -1 (not helpful).")

    cleaned = None
    if comment and comment.strip():
        cleaned, _ = redact_pii(" ".join(comment.split()))
        cleaned = cleaned[:MAX_COMMENT_CHARS]

    # Only the owner of the conversation may rate one of its answers.
    owned = conn.execute(
        f"""SELECT m.message_id FROM {SCHEMA_NAME}.chat_messages m
            JOIN {SCHEMA_NAME}.chat_sessions s ON s.session_id = m.session_id
            WHERE m.message_id = %s AND s.user_id = %s AND m.role = 'assistant'""",
        (message_id, user_id),
    ).fetchone()
    if owned is None:
        raise MessageNotFound(message_id)

    conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.answer_feedback (message_id, user_id, rating, comment)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (message_id, user_id) DO UPDATE SET
                rating = EXCLUDED.rating, comment = EXCLUDED.comment, created_at = now()""",
        (message_id, user_id, rating, cleaned),
    )
    conn.commit()


def get_ratings(conn, user_id: str, message_ids: list) -> dict:
    """{message_id: 1 | -1} for the answers this user has already rated."""
    if not message_ids:
        return {}
    rows = conn.execute(
        f"""SELECT message_id, rating FROM {SCHEMA_NAME}.answer_feedback
            WHERE user_id = %s AND message_id = ANY(%s)""",
        (user_id, list(message_ids)),
    ).fetchall()
    return {row[0]: row[1] for row in rows}
