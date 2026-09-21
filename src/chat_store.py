import json
import uuid

from db import SCHEMA_NAME

TITLE_MAX_CHARS = 60


class SessionNotFound(Exception):
    """The session does not exist OR belongs to someone else - deliberately
    indistinguishable, so one user cannot probe another's session ids."""


# Every query here is scoped by user_id. That is what keeps conversations from
# mixing between users: a session id alone is never enough to read or write.


def create_session(conn, user_id: str, title: str = None) -> str:
    session_id = str(uuid.uuid4())
    conn.execute(
        f"INSERT INTO {SCHEMA_NAME}.chat_sessions (session_id, user_id, title) VALUES (%s, %s, %s)",
        (session_id, user_id, title),
    )
    conn.commit()
    return session_id


def get_session(conn, user_id: str, session_id: str):
    row = conn.execute(
        f"""SELECT session_id, title, summary, summary_upto, created_at, updated_at
            FROM {SCHEMA_NAME}.chat_sessions WHERE session_id = %s AND user_id = %s""",
        (session_id, user_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row[0], "title": row[1], "summary": row[2], "summary_upto": row[3],
        "created_at": row[4], "updated_at": row[5],
    }


def list_sessions(conn, user_id: str, limit: int = 50) -> list:
    rows = conn.execute(
        f"""SELECT session_id, title, updated_at FROM {SCHEMA_NAME}.chat_sessions
            WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s""",
        (user_id, limit),
    ).fetchall()
    return [{"session_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def add_message(conn, user_id: str, session_id: str, role: str, content: str, metadata: dict = None) -> int:
    """Appends a message to a session the user owns (SessionNotFound
    otherwise) and bumps the session's updated_at."""
    if get_session(conn, user_id, session_id) is None:
        raise SessionNotFound(session_id)
    row = conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.chat_messages (session_id, role, content, metadata)
            VALUES (%s, %s, %s, %s) RETURNING message_id""",
        (session_id, role, content, json.dumps(metadata) if metadata is not None else None),
    ).fetchone()
    conn.execute(
        f"UPDATE {SCHEMA_NAME}.chat_sessions SET updated_at = now() WHERE session_id = %s AND user_id = %s",
        (session_id, user_id),
    )
    conn.commit()
    return row[0]


def get_messages(conn, user_id: str, session_id: str) -> list:
    """All messages of a session the user owns, oldest first."""
    rows = conn.execute(
        f"""SELECT m.message_id, m.role, m.content, m.metadata, m.created_at
            FROM {SCHEMA_NAME}.chat_messages m
            JOIN {SCHEMA_NAME}.chat_sessions s ON s.session_id = m.session_id
            WHERE m.session_id = %s AND s.user_id = %s
            ORDER BY m.message_id""",
        (session_id, user_id),
    ).fetchall()
    return [
        {"message_id": r[0], "role": r[1], "content": r[2], "metadata": r[3], "created_at": r[4]}
        for r in rows
    ]


def set_title_if_empty(conn, user_id: str, session_id: str, text: str) -> None:
    title = " ".join(text.split())[:TITLE_MAX_CHARS]
    conn.execute(
        f"""UPDATE {SCHEMA_NAME}.chat_sessions SET title = %s
            WHERE session_id = %s AND user_id = %s AND (title IS NULL OR title = '')""",
        (title, session_id, user_id),
    )
    conn.commit()


def set_summary(conn, user_id: str, session_id: str, summary: str, summary_upto: int) -> None:
    conn.execute(
        f"""UPDATE {SCHEMA_NAME}.chat_sessions SET summary = %s, summary_upto = %s
            WHERE session_id = %s AND user_id = %s""",
        (summary, summary_upto, session_id, user_id),
    )
    conn.commit()


def delete_session(conn, user_id: str, session_id: str) -> bool:
    """Deletes the session and (via ON DELETE CASCADE) its messages."""
    row = conn.execute(
        f"DELETE FROM {SCHEMA_NAME}.chat_sessions WHERE session_id = %s AND user_id = %s RETURNING session_id",
        (session_id, user_id),
    ).fetchone()
    conn.commit()
    return row is not None
