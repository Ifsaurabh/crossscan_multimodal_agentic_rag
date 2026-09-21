from db import get_connection, SCHEMA_NAME
from setup_vector_db import TEXT_EMBEDDING_DIM

# Application tables: accounts, login sessions, chat history, usage counters
# and long-term memory. Every user-owned table hangs off users(user_id) with
# ON DELETE CASCADE, so deleting a user erases all of their data.
SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    daily_request_limit INTEGER,
    daily_token_limit INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.auth_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS auth_sessions_user_idx
    ON {SCHEMA_NAME}.auth_sessions (user_id);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.users(user_id) ON DELETE CASCADE,
    title TEXT,
    summary TEXT,
    summary_upto INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_sessions_user_idx
    ON {SCHEMA_NAME}.chat_sessions (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.chat_messages (
    message_id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.chat_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx
    ON {SCHEMA_NAME}.chat_messages (session_id, message_id);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.usage_daily (
    user_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.users(user_id) ON DELETE CASCADE,
    day DATE NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.user_memories (
    memory_id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.users(user_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector({TEXT_EMBEDDING_DIM}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS user_memories_user_idx
    ON {SCHEMA_NAME}.user_memories (user_id);

-- Online evaluation (quality of answers to REAL questions, in production).
-- A thumbs up/down (+ optional comment) from the user who received the answer.
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.answer_feedback (
    feedback_id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES {SCHEMA_NAME}.chat_messages(message_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES {SCHEMA_NAME}.users(user_id) ON DELETE CASCADE,
    rating SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (message_id, user_id)
);

-- Reference-free LLM-judge scores for a SAMPLE of live answers. No text is
-- stored here beyond the judge's one-line reason; the answer itself stays in
-- chat_messages, so deleting a user or chat erases the scores too.
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.online_eval_scores (
    message_id BIGINT PRIMARY KEY REFERENCES {SCHEMA_NAME}.chat_messages(message_id) ON DELETE CASCADE,
    faithfulness REAL,
    relevance REAL,
    judge_model TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def setup_app_db():
    conn = get_connection()
    conn.execute(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"App tables (users, auth_sessions, chat_sessions, chat_messages, usage_daily, user_memories, answer_feedback, online_eval_scores) ready in schema '{SCHEMA_NAME}'.")


if __name__ == "__main__":
    setup_app_db()
