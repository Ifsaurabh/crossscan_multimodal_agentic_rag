import contextlib
import sys
import time

import psycopg

import chat_store
import memory
import online_eval
import query_guardrail
import quotas
import usage_tracker
from chat_store import SessionNotFound
from db import connection

MAX_MESSAGE_CHARS = 2000
DATABASE_DOWN_REASON = "the database could not be reached"


class ServiceUnavailable(Exception):
    """A backend the answer depends on is down or out of quota (every model in
    the tier failed, or Neo4j is unreachable). The user is not charged."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(
            "The assistant is temporarily unavailable "
            f"({reason}). This attempt did not count against your daily limit; please try again later."
        )


def _unavailable_reason(error: Exception):
    """A short reason if `error` is a known 'backend down' failure, else None."""
    from neo4j.exceptions import AuthError, ServiceUnavailable as Neo4jDown

    from llm_connection import AllProvidersFailed

    if isinstance(error, AllProvidersFailed):
        return "all language models are busy or out of quota"
    if isinstance(error, (Neo4jDown, AuthError)):
        return "the knowledge graph database is unreachable"
    return None


def collect_sources(result: dict) -> list:
    """Unique (paper, page) pairs the answer was built from."""
    seen, sources = set(), []
    for sq in result.get("sub_queries", []):
        for chunk in sq.get("chunks", []):
            key = (chunk.get("source_pdf"), chunk.get("page_start"))
            if key not in seen and key[0]:
                seen.add(key)
                sources.append({"source_pdf": key[0], "page": key[1]})
    return sources


def collect_images(result: dict) -> list:
    seen, images = set(), []
    for sq in result.get("sub_queries", []):
        for image in sq.get("images", []):
            name = image.get("image_file")
            if name and name not in seen:
                seen.add(name)
                images.append({"image_file": name, "page": image.get("page")})
    return images


def _reply(session_id, answer, **extra) -> dict:
    reply = {
        "session_id": session_id, "answer": answer, "blocked": False, "block_reason": None,
        "cache_hit": False, "guardrail_flags": [], "sources": [], "images": [],
        "latency_s": 0.0, "is_command": False, "usage": None, "message_id": None,
    }
    reply.update(extra)
    return reply


def _recall_query(messages: list, text: str) -> str:
    """The text used to look up the user's long-term notes. A follow-up such as
    "And which of them was the most accurate?" carries no topic of its own, so
    on its own it sits far from a note about the user's interests (measured
    cosine distance 0.58, over the 0.5 limit). Prefixing the user's previous
    question restores the topic (0.21). Only the user's own earlier question is
    used, never the assistant's long answer."""
    previous = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
    return f"{previous} {text}" if previous else text


def _maybe_summarize(conn, user_id, session_id, all_messages, session, summarize_fn):
    """Folds newly-old messages into the session's running summary. Best
    effort: a failure (or exhausted model quota) must never fail the turn -
    it just leaves the summary as it was and tries again later."""
    summary_upto = session["summary_upto"] if session else 0
    if not memory.should_summarize(len(all_messages), summary_upto):
        return
    older, _ = memory.split_history(all_messages)
    try:
        summary = summarize_fn(session["summary"] if session else "", older[summary_upto:])
        chat_store.set_summary(conn, user_id, session_id, summary, len(older))
    except Exception as e:
        print(f"   (conversation summary skipped: {e})")


def handle_message(
    user: dict, session_id, text: str, graph=None, conn=None, invoke=None, summarize_fn=None, embed_fn=None,
    online_eval_fn=None,
) -> dict:
    """One chat turn for an authenticated user.

    Raises quotas.QuotaExceeded (refused before any model call),
    chat_store.SessionNotFound (unknown/foreign session) or ValueError (bad
    message). Everything stored is PII-redacted."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Message is empty.")
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Message is too long (max {MAX_MESSAGE_CHARS} characters).")

    # A connection passed in belongs to the caller. Otherwise one is borrowed
    # from the pool for this turn and handed back in the finally block.
    borrowed = contextlib.ExitStack()
    try:
        if conn is None:
            conn = borrowed.enter_context(connection())
        user_id = user["user_id"]

        command = memory.parse_command(text)
        if command:
            answer = memory.run_command(conn, user_id, command[0], command[1], embed_fn=embed_fn)
            return _reply(session_id, answer, is_command=True)

        quotas.check_quota(conn, user)

        # A new conversation is only created after the graph succeeds, so a
        # failed first message leaves no empty session behind.
        session = None
        if session_id is not None:
            session = chat_store.get_session(conn, user_id, session_id)
            if session is None:
                raise SessionNotFound(session_id)

        redacted_text, _ = query_guardrail.redact_pii(text)
        messages = chat_store.get_messages(conn, user_id, session_id) if session else []
        _, recent = memory.split_history(messages)
        history = memory.format_history(session["summary"] if session else "", recent)
        notes = memory.format_notes(
            memory.recall(conn, user_id, _recall_query(messages, redacted_text), embed_fn=embed_fn)
        )

        if invoke is None:
            from run_query import invoke_graph as invoke
        if summarize_fn is None:
            summarize_fn = memory.summarize

        # Only a limited number of questions run at once; when every slot is
        # taken the user gets a "busy" refusal (not charged) instead of a
        # long wait. The slot is taken before usage tracking begins.
        with contextlib.nullcontext() if quotas.is_unlimited(user) else quotas.concurrency_limiter.slot():
            usage_tracker.begin_request()
            start = time.time()
            try:
                result = invoke(graph, redacted_text, history=history, notes=notes)
            except Exception as e:
                reason = _unavailable_reason(e)
                if reason is None:
                    raise
                raise ServiceUnavailable(reason) from e
            finally:
                usage = usage_tracker.end_request()
        latency = round(time.time() - start, 3)

        blocked = bool(result.get("blocked"))
        # Only a real answer uses up the user's allowance. A failed run (an
        # exception above) is the system's fault, and a message refused by
        # the input guardrail made no model call, so neither is charged.
        if not blocked or usage["prompt_tokens"] + usage["output_tokens"] > 0:
            quotas.record_usage(conn, user_id, usage["prompt_tokens"], usage["output_tokens"])

        if session_id is None:
            session_id = chat_store.create_session(conn, user_id)
        answer = result.get("final_answer")
        if blocked:
            answer = f"Your message could not be processed ({result.get('block_reason')})."
        answer = answer or "I could not produce an answer."

        sources = collect_sources(result)
        images = collect_images(result)
        flags = list(result.get("guardrail_flags", []))
        cache_hit = bool(result.get("cache_hit"))

        chat_store.add_message(conn, user_id, session_id, "user", redacted_text)
        # Latency and tokens are stored with the answer: they are what the
        # online health report (online_report.py) is computed from.
        assistant_message_id = chat_store.add_message(
            conn, user_id, session_id, "assistant", answer,
            metadata={
                "sources": sources, "images": images, "flags": flags, "cache_hit": cache_hit, "blocked": blocked,
                "latency_s": latency, "prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["output_tokens"],
            },
        )
        chat_store.set_title_if_empty(conn, user_id, session_id, redacted_text)

        # Online evaluation: a random sample of real answers is judged in a
        # background thread. Never charged to the user, never allowed to fail the turn.
        try:
            (online_eval_fn or online_eval.maybe_score)(
                assistant_message_id, redacted_text, answer, result,
                blocked=blocked or not result.get("final_answer"),
            )
        except Exception as e:
            print(f"   (online evaluation skipped: {e})")

        all_messages = messages + [
            {"role": "user", "content": redacted_text},
            {"role": "assistant", "content": answer},
        ]
        _maybe_summarize(conn, user_id, session_id, all_messages, session, summarize_fn)

        return _reply(
            session_id, answer, blocked=blocked, block_reason=result.get("block_reason"),
            cache_hit=cache_hit, guardrail_flags=flags, sources=sources, images=images,
            latency_s=latency, usage=usage, message_id=assistant_message_id,
        )
    except psycopg.OperationalError as e:
        # Postgres unreachable (Neon waking or down), a connection lost mid-turn, or the
        # pool timing out (psycopg_pool.PoolTimeout is one of these). Same friendly
        # answer as a model outage, instead of a crash.
        raise ServiceUnavailable(DATABASE_DOWN_REASON) from e
    finally:
        borrowed.__exit__(*sys.exc_info())
