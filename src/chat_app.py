"""CrossScan chat UI. Run from the project root:  streamlit run src/chat_app.py

Talks to the same service layer as the API (auth, quotas, chat store, memory),
in one process - no separate API server needed for the UI.
"""
import logging
from pathlib import Path

import psycopg
import streamlit as st

import auth
import chat_store
import chatbot
import feedback
import memory
import online_report
import quotas
from db import connection

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"

DATABASE_DOWN_MESSAGE = (
    "The service is temporarily unavailable (the database could not be reached). "
    "Please try again in a minute."
)
UNEXPECTED_ERROR_MESSAGE = (
    "Something went wrong while answering. Please try again; if it keeps happening, tell the administrator."
)

PATTERN_GRAPH = "Agentic - Fixed Graph"
PATTERN_TOOLS = "Agentic - Tool-Calling"

st.set_page_config(page_title="CrossScan", layout="wide")


@st.cache_resource
def load_graph():
    from retrieval_graph import build_graph

    return build_graph()


def registration_enabled() -> bool:
    return auth.registration_enabled()


def login_view(conn):
    st.title("CrossScan")
    st.caption("Ask questions about a corpus of research papers (lung cancer imaging, land cover / remote sensing).")

    tabs = st.tabs(["Sign in", "Create account"]) if registration_enabled() else [st.container()]

    with tabs[0]:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            try:
                st.session_state["token"] = auth.authenticate(conn, username, password)
                st.rerun()
            except auth.LockedOut as e:
                st.error(f"{e} (retry in about {e.retry_after // 60 + 1} min)")
            except auth.AuthError as e:
                st.error(str(e))

    if registration_enabled():
        with tabs[1]:
            with st.form("register"):
                new_username = st.text_input("Choose a username")
                new_password = st.text_input("Choose a password (8+ characters)", type="password")
                created = st.form_submit_button("Create account")
            if created:
                try:
                    quotas.check_registration_allowed()
                    auth.create_user(conn, new_username, new_password, role="user")
                    st.success("Account created. You can sign in now.")
                except quotas.QuotaExceeded as e:
                    st.error(str(e))
                except auth.AuthError as e:
                    st.error(str(e))
    else:
        st.info("Accounts are created by an administrator.")


def render_extras(metadata):
    """Sources, images and warnings stored with an assistant message."""
    if not metadata:
        return
    if metadata.get("flags"):
        st.warning("Guardrail notes: " + ", ".join(metadata["flags"]))
    sources = metadata.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.write(f"{s['source_pdf']}, p.{s['page']}")
    shown = 0
    for image in metadata.get("images") or []:
        match = next(iter(IMAGES_DIR.rglob(image["image_file"])), None) if IMAGES_DIR.exists() else None
        if match is not None and shown < 4:
            st.image(str(match), caption=f"{image['image_file']} (p.{image.get('page')})", width=320)
            shown += 1


def render_feedback(conn, user, message_id, current_rating):
    """Thumbs up/down under an answer. The chosen one is highlighted; clicking
    again (or the other one) changes the rating."""
    if message_id is None:
        return
    up, down, _ = st.columns([1, 1, 12])
    for column, rating, icon in ((up, 1, "👍"), (down, -1, "👎")):
        chosen = current_rating == rating
        if column.button(icon, key=f"rate-{message_id}-{rating}", type="primary" if chosen else "secondary"):
            feedback.submit_feedback(conn, user["user_id"], message_id, rating)
            st.rerun()


def admin_health_panel(conn):
    """Online-evaluation numbers for admins: how the live system is doing."""
    with st.sidebar.expander("Live quality (admin, 7 days)"):
        report = online_report.summary(conn, 7)
        for warning in online_report.alerts(report):
            st.warning(warning)
        st.json(report)
        queue = online_report.review_queue(conn, 7, limit=10)
        st.caption(f"{len(queue)} answer(s) to review (low judge score or thumbs down)")
        for item in queue:
            st.write(f"#{item['message_id']} · faith {item['faithfulness']} · rel {item['relevance']} · thumbs {item['thumbs']}")
            st.caption(item["question"] or "")


def sidebar(conn, user):
    st.sidebar.subheader(user["username"])
    st.sidebar.caption(f"Role: {user['role']}")

    # Admin-only: the tool-calling demo has no guardrails, cache, memory or
    # quota, so with open sign-up it would let any user bypass their daily
    # limits and spend the shared Gemini quota. Non-admins always get the
    # fixed graph.
    if auth.is_admin(user):
        st.sidebar.selectbox(
            "Retrieval pattern",
            [PATTERN_GRAPH, PATTERN_TOOLS],
            key="retrieval_pattern",
            help="Fixed Graph = the real pipeline (guardrails, cache, memory, quotas). "
                 "Tool-Calling = demo only - no guardrails/cache/memory, not counted "
                 "against quota, answers are not saved to chat history.",
        )
    else:
        st.session_state["retrieval_pattern"] = PATTERN_GRAPH

    q = quotas.remaining(conn, user)
    if q["requests_limit"] is None:
        st.sidebar.caption(f"Messages today: {q['requests_used']} (unlimited)")
        st.sidebar.caption(f"Tokens today: {q['tokens_used']:,} (unlimited)")
    else:
        st.sidebar.caption(f"Messages today: {q['requests_used']} / {q['requests_limit']}")
        st.sidebar.caption(f"Tokens today: {q['tokens_used']:,} / {q['tokens_limit']:,}")

    if st.sidebar.button("New chat", use_container_width=True):
        st.session_state["session_id"] = None
        st.session_state.pop("demo_turns", None)
        st.rerun()

    st.sidebar.markdown("**Your chats**")
    for s in chat_store.list_sessions(conn, user["user_id"]):
        label = (s["title"] or "Untitled")[:40]
        if st.sidebar.button(label, key=f"session-{s['session_id']}", use_container_width=True):
            st.session_state["session_id"] = s["session_id"]
            st.rerun()

    session_id = st.session_state.get("session_id")
    if session_id and st.sidebar.button("Delete this chat", use_container_width=True):
        chat_store.delete_session(conn, user["user_id"], session_id)
        st.session_state["session_id"] = None
        st.rerun()

    with st.sidebar.expander("Your saved notes"):
        notes = memory.list_memories(conn, user["user_id"])
        if not notes:
            st.caption("None yet. Type /remember <text> in the chat.")
        for note in notes:
            st.write(f"#{note['memory_id']}: {note['content']}")
            if st.button("Forget", key=f"forget-{note['memory_id']}"):
                memory.forget(conn, user["user_id"], note["memory_id"])
                st.rerun()

    if auth.is_admin(user):
        admin_health_panel(conn)

    if st.sidebar.button("Sign out", use_container_width=True):
        auth.revoke_token(conn, st.session_state.get("token"))
        for key in ("token", "session_id"):
            st.session_state.pop(key, None)
        st.rerun()


def chat_view(conn, user):
    st.title("CrossScan")
    session_id = st.session_state.get("session_id")
    messages = chat_store.get_messages(conn, user["user_id"], session_id) if session_id else []

    if not messages:
        st.caption("Ask about methods, datasets, results or figures in the papers. Commands: /remember, /memories, /forget.")

    ratings = feedback.get_ratings(
        conn, user["user_id"], [m["message_id"] for m in messages if m["role"] == "assistant"],
    )
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_extras(message["metadata"])
                render_feedback(conn, user, message["message_id"], ratings.get(message["message_id"]))

    # Tool-calling demo turns live only in this browser session (not in the
    # database), so they are re-rendered from session_state on every rerun.
    for turn in st.session_state.get("demo_turns", []):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.caption(f"{PATTERN_TOOLS} - demo: no guardrails, cache, memory or quota; not saved to history")
            st.markdown(turn["answer"])

    prompt = st.chat_input("Ask about the papers...")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    # Checked again here (not only via the sidebar widget) so a non-admin can
    # never reach the unguarded path.
    if auth.is_admin(user) and st.session_state.get("retrieval_pattern") == PATTERN_TOOLS:
        try:
            with st.spinner("Agent is choosing tools..."):
                import agentic_demo  # lazy: needs langchain-google-genai, only this path uses it

                answer = agentic_demo.ask(prompt)
        except Exception as e:
            st.error(f"Tool-calling agent failed: {e}")
            return
        st.session_state.setdefault("demo_turns", []).append({"question": prompt, "answer": answer})
        st.rerun()

    try:
        with st.spinner("Searching the papers..."):
            result = chatbot.handle_message(user, session_id, prompt, graph=load_graph(), conn=conn)
    except quotas.QuotaExceeded as e:
        st.warning(str(e))
        return
    except chatbot.ServiceUnavailable as e:
        st.warning(str(e))
        return
    except chat_store.SessionNotFound:
        st.session_state["session_id"] = None
        st.error("That chat no longer exists. Start a new one.")
        return
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:  # anything unexpected: logged for the operator, the page stays usable
        logging.exception("chat turn failed")
        if auth.is_admin(user):
            st.error(f"Unexpected error (details are shown to admins only): {type(e).__name__}: {e}")
        else:
            st.error(UNEXPECTED_ERROR_MESSAGE)
        return

    if result["is_command"]:
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
        return

    st.session_state["session_id"] = result["session_id"]
    st.rerun()


@st.cache_resource
def sync_admin():
    with connection() as conn:
        auth.ensure_admin_from_env(conn)


def main():
    try:
        sync_admin()
        with connection() as conn:
            user = auth.get_user_by_token(conn, st.session_state.get("token"))
            if user is None:
                st.session_state.pop("token", None)
                login_view(conn)
                return
            sidebar(conn, user)
            chat_view(conn, user)
    except psycopg.OperationalError:
        # Postgres unreachable or the connection pool timed out, at login / sidebar time.
        logging.exception("database unavailable while loading the page")
        st.title("CrossScan")
        st.warning(DATABASE_DOWN_MESSAGE)


main()
