from dotenv import load_dotenv

from shared import llm_connection
from shared import query_guardrail
from shared.db import SCHEMA_NAME
from shared.prompt_registry import get_prompt

load_dotenv()

# Conversation memory
HISTORY_WINDOW = 6          # most recent messages sent verbatim to Agent 1
SUMMARY_BATCH = 6           # summarise once this many NEW messages fall out of the window
MAX_HISTORY_MESSAGE_CHARS = 600

# Long-term memory
MAX_MEMORIES_PER_USER = 50
MAX_MEMORY_CHARS = 500
RECALL_TOP_K = 3
RECALL_MAX_DISTANCE = 0.5   # cosine distance; bge embeddings are normalised

COMMANDS = ("remember", "forget", "memories")

SUMMARY_SYSTEM_INSTRUCTION = """You maintain a running summary of a conversation between a user and a research assistant that answers questions about a corpus of research papers.

Given the previous summary (it may be empty) and the newer messages, write an updated summary in at most 150 words.
- Keep: the topics discussed, which papers, methods or datasets came up, and any goals or preferences the user stated.
- Drop pleasantries and repeated content.
- Do not add anything that is not in the messages.

Output only the summary text."""


class MemoryCommandError(Exception):
    """A user-facing problem with a memory command (limit reached, rejected text)."""


# ---------- conversation memory ----------

def _cut(text: str) -> str:
    text = text or ""
    return text[:MAX_HISTORY_MESSAGE_CHARS] + "..." if len(text) > MAX_HISTORY_MESSAGE_CHARS else text


def citations_of(message: dict) -> list:
    """The papers an assistant message was built from, as [{"paper", "pages"}], read from the sources stored with the
    message (not parsed out of its text, which is cut). Empty when the answer had no sources, for example a
    general-knowledge answer; this tells the planner that "it" cannot mean a paper from that turn."""
    pages_by_paper = {}
    for source in (message.get("metadata") or {}).get("sources") or []:
        paper = source.get("source_pdf")
        if not paper:
            continue
        pages = pages_by_paper.setdefault(paper, set())
        if isinstance(source.get("page"), int):
            pages.add(source["page"])
    return [{"paper": paper, "pages": sorted(pages)} for paper, pages in pages_by_paper.items()]


def history_payload(summary: str, recent_messages: list) -> dict:
    """What Agent 1 receives about the conversation: {"summary", "recent_turns"}, or {} when there is none (an empty
    value, so a first question can still use the shared answer cache). Each turn is {"role", "text"} with the text
    cut at MAX_HISTORY_MESSAGE_CHARS; an assistant turn also carries its "citations" (see citations_of)."""
    turns = []
    for message in recent_messages or []:
        turn = {"role": "user" if message["role"] == "user" else "assistant", "text": _cut(message["content"])}
        if turn["role"] == "assistant":
            turn["citations"] = citations_of(message)
        turns.append(turn)
    if not summary and not turns:
        return {}
    return {"summary": summary or "", "recent_turns": turns}


def split_history(messages: list, window: int = HISTORY_WINDOW):
    """(older, recent): the recent window kept verbatim, everything before it."""
    if window <= 0:
        return list(messages), []
    return messages[:-window], messages[-window:]


def should_summarize(total_messages: int, summary_upto: int, window: int = HISTORY_WINDOW, batch: int = SUMMARY_BATCH) -> bool:
    older = max(0, total_messages - window)
    return older - summary_upto >= batch


def summarize(previous_summary: str, messages: list, client=None) -> str:
    """One LLM call: fold newly-old messages into the running summary."""
    def cited(message):
        found = citations_of(message) if message["role"] != "user" else []
        return " [cited: " + "; ".join(f"{c['paper']} p.{', '.join(map(str, c['pages']))}" if c["pages"] else c["paper"]
                                       for c in found) + "]" if found else ""

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:MAX_HISTORY_MESSAGE_CHARS]}{cited(m)}"
        for m in messages
    )
    user_content = f"Previous summary: {previous_summary or '(none)'}\n\nNewer messages:\n{transcript}\n\nWrite the updated summary now."
    instruction = get_prompt("conversation-summary", SUMMARY_SYSTEM_INSTRUCTION)
    response = llm_connection.generate(instruction, user_content, task="conversation_summary", client=client)
    return response.text.strip()


# ---------- long-term memory ----------

def parse_command(text: str):
    """'/remember I study lung CT' -> ('remember', 'I study lung CT'). None if
    the text is not a memory command."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    head, _, arg = stripped[1:].partition(" ")
    head = head.lower()
    if head not in COMMANDS:
        return None
    return head, arg.strip()


def _default_embed(text: str):
    from retrieval.retrieval_executor import embed_query

    return embed_query(text)


def count_memories(conn, user_id: str) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM {SCHEMA_NAME}.user_memories WHERE user_id = %s", (user_id,)
    ).fetchone()[0]


def remember(conn, user_id: str, text: str, embed_fn=None) -> int:
    """Stores a note the user EXPLICITLY asked to keep (never inferred
    automatically - consent by design). PII is redacted before storage, and
    text that looks like a prompt injection is refused because notes are
    later shown to the model."""
    embed_fn = embed_fn or _default_embed
    cleaned, _ = query_guardrail.redact_pii(text.strip())
    if not cleaned:
        raise MemoryCommandError("Nothing to remember. Usage: /remember <something about your interests>")
    if len(cleaned) > MAX_MEMORY_CHARS:
        raise MemoryCommandError(f"That note is too long (max {MAX_MEMORY_CHARS} characters).")
    if query_guardrail.detect_prompt_injection(cleaned):
        raise MemoryCommandError("That note looks like an instruction to the assistant, so it was not saved.")
    if count_memories(conn, user_id) >= MAX_MEMORIES_PER_USER:
        raise MemoryCommandError(f"You have reached the limit of {MAX_MEMORIES_PER_USER} notes. Use /forget to remove some.")

    embedding = embed_fn(cleaned)
    row = conn.execute(
        f"""INSERT INTO {SCHEMA_NAME}.user_memories (user_id, content, embedding)
            VALUES (%s, %s, %s) RETURNING memory_id""",
        (user_id, cleaned, embedding),
    ).fetchone()
    conn.commit()
    return row[0]


def recall(conn, user_id: str, query: str, k: int = RECALL_TOP_K, embed_fn=None) -> list:
    """The user's own notes most relevant to this query (cosine distance under
    the threshold). Scoped by user_id - one user never sees another's notes."""
    if count_memories(conn, user_id) == 0:
        return []
    embed_fn = embed_fn or _default_embed
    embedding = embed_fn(query)
    rows = conn.execute(
        f"""SELECT content, embedding <=> %s AS distance
            FROM {SCHEMA_NAME}.user_memories WHERE user_id = %s
            ORDER BY distance LIMIT %s""",
        (embedding, user_id, k),
    ).fetchall()
    return [content for content, distance in rows if distance <= RECALL_MAX_DISTANCE]


def list_memories(conn, user_id: str) -> list:
    rows = conn.execute(
        f"""SELECT memory_id, content, created_at FROM {SCHEMA_NAME}.user_memories
            WHERE user_id = %s ORDER BY memory_id""",
        (user_id,),
    ).fetchall()
    return [{"memory_id": r[0], "content": r[1], "created_at": r[2]} for r in rows]


def forget(conn, user_id: str, memory_id: int) -> bool:
    row = conn.execute(
        f"DELETE FROM {SCHEMA_NAME}.user_memories WHERE memory_id = %s AND user_id = %s RETURNING memory_id",
        (memory_id, user_id),
    ).fetchone()
    conn.commit()
    return row is not None


def forget_all(conn, user_id: str) -> int:
    rows = conn.execute(
        f"DELETE FROM {SCHEMA_NAME}.user_memories WHERE user_id = %s RETURNING memory_id", (user_id,)
    ).fetchall()
    conn.commit()
    return len(rows)


def run_command(conn, user_id: str, command: str, arg: str, embed_fn=None) -> str:
    """Executes /remember, /forget, /memories and returns the reply text."""
    try:
        if command == "remember":
            memory_id = remember(conn, user_id, arg, embed_fn=embed_fn)
            return f"Saved note #{memory_id}. I will use it as background about your interests. (/forget {memory_id} removes it.)"

        if command == "memories":
            memories = list_memories(conn, user_id)
            if not memories:
                return "You have no saved notes. Use /remember <text> to add one."
            return "Your saved notes:\n" + "\n".join(f"#{m['memory_id']}: {m['content']}" for m in memories)

        if command == "forget":
            if arg.lower() == "all":
                return f"Removed {forget_all(conn, user_id)} note(s)."
            if not arg.isdigit():
                return "Usage: /forget <note number> or /forget all"
            return f"Removed note #{arg}." if forget(conn, user_id, int(arg)) else f"No note #{arg} found."
    except MemoryCommandError as e:
        return str(e)
    return "Unknown command."
