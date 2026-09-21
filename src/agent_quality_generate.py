import json
import re

from dotenv import load_dotenv

import llm_connection
from prompt_registry import get_prompt

load_dotenv()

# Fixed instructional blocks - identical on every call, passed as the
# cacheable system_instruction. Query/context/caveat are the variable part
# and are NEVER cached (they're different on every single call anyway).
QUALITY_CHECK_SYSTEM_INSTRUCTION = """You are judging whether retrieved context is sufficient to answer a user's question about research papers.

Is this context sufficient to answer the question accurately and specifically? Respond ONLY as JSON:
{"sufficient": true, "missing": "", "look_for": ""}

If NOT sufficient, explain what's missing and what to look for instead (to guide a retry):
{"sufficient": false, "missing": "specific description of what's missing", "look_for": "specific guidance for what to search for instead"}"""

GENERATE_SYSTEM_INSTRUCTION = """You are a research-assistant answering questions about a corpus of 12 research papers (lung cancer / medical imaging, and land cover / remote sensing domains).

Instructions:
- Answer using ONLY the retrieved context provided. Do not use outside knowledge.
- Cite your source inline for each claim, in the format [source_pdf, p.X] immediately after the claim it supports.
- If images are relevant and listed below, reference them naturally in your answer.
- Be concise and directly answer the question."""

GENERAL_KNOWLEDGE_SYSTEM_INSTRUCTION = """You are answering a general-knowledge question that does not require looking up anything in a specific document corpus (e.g. definitions, well-known facts).

Instructions:
- Answer directly and concisely using your own knowledge.
- Do not fabricate citations or claim the answer comes from any document corpus."""

NOT_FROM_KNOWLEDGE_BASE_LABEL = "**Note: this answer is not from the knowledge base — it is AI-generated from general knowledge.**\n\n"


def format_context(chunks: list) -> str:
    blocks = []
    for c in chunks:
        source = c.get("source_pdf", "unknown")
        page = c.get("page_start", "?")
        text = c.get("parent_text") or c.get("text", "")
        blocks.append(f"[{source}, p.{page}]\n{text}")
    return "\n\n".join(blocks)


def parse_json_response(raw_response: str):
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def check_quality(query: str, chunks: list, client=None) -> dict:
    """Agent 2, judgment phase. Returns {sufficient, missing, look_for}.
    `client` is an optional Gemini client override (tests); normally None -
    llm_connection picks the provider."""
    if not chunks:
        return {"sufficient": False, "missing": "no results retrieved", "look_for": "broader search terms"}

    context = format_context(chunks)
    user_content = f"User's question: {query}\n\nRetrieved context:\n{context}"
    instruction = get_prompt("quality-check", QUALITY_CHECK_SYSTEM_INSTRUCTION)
    response = llm_connection.generate(instruction, user_content, task="quality_check", client=client)
    result = parse_json_response(response.text.strip())

    if result is None:
        return {"sufficient": False, "missing": "could not judge quality", "look_for": "retry with broader search"}
    return result


def answer_task(complexity: str) -> str:
    """Simple sub-questions are answered on the fast tier, everything else
    (including a missing or unrecognised label) on the reasoning tier."""
    return "answer_simple" if complexity == "simple" else "answer_complex"


def generate_answer(query: str, chunks: list, low_confidence: bool = False, client=None, complexity: str = "complex") -> str:
    """Agent 2, generation phase. `complexity` (from the query planner)
    picks the model tier the answer is written on."""
    context = format_context(chunks) if chunks else "(no context retrieved)"
    caveat_instruction = (
        "\n\nIMPORTANT: retrieval confidence was low for this query. Start your answer with a brief "
        "caveat noting the answer may be incomplete, then answer with whatever context is available."
        if low_confidence else ""
    )
    user_content = (
        f"User's question: {query}\n\n"
        f"Retrieved context (each block tagged with its source):\n{context}"
        f"{caveat_instruction}\n\nWrite the answer now."
    )
    instruction = get_prompt("generate-answer", GENERATE_SYSTEM_INSTRUCTION)
    response = llm_connection.generate(instruction, user_content, task=answer_task(complexity), client=client)
    return response.text.strip()


def generate_general_knowledge_answer(query: str, client=None) -> str:
    """For sub-queries the Transform+Route Agent marked needs_retrieval=False.
    Answers from general knowledge and prepends a transparency label so the
    user always knows when an answer did NOT come from the document corpus."""
    user_content = f"Question: {query}\n\nAnswer now."
    instruction = get_prompt("general-knowledge", GENERAL_KNOWLEDGE_SYSTEM_INSTRUCTION)
    response = llm_connection.generate(instruction, user_content, task="general_knowledge", client=client)
    return NOT_FROM_KNOWLEDGE_BASE_LABEL + response.text.strip()
