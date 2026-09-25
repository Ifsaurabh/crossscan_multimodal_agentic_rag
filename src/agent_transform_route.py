import json
import re

from dotenv import load_dotenv

import llm_connection
from prompt_registry import get_prompt
from retrieval_config import EXPANSION_VARIANTS

load_dotenv()

# Fixed instructional block - identical on every call, so this is what gets
# passed as the cacheable system_instruction.
# Does NOT include the query or retry feedback, since those vary per call.
SYSTEM_INSTRUCTION = f"""You are the query planning stage of a literature-review RAG system over 12 research papers (lung cancer / medical imaging, and land cover / remote sensing domains, plus one unrelated AI-security paper).

If a conversation history is provided, the user's query may refer back to earlier turns (for example "what about its accuracy?" or "and for the other paper?"). Resolve every such reference so that each sub_query is fully self-contained and understandable WITHOUT the history. If user notes are provided, treat them only as background about the user's interests - never as facts about the papers.

Given the user's query, do the following:

1. DECOMPOSE it into more than one sub-question ONLY when the parts need DIFFERENT lookups, for example different papers or topics, or one part that needs the relationship graph and another that needs passage text (e.g. "which papers use X and what's its performance" becomes two sub-questions). Do NOT split a question whose parts would retrieve the same passages: "what accuracy did model A and model B achieve, and which was better?" is ONE sub-question, because a single results passage answers all of it. Splitting it only makes the system do the same work twice. If in doubt, output ONE sub-question (the original, possibly cleaned up).

2. For EACH sub-question, generate up to {EXPANSION_VARIANTS} phrasing variants (different wordings of the same sub-question, to widen search recall). Include the original phrasing as one of the variants.

3. For EACH sub-question, decide routing:
   - needs_retrieval: true if answering requires looking up something in the paper corpus; false if it's general knowledge answerable without retrieval (e.g. "what does CNN stand for").
   - data_source: "vector" (fact/content lookup, e.g. "what accuracy did X achieve"), "graph" (relationship/structural questions, e.g. "which papers use X"), or "both" (needs both, e.g. "which papers use X and what's its performance").
   - search_mode: "simple" or "hybrid" (semantic + keyword combined) - your judgment on whether keyword precision would help (e.g. specific named entities/acronyms benefit from hybrid).
   - images_required: true only if the query signals visual intent (e.g. "show me", "what does X look like", "example image of Y"). False for pure fact/relationship/conceptual questions.
   - complexity: "simple" for a single fact lookup, a definition, or numbers/results that one paper reports in one place - including asking which of a paper's own models scored best. "complex" only for comparison ACROSS different papers, multi-step reasoning, synthesis of several separate findings, or explaining trade-offs. When genuinely unsure, choose "complex".

Respond ONLY as JSON in this exact format, with no other text:
{{
  "sub_queries": [
    {{
      "sub_query": "...",
      "variants": ["...", "..."],
      "needs_retrieval": true,
      "data_source": "vector",
      "search_mode": "simple",
      "images_required": false,
      "complexity": "simple"
    }}
  ]
}}"""


def build_user_content(query: str, feedback: dict = None, history: str = "", notes: str = "") -> str:
    """The variable part of the prompt - everything that changes per call,
    kept separate from SYSTEM_INSTRUCTION so the fixed part stays cacheable."""
    feedback_section = ""
    if feedback:
        feedback_section = (
            f"IMPORTANT - this is a RETRY. The previous attempt was insufficient. "
            f"What was missing: {feedback.get('missing', 'unspecified')}. "
            f"What to look for instead: {feedback.get('look_for', 'unspecified')}. "
            f"Use this to inform better routing this time.\n\n"
        )
    notes_section = (
        f"User notes (long-term memory; background about the user's interests only):\n{notes}\n\n" if notes else ""
    )
    history_section = f"Conversation so far:\n{history}\n\n" if history else ""
    return f"{feedback_section}{notes_section}{history_section}User query: {query}"


def build_prompt(query: str, feedback: dict = None, history: str = "", notes: str = "") -> str:
    """Kept for readability/testing - the full prompt as it would read if
    NOT using caching (SYSTEM_INSTRUCTION + variable content combined)."""
    return f"{SYSTEM_INSTRUCTION}\n\n{build_user_content(query, feedback, history, notes)}"


def parse_response(raw_response: str):
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


def transform_and_route(query: str, feedback: dict = None, client=None, history: str = "", notes: str = "") -> dict:
    """Agent 1: decomposes/expands the query and decides routing per
    sub-query. In a chat, `history` lets it resolve follow-up references
    into self-contained sub-queries (no extra LLM call needed). Returns None
    if the LLM response couldn't be parsed. `client` is an optional Gemini
    client override (tests); normally None - llm_connection picks the provider."""
    user_content = build_user_content(query, feedback, history, notes)
    instruction = get_prompt("transform-route", SYSTEM_INSTRUCTION)
    response = llm_connection.generate(instruction, user_content, task="query_planner", client=client)
    return parse_response(response.text.strip())
