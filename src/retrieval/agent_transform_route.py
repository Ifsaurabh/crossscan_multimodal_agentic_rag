import json
import re

from dotenv import load_dotenv

from shared import llm_connection
from shared.prompt_registry import get_prompt
from retrieval.retrieval_config import EXPANSION_VARIANTS, MAX_SUB_QUERIES

load_dotenv()

SEARCH_MODES = ("simple", "hybrid")

# Fixed instructional block - identical on every call, so this is what gets
# passed as the system_instruction.
# Does NOT include the query or retry feedback, since those vary per call.
SYSTEM_INSTRUCTION = f"""You are the query planning stage of a literature-review RAG system over 12 research papers (lung cancer / medical imaging, and land cover / remote sensing domains, plus one unrelated AI-security paper).

The message is ONE JSON object with these fields:
- "summary": a summary of older turns (may be empty).
- "recent_turns": the latest turns, each {{"role", "text"}}. An assistant turn also has "citations": a list of {{"paper", "pages"}} the answer was built from.
- "user_notes": a list of background notes about the user's interests (may be empty).
- "retry_feedback": null, or what the previous attempt missed (see below).
- "query": the user's new question.

The query may refer back to earlier turns (for example "what about its accuracy?" or "and for the other paper?"). Resolve every such reference so that each sub_query is fully self-contained and understandable WITHOUT the history. Use the citations of the recent assistant turns to find the paper a reference points to: "it", "that paper" or "the model" usually mean the paper(s) cited by the latest assistant turn. An assistant turn whose citations list is empty had no sources (for example a general-knowledge answer), so a reference to "it" cannot mean a paper from that turn. If several papers were cited and the reference is ambiguous, name the papers in the sub_query, or split it into one sub-question per paper. Treat user_notes only as background about the user's interests - never as facts about the papers.

Given the user's query, make these decisions:

1. needs_retrieval - ONE decision for the whole query. true if answering requires looking something up in the paper corpus. false only when the whole query is general knowledge answerable without retrieval (e.g. "what does CNN stand for"). If any part of the query needs the papers, it is true.

2. DECOMPOSE it into more than one sub-question ONLY when the parts need DIFFERENT lookups, for example different papers or topics. Do NOT split a question whose parts would retrieve the same passages: "what accuracy did model A and model B achieve, and which was better?" is ONE sub-question, because a single results passage answers all of it. Splitting it only makes the system do the same work twice. If in doubt, output ONE sub-question (the original, possibly cleaned up). Never output more than {MAX_SUB_QUERIES} sub-questions: if the query has more parts, keep the {MAX_SUB_QUERIES} that matter most.

3. For EACH sub-question, decide whether to EXPAND it into search variants.
   - expand: true only when the wording could miss relevant passages, for example when it uses an acronym, a vague term or a wording the papers may phrase differently. A precise question with specific terms should NOT be expanded.
   - variants: when expand is true, up to {EXPANSION_VARIANTS} phrasing variants (different wordings of the same sub-question, to widen search recall), including the original phrasing as one of them. When expand is false, only the original sub-question.

4. For EACH sub-question, decide how to search:
   - search_mode: "simple" or "hybrid" (semantic + keyword combined) - your judgment on whether keyword precision would help (e.g. specific named entities/acronyms benefit from hybrid).
   - listing: true when the question asks WHICH papers or documents do, use or report something across the corpus (e.g. "which papers use CNN"), so passages from many different papers are needed. Such questions are searched more widely. False otherwise.
   - images_required: true only if the query signals visual intent (e.g. "show me", "what does X look like", "example image of Y"). False for pure fact/relationship/conceptual questions.
   - tables_required: true only if the query asks for a table, or for the numbers a paper reports in a results table (e.g. "what is the recall in Table 3", "show the table comparing the models", "list the accuracy of each method in the results table"). False for questions that do not point to a table.

If retry_feedback is not null, this is a RETRY: the previous plan did not find enough. retry_feedback holds "missing" (what was missing), "look_for" (what to search for instead), "previous_plan" (the sub-questions tried, with their search mode and whether they were expanded) and "sources" (the papers the previous attempt retrieved from). Decide every point again using it and change the approach (a different decomposition, expansion, search mode or wording) instead of repeating the previous plan.

Respond ONLY as JSON in this exact format, with no other text:
{{
  "needs_retrieval": true,
  "sub_queries": [
    {{
      "sub_query": "...",
      "expand": false,
      "variants": ["..."],
      "search_mode": "simple",
      "listing": false,
      "images_required": false,
      "tables_required": false
    }}
  ]
}}"""


def build_user_content(query: str, feedback: dict = None, history: dict = None, notes: list = None) -> str:
    """The variable part of the prompt, as ONE JSON object: everything that changes per
    call, kept separate from SYSTEM_INSTRUCTION so the fixed part stays constant.

        {"summary": "...", "recent_turns": [{"role", "text", "citations"?}], "user_notes": [...],
         "retry_feedback": null | {"missing", "look_for", "previous_plan", "sources"}, "query": "..."}

    `history` is memory.history_payload(...) ({} when there is none), `notes` the user's recalled notes (a list), and on
    a retry `feedback` carries what the quality check said was missing and what to look for, the previous plan
    ({sub_query, search_mode, expand} per sub-question) and the papers the previous attempt retrieved from."""
    history = history or {}
    retry_feedback = None
    if feedback:
        retry_feedback = {
            "missing": feedback.get("missing", "unspecified"),
            "look_for": feedback.get("look_for", "unspecified"),
        }
        if feedback.get("previous_plan"):
            retry_feedback["previous_plan"] = [
                {"sub_query": p.get("sub_query", ""), "search_mode": p.get("search_mode", "simple"),
                 "expanded": bool(p.get("expand"))} for p in feedback["previous_plan"]
            ]
        if feedback.get("sources"):
            retry_feedback["sources"] = list(feedback["sources"])
    return json.dumps({
        "summary": history.get("summary", ""),
        "recent_turns": history.get("recent_turns", []),
        "user_notes": list(notes or []),
        "retry_feedback": retry_feedback,
        "query": query,
    }, ensure_ascii=False)


def build_prompt(query: str, feedback: dict = None, history: str = "", notes: str = "") -> str:
    """Kept for readability/testing - the full prompt as it would read
    (SYSTEM_INSTRUCTION + variable content combined)."""
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


def _clean_variants(sub_query: str, raw_variants, expand: bool) -> list:
    """The searches to run for one sub-question. Without expansion that is the
    sub-question itself. With expansion it is the sub-question first, then the
    model's other wordings (no duplicates), capped at EXPANSION_VARIANTS."""
    if not expand:
        return [sub_query]
    variants, seen = [], set()
    for text in [sub_query] + list(raw_variants or []):
        if not isinstance(text, str) or not text.strip():
            continue
        key = text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        variants.append(text.strip())
    return variants[:EXPANSION_VARIANTS]


def normalize_plan(raw):
    """Turns the planner's JSON into one clean shape, or None when it holds no
    usable sub-question:

        {"needs_retrieval": bool,
         "sub_queries": [{"sub_query", "expand", "variants", "search_mode", "listing", "images_required",
                          "tables_required"}]}

    Tolerant of the older output format (still served by Langfuse until the
    prompt is synced): per-sub-question needs_retrieval, no top-level flag, no
    `expand` (it counts as expanded when several variants were given), no
    `tables_required` (it counts as false), and extra fields such as
    data_source and complexity, which are ignored."""
    if not isinstance(raw, dict) or not isinstance(raw.get("sub_queries"), list):
        return None

    items = [item for item in raw["sub_queries"] if isinstance(item, dict)]
    sub_queries = []
    for item in items:
        text = item.get("sub_query")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        raw_variants = item.get("variants")
        expand = item.get("expand")
        if not isinstance(expand, bool):
            expand = isinstance(raw_variants, list) and len(raw_variants) > 1
        mode = item.get("search_mode")
        sub_queries.append({
            "sub_query": text,
            "expand": expand,
            "variants": _clean_variants(text, raw_variants, expand),
            "search_mode": mode if mode in SEARCH_MODES else "simple",
            "listing": item.get("listing") is True,
            "images_required": item.get("images_required") is True,
            "tables_required": item.get("tables_required") is True,
        })

    if not sub_queries:
        return None
    sub_queries = sub_queries[:MAX_SUB_QUERIES]  # the prompt asks for this, but an older prompt or a model may not obey

    needs_retrieval = raw.get("needs_retrieval")
    if not isinstance(needs_retrieval, bool):
        needs_retrieval = any(item.get("needs_retrieval", True) is not False for item in items)
    return {"needs_retrieval": needs_retrieval, "sub_queries": sub_queries}


def fallback_plan(query: str) -> dict:
    """What to do when the planner's answer cannot be used: search the papers
    once for the question as asked."""
    return {
        "needs_retrieval": True,
        "sub_queries": [{
            "sub_query": query, "expand": False, "variants": [query], "search_mode": "simple",
            "listing": False, "images_required": False, "tables_required": False,
        }],
    }


def transform_and_route(query: str, feedback: dict = None, client=None, history: dict = None, notes: list = None) -> dict:
    """Agent 1: decides whether the question needs the papers, how to split it,
    whether to expand it into variants and how to search. In a chat, `history`
    lets it resolve follow-up references into self-contained sub-queries (no
    extra LLM call needed). Returns the planner's parsed JSON (pass it to
    normalize_plan), or None if the response couldn't be parsed. `client` is an
    optional Gemini client override (tests); normally None - llm_connection
    picks the provider."""
    user_content = build_user_content(query, feedback, history, notes)
    instruction = get_prompt("transform-route", SYSTEM_INSTRUCTION)
    response = llm_connection.generate(instruction, user_content, task="query_planner", client=client)
    return parse_response(response.text.strip())
