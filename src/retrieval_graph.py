from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

import query_guardrail
import query_cache
import agent_transform_route
import agent_quality_generate
import output_guardrail
import reranker as reranker_module
from db import connection
from retrieval_executor import (
    vector_search, hybrid_vector_search, graph_search,
    get_images_for_sections, get_tables_for_sections,
)
from retrieval_config import MAX_RETRY_ATTEMPTS


class SubQueryState(TypedDict):
    sub_query: str
    variants: list
    needs_retrieval: bool
    data_source: str
    search_mode: str
    images_required: bool
    complexity: str
    chunks: list
    images: list
    tables: list
    sufficient: bool
    feedback: Optional[dict]
    attempt: int
    answer: Optional[str]


class GraphState(TypedDict):
    raw_query: str
    cleaned_query: Optional[str]
    blocked: bool
    block_reason: Optional[str]
    cache_hit: bool
    sub_queries: list
    final_answer: Optional[str]
    guardrail_flags: list
    history: str
    notes: str
    use_cache: bool  # False for evaluation runs: never read or write the answer cache
    combined_key: Optional[str]  # cache #2 key, computed BEFORE retrieval may merge sub-queries


# Fixed text (no LLM involved) put at the top of an answer when the user asked
# for personal medical advice. The system only summarises research papers.
MEDICAL_NOTE = (
    "*Note: I summarise published research papers. I am not a medical professional, and this is not "
    "medical advice. Please discuss decisions about your own health with your doctor or care team.*"
)


def _with_safety_note(state: GraphState, answer: Optional[str]) -> Optional[str]:
    """Adds the medical note when the input guardrail raised the flag. Applied on
    EVERY path that returns an answer, including cache hits: a cached answer
    may have been stored for a neutral phrasing of the same question."""
    if answer and query_guardrail.MEDICAL_ADVICE_FLAG in state.get("guardrail_flags", []) and MEDICAL_NOTE not in answer:
        return f"{MEDICAL_NOTE}\n\n{answer}"
    return answer


def node_input_guardrail(state: GraphState) -> GraphState:
    result = query_guardrail.check_input(state["raw_query"])
    flags = list(state.get("guardrail_flags", []))
    if result["medical_advice_detected"]:
        flags.append(query_guardrail.MEDICAL_ADVICE_FLAG)  # detected here, acted on when the answer is returned
    return {
        **state,
        "cleaned_query": result["cleaned_query"],
        "blocked": result["blocked"],
        "block_reason": ", ".join(result["reasons"]) if result["blocked"] else None,
        "guardrail_flags": flags,
    }


def node_cache_check(state: GraphState) -> GraphState:
    if state["blocked"]:
        return state

    # A follow-up ("what about its accuracy?") only makes sense with its
    # conversation, and user notes can shape routing, so neither may use the
    # raw query text as a SHARED cache key. Cache check #2 still runs on the
    # resolved, self-contained sub-queries.
    if state.get("history") or state.get("notes") or not state.get("use_cache", True):
        return {**state, "cache_hit": False}

    with connection() as conn:
        cached = query_cache.get_cached(conn, state["cleaned_query"])

    if cached:
        return {**state, "cache_hit": True, "final_answer": _with_safety_note(state, cached["answer"])}
    return {**state, "cache_hit": False}


def node_transform_route(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    result = agent_transform_route.transform_and_route(
        state["cleaned_query"], history=state.get("history", ""), notes=state.get("notes", "")
    )
    if result is None or not result.get("sub_queries"):
        sub_queries = [{
            "sub_query": state["cleaned_query"], "variants": [state["cleaned_query"]],
            "needs_retrieval": True, "data_source": "vector", "search_mode": "simple",
            "images_required": False, "complexity": "complex",
        }]
    else:
        sub_queries = result["sub_queries"]

    for sq in sub_queries:
        # A missing or unrecognised label counts as "complex": better to spend
        # the stronger model than to answer a hard question on the fast tier.
        if sq.get("complexity") not in ("simple", "complex"):
            sq["complexity"] = "complex"
        sq["chunks"] = []
        sq["images"] = []
        sq["tables"] = []
        sq["sufficient"] = False
        sq["feedback"] = None
        sq["attempt"] = 0
        sq["answer"] = None

    return {**state, "sub_queries": sub_queries}


def node_cache_check_2(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"] or not state.get("use_cache", True):
        return state

    combined_query = " | ".join(sq["sub_query"] for sq in state["sub_queries"])
    with connection() as conn:
        cached = query_cache.get_cached(conn, combined_query)

    if cached:
        return {**state, "cache_hit": True, "final_answer": _with_safety_note(state, cached["answer"])}
    # Remembered so the answer is WRITTEN under the same key it was READ with:
    # retrieval may later merge sub-queries, which would change the text.
    return {**state, "combined_key": combined_query}


def _retrieve_for_sub_query(sq: dict) -> list:
    if not sq["needs_retrieval"]:
        return []

    def plain_search(variant):
        if sq["search_mode"] == "hybrid":
            return hybrid_vector_search(variant)
        return vector_search(variant)

    all_chunks = []
    for variant in sq["variants"] or [sq["sub_query"]]:
        if sq["data_source"] in ("vector", "both"):
            all_chunks.extend(plain_search(variant))

        if sq["data_source"] in ("graph", "both"):
            # The graph is a FILTER: it names the papers worth looking at, and the
            # passages come from a vector search limited to those papers.
            graph_results = graph_search(variant)
            if graph_results:
                papers = list({r["source_pdf"] for r in graph_results})
                all_chunks.extend(vector_search(variant, source_pdfs=papers))
            elif sq["data_source"] == "graph":
                # Graph alone found nothing to filter by: fall back to the normal
                # search, so the question is never left with no context at all.
                all_chunks.extend(plain_search(variant))

    seen = set()
    deduped = []
    for c in all_chunks:
        cid = c.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            deduped.append(c)
    return deduped


OVERLAP_MERGE_THRESHOLD = 0.8


def _chunk_ids(sq: dict) -> set:
    return {c.get("chunk_id") for c in sq["chunks"] if c.get("chunk_id") is not None}


def merge_overlapping_sub_queries(sub_queries: list, threshold: float = OVERLAP_MERGE_THRESHOLD) -> list:
    """Sub-questions that retrieved (almost) the same passages are answered
    together: each one would otherwise pay for its own quality check and its
    own answer over identical context. Safety net for a planner that split a
    question needlessly.

    Only first-pass sub-queries are merged (attempt == 0), so the retry loop
    is never disturbed. Overlap is measured against the SMALLER chunk set, so
    a sub-query fully contained in another counts as overlapping. The merged
    question keeps both wordings, takes the stronger complexity, and unions
    chunks and images."""
    merged = []
    for sq in sub_queries:
        target = None
        if sq["needs_retrieval"] and sq["attempt"] == 0 and sq["chunks"]:
            mine = _chunk_ids(sq)
            for other in merged:
                if not (other["needs_retrieval"] and other["attempt"] == 0 and other["chunks"]):
                    continue
                theirs = _chunk_ids(other)
                smaller = min(len(mine), len(theirs))
                if smaller and len(mine & theirs) / smaller >= threshold:
                    target = other
                    break

        if target is None:
            merged.append(sq)
            continue

        target["sub_query"] = f"{target['sub_query'].rstrip()} Also: {sq['sub_query'].lstrip()}"
        if "complex" in (target.get("complexity"), sq.get("complexity")):
            target["complexity"] = "complex"
        known = _chunk_ids(target)
        target["chunks"] = target["chunks"] + [c for c in sq["chunks"] if c.get("chunk_id") not in known]
        target["images_required"] = target["images_required"] or sq["images_required"]
        seen_images = {i.get("image_file") for i in target["images"]}
        target["images"] = target["images"] + [i for i in sq["images"] if i.get("image_file") not in seen_images]
        seen_tables = {t.get("table_id") for t in target["tables"]}
        target["tables"] = target["tables"] + [t for t in sq["tables"] if t.get("table_id") not in seen_tables]

    return merged


def node_retrieval_executor(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    for sq in state["sub_queries"]:
        if sq["sufficient"]:
            continue
        sq["chunks"] = _retrieve_for_sub_query(sq)
        chunk_ids = [c["chunk_id"] for c in sq["chunks"]]
        if sq["images_required"]:
            sq["images"] = get_images_for_sections(chunk_ids=chunk_ids) if chunk_ids else []
        # Unlike images (gated behind visual intent), tables are just another
        # source of factual/numeric content - fetched whenever there are
        # retrieved chunks to look near, no separate "tables_required" flag.
        sq["tables"] = get_tables_for_sections(chunk_ids=chunk_ids) if chunk_ids else []

    return {**state, "sub_queries": merge_overlapping_sub_queries(state["sub_queries"])}


def node_quality_check(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    for sq in state["sub_queries"]:
        if sq["sufficient"]:
            continue
        if not sq["needs_retrieval"]:
            # Nothing was retrieved on purpose (general-knowledge sub-query) -
            # there's no retrieval quality to judge, so don't burn a judgment
            # call or trigger the retry loop for this one.
            sq["sufficient"] = True
            continue
        sq["attempt"] += 1
        judgment = agent_quality_generate.check_quality(sq["sub_query"], sq["chunks"])
        sq["sufficient"] = judgment.get("sufficient", False)
        sq["feedback"] = {"missing": judgment.get("missing", ""), "look_for": judgment.get("look_for", "")}

    return state


def node_reranker(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    for sq in state["sub_queries"]:
        # Reranker is only ever routed to when max_attempt==1 (see
        # route_after_quality_check) - so a sub-query's own attempt is 1.
        if sq["sufficient"] or sq["attempt"] != 1:
            continue
        sq["chunks"] = reranker_module.rerank(sq["sub_query"], sq["chunks"])

    return state


def node_transform_route_retry(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    for sq in state["sub_queries"]:
        # retry_route is only ever routed to when max_attempt==2 (see
        # route_after_quality_check) - so a sub-query's own attempt is 2 here,
        # not 3. Fixed 2026-09-25: same off-by-one as the reranker guard above
        # - this previously checked attempt==3, which never matched, so the
        # feedback-informed re-routing silently never happened either.
        if sq["sufficient"] or sq["attempt"] != 2:
            continue
        result = agent_transform_route.transform_and_route(sq["sub_query"], feedback=sq["feedback"])
        if result and result.get("sub_queries"):
            updated = result["sub_queries"][0]
            sq["data_source"] = updated.get("data_source", sq["data_source"])
            sq["search_mode"] = updated.get("search_mode", sq["search_mode"])
            sq["variants"] = updated.get("variants", sq["variants"])

    return state


def node_generate(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    answers = []
    all_chunks = []
    flags = list(state.get("guardrail_flags", []))  # keeps flags raised earlier (input guardrail)
    for sq in state["sub_queries"]:
        if not sq["needs_retrieval"]:
            answer = agent_quality_generate.generate_general_knowledge_answer(sq["sub_query"])
        else:
            low_confidence = not sq["sufficient"]
            answer = agent_quality_generate.generate_answer(
                sq["sub_query"], sq["chunks"], tables=sq["tables"], low_confidence=low_confidence,
                complexity=sq.get("complexity", "complex"),
            )
        check = output_guardrail.check_output(answer, sq["chunks"])
        answer = check["cleaned_answer"]  # PII/credential-redacted - this is what the user actually sees
        flags.extend(check["flags"])

        sq["answer"] = answer
        answers.append(answer)
        all_chunks.extend(sq["chunks"])

    final_answer = "\n\n".join(answers)

    if state.get("use_cache", True):
        combined_query = state.get("combined_key") or " | ".join(sq["sub_query"] for sq in state["sub_queries"])
        with connection() as conn:
            if not (state.get("history") or state.get("notes")):
                query_cache.write_cache(conn, state["cleaned_query"], all_chunks, final_answer)
            query_cache.write_cache(conn, combined_query, all_chunks, final_answer)

    # The note is added AFTER caching, so the shared cache never stores it.
    return {**state, "final_answer": _with_safety_note(state, final_answer), "guardrail_flags": flags}


def route_after_input_guardrail(state: GraphState) -> str:
    return "end" if state["blocked"] else "cache_check"


def route_after_cache_check(state: GraphState) -> str:
    return "end" if state["cache_hit"] else "transform_route"


def route_after_cache_check_2(state: GraphState) -> str:
    return "end" if state["cache_hit"] else "retrieval_executor"


def route_after_quality_check(state: GraphState) -> str:
    if state["blocked"] or state["cache_hit"]:
        return "generate"

    all_sufficient = all(sq["sufficient"] for sq in state["sub_queries"])
    if all_sufficient:
        return "generate"

    max_attempt = max(sq["attempt"] for sq in state["sub_queries"])
    if max_attempt >= MAX_RETRY_ATTEMPTS:
        return "generate"
    if max_attempt == 1:
        return "reranker"
    return "retry_route"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("input_guardrail", node_input_guardrail)
    graph.add_node("cache_check", node_cache_check)
    graph.add_node("transform_route", node_transform_route)              # agent 1
    graph.add_node("cache_check_2", node_cache_check_2)
    graph.add_node("retrieval_executor", node_retrieval_executor)
    graph.add_node("quality_check", node_quality_check)                  # agent 2
    graph.add_node("reranker", node_reranker)
    graph.add_node("retry_route", node_transform_route_retry)            # agent 1
    graph.add_node("generate", node_generate)                            # agent 2

    graph.set_entry_point("input_guardrail")
    graph.add_conditional_edges("input_guardrail", route_after_input_guardrail, {"end": END, "cache_check": "cache_check"})
    graph.add_conditional_edges("cache_check", route_after_cache_check, {"end": END, "transform_route": "transform_route"})
    graph.add_edge("transform_route", "cache_check_2")
    graph.add_conditional_edges("cache_check_2", route_after_cache_check_2, {"end": END, "retrieval_executor": "retrieval_executor"})
    graph.add_edge("retrieval_executor", "quality_check")
    graph.add_conditional_edges(
        "quality_check", route_after_quality_check,
        {"generate": "generate", "reranker": "reranker", "retry_route": "retry_route"},
    )
    graph.add_edge("reranker", "quality_check")
    graph.add_edge("retry_route", "retrieval_executor")
    graph.add_edge("generate", END)

    return graph.compile()
