from functools import partial
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from shared import langfuse_client
from shared import query_guardrail
from retrieval import query_cache
from retrieval import agent_transform_route
from retrieval import agent_quality_generate
from retrieval import output_guardrail
from retrieval import reranker as reranker_module
from retrieval import parallel
from retrieval import table_context
from retrieval import tracing
from shared.db import connection
from retrieval.retrieval_executor import (
    vector_search, hybrid_vector_search, embed_queries,
    get_images_for_parents, get_tables_for_parents, search_tables,
)
from retrieval.retrieval_config import MAX_RETRY_ATTEMPTS, LISTING_TOP_K


class SubQueryState(TypedDict):
    sub_query: str
    variants: list
    needs_retrieval: bool
    expand: bool
    search_mode: str
    listing: bool
    images_required: bool
    tables_required: bool
    images_missing: bool
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
    injection_score: Optional[float]  # Prompt Guard score of the query (None: model off/unavailable or regex already blocked)
    history: dict   # memory.history_payload(...), {} when there is none
    notes: list     # the user's recalled notes
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
    if result.get("injection_suspected") and not result["blocked"]:
        flags.append("prompt_injection_suspected")  # the model's flag only annotates; the query is still answered
    score = result.get("injection_score")
    if score is not None:
        # Lets the threshold be tuned from real traffic: filter the Langfuse scores by value.
        langfuse_client.score_current_trace(
            "prompt_injection_score", score,
            comment="flagged" if result.get("injection_suspected") else None,
        )
    return {
        **state,
        "injection_score": score,
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

    with tracing.step("cache_lookup", check=1) as span, connection() as conn:
        cached = query_cache.get_cached(conn, state["cleaned_query"])
        span.update(metadata={"hit": bool(cached)})

    if cached:
        langfuse_client.score_current_trace("cache_hit", 1.0, comment="check 1: before the planner")
        return {**state, "cache_hit": True, "final_answer": _with_safety_note(state, cached["answer"])}
    return {**state, "cache_hit": False}


def _plan_to_sub_queries(plan: dict, attempt: int = 0) -> list:
    """The planner's clean plan (agent_transform_route.normalize_plan) as the
    sub-query records the rest of the graph works on. `needs_retrieval` is one
    decision for the whole question, so every sub-query carries the same value."""
    return [{
        "sub_query": p["sub_query"], "variants": p["variants"], "expand": p["expand"],
        "needs_retrieval": plan["needs_retrieval"], "search_mode": p["search_mode"],
        "listing": p["listing"], "images_required": p["images_required"],
        "tables_required": p["tables_required"], "images_missing": False,
        "chunks": [], "images": [], "tables": [], "sufficient": False, "feedback": None,
        "attempt": attempt, "answer": None,
    } for p in plan["sub_queries"]]


def node_transform_route(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    result = agent_transform_route.transform_and_route(
        state["cleaned_query"], history=state.get("history", ""), notes=state.get("notes", "")
    )
    # An answer the planner gave that cannot be used counts as "search the papers
    # once for the question as asked".
    plan = agent_transform_route.normalize_plan(result) or agent_transform_route.fallback_plan(state["cleaned_query"])
    return {**state, "sub_queries": _plan_to_sub_queries(plan)}


def node_cache_check_2(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"] or not state.get("use_cache", True):
        return state

    combined_query = " | ".join(sq["sub_query"] for sq in state["sub_queries"])
    with tracing.step("cache_lookup", check=2, sub_queries=len(state["sub_queries"])) as span, connection() as conn:
        cached = query_cache.get_cached(conn, combined_query)
        span.update(metadata={"hit": bool(cached)})

    if cached:
        langfuse_client.score_current_trace("cache_hit", 1.0, comment="check 2: after the planner")
        return {**state, "cache_hit": True, "final_answer": _with_safety_note(state, cached["answer"])}
    # Remembered so the answer is WRITTEN under the same key it was READ with:
    # retrieval may later merge sub-queries, which would change the text.
    return {**state, "combined_key": combined_query}


def _group_by_paper(chunks: list) -> list:
    """Chunks regrouped so the ones from the same paper sit together, papers in
    the order they first appear, chunks keeping their order inside a paper."""
    first_seen = {}
    for c in chunks:
        first_seen.setdefault(c.get("source_pdf"), len(first_seen))
    return sorted(chunks, key=lambda c: first_seen[c.get("source_pdf")])


def _retrieve_for_sub_query(sq: dict) -> list:
    if not sq["needs_retrieval"]:
        return []

    # A listing question ("which papers use X") needs passages from many papers,
    # so every search is wider and the answer's context is grouped by paper.
    listing = bool(sq.get("listing"))
    wider = {"top_k": LISTING_TOP_K} if listing else {}

    def search(variant, query_vec):
        if sq["search_mode"] == "hybrid":
            return hybrid_vector_search(variant, query_vec=query_vec, **wider)
        return vector_search(variant, query_vec=query_vec, **wider)

    # All the variants are embedded in ONE call, then searched at the same time. The results are put back in variant
    # order, so the merged chunks are the same as when the searches ran one after another.
    variants = sq["variants"] or [sq["sub_query"]]
    vectors = embed_queries(variants)
    all_chunks = []
    for found in parallel.run_ordered([partial(search, v, vec) for v, vec in zip(variants, vectors)]):
        all_chunks.extend(found)

    seen = set()
    deduped = []
    for c in all_chunks:
        cid = c.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            deduped.append(c)
    return _group_by_paper(deduped) if listing else deduped


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
    question keeps both wordings, and unions chunks, images and tables."""
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
        target["listing"] = bool(target.get("listing") or sq.get("listing"))
        known = _chunk_ids(target)
        target["chunks"] = target["chunks"] + [c for c in sq["chunks"] if c.get("chunk_id") not in known]
        target["images_required"] = target["images_required"] or sq["images_required"]
        target["tables_required"] = target.get("tables_required", False) or sq.get("tables_required", False)
        seen_images = {i.get("image_file") for i in target["images"]}
        target["images"] = target["images"] + [i for i in sq["images"] if i.get("image_file") not in seen_images]
        seen_tables = {t.get("table_id") for t in target["tables"]}
        target["tables"] = target["tables"] + [t for t in sq["tables"] if t.get("table_id") not in seen_tables]

    return merged


def node_retrieval_executor(state: GraphState) -> GraphState:
    if state["blocked"] or state["cache_hit"]:
        return state

    # The sub-questions that still need chunks are retrieved at the same time too (results kept in order).
    pending = [sq for sq in state["sub_queries"] if not sq["sufficient"]]
    with tracing.timed("retrieval_s"):
        results = parallel.run_ordered([partial(_retrieve_for_sub_query, sq) for sq in pending])
    for sq, chunks in zip(pending, results):
        sq["chunks"] = chunks
        # Images and tables are NOT looked up here: the chunks can still change (the reranker, a retry), so they are
        # attached once, at the end, for the final chunks (node_attach_media).

    return {**state, "sub_queries": merge_overlapping_sub_queries(state["sub_queries"])}


def _parent_ids(chunks: list) -> list:
    """The distinct parent ids of the chunks, in order."""
    seen, ids = set(), []
    for chunk in chunks:
        parent_id = chunk.get("parent_id")
        if parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            ids.append(parent_id)
    return ids


def node_attach_media(state: GraphState) -> GraphState:
    """Runs once, after the quality check, the reranker and any retry are finished, so what is attached belongs to the
    FINAL chunks that go to the answer. Tables: those of the chunks' parents (always), plus, when the planner set
    `tables_required`, the best matches of the table search whatever the chunks are; merged, then size-capped. Images:
    only when the planner set `images_required`, those of the chunks' parents."""
    if state["blocked"] or state["cache_hit"]:
        return state

    for sq in state["sub_queries"]:
        if not sq["needs_retrieval"]:
            continue
        parent_ids = _parent_ids(sq["chunks"])
        from_parents = get_tables_for_parents(parent_ids) if parent_ids else []
        from_search = search_tables(sq["sub_query"]) if sq.get("tables_required") else []
        sq["tables"] = table_context.cap_tables(table_context.merge_tables(from_parents, from_search))
        if sq["images_required"]:
            sq["images"] = get_images_for_parents(parent_ids) if parent_ids else []
            sq["images_missing"] = not sq["images"]

    return state


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


# retry_route is only ever routed to when max_attempt==2 (see route_after_quality_check),
# so a sub-query that failed twice has attempt 2 here, and the plan made by the retry
# starts at 2 as well: its first quality check is attempt 3, the last one allowed.
RETRY_ATTEMPT = 2


def _retry_feedback(sub_queries: list, failing: list) -> dict:
    """What agent 1 is told on a retry: what the quality check said was missing
    and what to look for (for every sub-question that failed), the plan that was
    tried, and the papers it retrieved from, so the new plan changes approach."""
    def judgment(sq, key):
        return (sq.get("feedback") or {}).get(key, "")

    if len(failing) == 1:
        missing, look_for = judgment(failing[0], "missing"), judgment(failing[0], "look_for")
    else:
        missing = " | ".join(f"{sq['sub_query']}: {judgment(sq, 'missing')}" for sq in failing)
        look_for = " | ".join(f"{sq['sub_query']}: {judgment(sq, 'look_for')}" for sq in failing)
    return {
        "missing": missing,
        "look_for": look_for,
        "previous_plan": [
            {"sub_query": sq["sub_query"], "search_mode": sq["search_mode"],
             "expand": sq.get("expand", len(sq["variants"]) > 1)}
            for sq in sub_queries
        ],
        "sources": sorted({c["source_pdf"] for sq in failing for c in sq["chunks"] if c.get("source_pdf")}),
    }


def node_transform_route_retry(state: GraphState) -> GraphState:
    """Runs agent 1 again on the WHOLE original question, with the feedback.
    It decides everything again (splitting, expansion, search mode, whether to
    retrieve at all), and its new plan replaces the old one."""
    if state["blocked"] or state["cache_hit"]:
        return state

    failing = [sq for sq in state["sub_queries"] if not sq["sufficient"] and sq["attempt"] == RETRY_ATTEMPT]
    if not failing:
        return state

    result = agent_transform_route.transform_and_route(
        state["cleaned_query"], feedback=_retry_feedback(state["sub_queries"], failing),
        history=state.get("history", ""), notes=state.get("notes", ""),
    )
    plan = agent_transform_route.normalize_plan(result)
    if plan is None:
        return state  # nothing usable came back: keep the plan that was tried
    return {**state, "sub_queries": _plan_to_sub_queries(plan, attempt=RETRY_ATTEMPT)}


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
            )
        with tracing.step("guardrail_output", chunks=len(sq["chunks"])) as span:
            check = output_guardrail.check_output(answer, sq["chunks"])
            span.update(metadata={"flags": list(check["flags"])})
        answer = check["cleaned_answer"]  # PII/credential-redacted - this is what the user actually sees
        flags.extend(check["flags"])

        sq["answer"] = answer
        answers.append(answer)
        all_chunks.extend(sq["chunks"])

    final_answer = "\n\n".join(answers)

    if state.get("use_cache", True):
        combined_query = state.get("combined_key") or " | ".join(sq["sub_query"] for sq in state["sub_queries"])
        with tracing.step("cache_write"), connection() as conn:
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
    graph.add_node("attach_media", node_attach_media)                    # images and tables of the FINAL chunks
    graph.add_node("generate", node_generate)                            # agent 2

    graph.set_entry_point("input_guardrail")
    graph.add_conditional_edges("input_guardrail", route_after_input_guardrail, {"end": END, "cache_check": "cache_check"})
    graph.add_conditional_edges("cache_check", route_after_cache_check, {"end": END, "transform_route": "transform_route"})
    graph.add_edge("transform_route", "cache_check_2")
    graph.add_conditional_edges("cache_check_2", route_after_cache_check_2, {"end": END, "retrieval_executor": "retrieval_executor"})
    graph.add_edge("retrieval_executor", "quality_check")
    graph.add_conditional_edges(
        "quality_check", route_after_quality_check,
        {"generate": "attach_media", "reranker": "reranker", "retry_route": "retry_route"},
    )
    graph.add_edge("reranker", "quality_check")
    graph.add_edge("retry_route", "retrieval_executor")
    graph.add_edge("attach_media", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
