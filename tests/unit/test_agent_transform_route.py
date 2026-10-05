from retrieval import agent_transform_route as atr
from retrieval.retrieval_config import EXPANSION_VARIANTS, MAX_SUB_QUERIES


# ---------- the prompt ----------

def test_the_planner_is_told_not_to_split_questions_that_retrieve_the_same_passages():
    text = atr.SYSTEM_INSTRUCTION

    assert "Do NOT split a question whose parts would retrieve the same passages" in text
    assert "ONE sub-question" in text


def test_the_planner_decides_whether_to_retrieve_once_for_the_whole_query():
    text = atr.SYSTEM_INSTRUCTION

    assert "ONE decision for the whole query" in text
    assert '"needs_retrieval": true,\n  "sub_queries"' in text  # a top-level field, not one per sub-question


def test_expansion_is_a_decision_with_a_cap_on_the_number_of_variants():
    text = atr.SYSTEM_INSTRUCTION

    assert "decide whether to EXPAND" in text
    assert "A precise question with specific terms should NOT be expanded" in text
    assert f"up to {EXPANSION_VARIANTS} phrasing variants" in text
    assert '"expand": false' in text


def test_the_planner_can_flag_a_question_that_lists_papers():
    assert "listing" in atr.SYSTEM_INSTRUCTION
    assert '"listing": false' in atr.SYSTEM_INSTRUCTION


def test_the_planner_is_told_to_change_approach_on_a_retry():
    assert "RETRY" in atr.SYSTEM_INSTRUCTION
    assert "instead of repeating the previous plan" in atr.SYSTEM_INSTRUCTION


def test_the_prompt_no_longer_mentions_a_graph_source_or_a_complexity_label():
    text = atr.SYSTEM_INSTRUCTION

    assert "data_source" not in text
    assert "complexity" not in text
    assert "graph" not in text.lower()


def test_system_instruction_tells_the_model_to_resolve_follow_ups():
    assert "self-contained" in atr.SYSTEM_INSTRUCTION
    assert "never as facts about the papers" in atr.SYSTEM_INSTRUCTION


# ---------- the variable part of the prompt ----------

def test_build_prompt_includes_query():
    prompt = atr.build_prompt("What accuracy did the CNN model achieve?")
    assert "What accuracy did the CNN model achieve?" in prompt
    assert "IMPORTANT - this is a RETRY" not in prompt


def user_json(*args, **kwargs):
    import json

    return json.loads(atr.build_user_content(*args, **kwargs))


def test_build_prompt_includes_feedback_on_retry():
    feedback = {"missing": "accuracy numbers", "look_for": "results section"}
    prompt = atr.build_prompt("query", feedback=feedback)
    assert '"retry_feedback": {"missing": "accuracy numbers", "look_for": "results section"}' in prompt


def test_the_input_is_one_json_object_with_every_field():
    content = user_json("what is a CNN?")

    assert content == {"summary": "", "recent_turns": [], "user_notes": [], "retry_feedback": None, "query": "what is a CNN?"}


def test_the_history_the_notes_and_the_question_are_in_their_own_fields():
    history = {"summary": "Talked about YOLO.", "recent_turns": [
        {"role": "user", "text": "which YOLO?"},
        {"role": "assistant", "text": "YOLOv8", "citations": [{"paper": "lung-cancer/a.pdf", "pages": [4]}]}]}

    content = user_json("and its recall?", history=history, notes=["studies lung CT"])

    assert content["summary"] == "Talked about YOLO." and content["recent_turns"] == history["recent_turns"]
    assert content["user_notes"] == ["studies lung CT"] and content["query"] == "and its recall?"
    assert content["retry_feedback"] is None


def test_a_retry_carries_the_feedback_the_previous_plan_and_the_papers_it_retrieved_from():
    feedback = {
        "missing": "F1 numbers", "look_for": "Table 4",
        "previous_plan": [
            {"sub_query": "what F1 did VGG16 get?", "search_mode": "simple", "expand": False},
            {"sub_query": "which dataset?", "search_mode": "hybrid", "expand": True},
        ],
        "sources": ["a.pdf", "b.pdf"],
    }

    retry = user_json("what F1?", feedback=feedback)["retry_feedback"]

    assert retry["missing"] == "F1 numbers" and retry["look_for"] == "Table 4" and retry["sources"] == ["a.pdf", "b.pdf"]
    assert retry["previous_plan"] == [
        {"sub_query": "what F1 did VGG16 get?", "search_mode": "simple", "expanded": False},
        {"sub_query": "which dataset?", "search_mode": "hybrid", "expanded": True}]


def test_a_retry_without_a_plan_or_sources_leaves_those_out():
    retry = user_json("q", feedback={"missing": "x", "look_for": "y"})["retry_feedback"]
    assert retry == {"missing": "x", "look_for": "y"}


def test_text_with_quotes_and_accents_survives_the_json():
    content = user_json('what is "F1"? naïve', notes=['likes "CT" scans'])
    assert content["query"] == 'what is "F1"? naïve' and content["user_notes"] == ['likes "CT" scans']


def test_an_empty_string_history_or_notes_still_work_for_callers_that_pass_the_old_default():
    content = user_json("q", history="", notes="")
    assert content["recent_turns"] == [] and content["user_notes"] == []


def test_the_instruction_describes_the_json_input_and_how_to_use_the_citations():
    for field in ('"summary"', '"recent_turns"', '"citations"', '"user_notes"', '"retry_feedback"', '"query"'):
        assert field in atr.SYSTEM_INSTRUCTION
    assert "citations list is empty" in atr.SYSTEM_INSTRUCTION and "IMPORTANT - this is a RETRY" not in atr.SYSTEM_INSTRUCTION


# ---------- parsing the answer ----------

def test_parse_response_handles_plain_json():
    raw = '{"needs_retrieval": true, "sub_queries": [{"sub_query": "q1", "variants": ["q1"]}]}'
    result = atr.parse_response(raw)
    assert result["sub_queries"][0]["sub_query"] == "q1"


def test_parse_response_strips_markdown_fence():
    raw = '```json\n{"sub_queries": []}\n```'
    result = atr.parse_response(raw)
    assert result == {"sub_queries": []}


def test_parse_response_returns_none_on_garbage():
    assert atr.parse_response("not json") is None


# ---------- normalize_plan ----------

def plan_item(**overrides):
    item = {"sub_query": "what accuracy did VGG16 get?", "expand": False, "variants": ["what accuracy did VGG16 get?"],
            "search_mode": "simple", "listing": False, "images_required": False, "tables_required": False}
    item.update(overrides)
    return item


def test_a_complete_plan_comes_out_unchanged():
    raw = {"needs_retrieval": True, "sub_queries": [plan_item()]}

    assert atr.normalize_plan(raw) == {"needs_retrieval": True, "sub_queries": [plan_item()]}


def test_a_question_is_split_into_at_most_three_sub_questions():
    raw = {"needs_retrieval": True, "sub_queries": [plan_item(sub_query=f"question {n}", variants=[f"question {n}"]) for n in range(6)]}

    plan = atr.normalize_plan(raw)

    assert MAX_SUB_QUERIES == 3
    assert [sq["sub_query"] for sq in plan["sub_queries"]] == ["question 0", "question 1", "question 2"]


def test_the_planner_is_told_the_limit():
    assert f"more than {MAX_SUB_QUERIES} sub-questions" in atr.SYSTEM_INSTRUCTION


def test_a_plan_with_no_usable_sub_question_is_none():
    for raw in (None, {}, [], "text", {"sub_queries": []}, {"sub_queries": "x"}, {"sub_queries": [{}]},
                {"sub_queries": [{"sub_query": "   "}]}, {"sub_queries": ["not a dict"]}):
        assert atr.normalize_plan(raw) is None


def test_without_expansion_the_only_search_is_the_sub_question_itself():
    item = plan_item(expand=False, variants=["one", "two", "three"])

    sub = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [item]})["sub_queries"][0]

    assert sub["variants"] == ["what accuracy did VGG16 get?"]


def test_with_expansion_the_original_comes_first_and_duplicates_are_dropped():
    item = plan_item(expand=True, variants=["VGG16 accuracy", "WHAT ACCURACY DID VGG16 GET?", "VGG16 accuracy", "VGG16 score"])

    sub = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [item]})["sub_queries"][0]

    assert sub["variants"] == ["what accuracy did VGG16 get?", "VGG16 accuracy", "VGG16 score"]


def test_expansion_is_capped_at_the_configured_number_of_variants():
    many = [f"wording {i}" for i in range(10)]

    sub = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [plan_item(expand=True, variants=many)]})["sub_queries"][0]

    assert len(sub["variants"]) == EXPANSION_VARIANTS
    assert sub["variants"][0] == "what accuracy did VGG16 get?"


def test_expansion_with_no_variants_still_searches_the_sub_question():
    sub = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [plan_item(expand=True, variants=None)]})["sub_queries"][0]

    assert sub["variants"] == ["what accuracy did VGG16 get?"]


def test_an_unknown_search_mode_means_simple_and_flags_must_be_real_booleans():
    item = plan_item(search_mode="semantic", listing="yes", images_required=1)

    sub = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [item]})["sub_queries"][0]

    assert (sub["search_mode"], sub["listing"], sub["images_required"]) == ("simple", False, False)


def test_hybrid_listing_and_image_requests_are_kept():
    item = plan_item(search_mode="hybrid", listing=True, images_required=True)

    sub = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [item]})["sub_queries"][0]

    assert (sub["search_mode"], sub["listing"], sub["images_required"]) == ("hybrid", True, True)


def test_needs_retrieval_false_for_the_whole_query_is_kept():
    plan = atr.normalize_plan({"needs_retrieval": False, "sub_queries": [plan_item(sub_query="what does CNN stand for?")]})

    assert plan["needs_retrieval"] is False


def test_the_older_output_format_is_still_understood():
    """Langfuse serves the old prompt until it is synced: per-sub-question needs_retrieval,
    no `expand`, and the retired data_source and complexity fields."""
    old = {"sub_queries": [
        {"sub_query": "q1", "variants": ["q1", "q1 reworded"], "needs_retrieval": True, "data_source": "graph",
         "search_mode": "hybrid", "images_required": False, "complexity": "complex"},
        {"sub_query": "q2", "variants": ["q2"], "needs_retrieval": False, "data_source": "vector",
         "search_mode": "simple", "images_required": False, "complexity": "simple"},
    ]}

    plan = atr.normalize_plan(old)

    assert plan["needs_retrieval"] is True  # at least one part needs the papers
    first, second = plan["sub_queries"]
    assert (first["expand"], first["variants"], first["search_mode"]) == (True, ["q1", "q1 reworded"], "hybrid")
    assert (second["expand"], second["variants"]) == (False, ["q2"])
    assert "data_source" not in first and "complexity" not in first


def test_the_older_format_with_every_part_answered_from_knowledge_needs_no_retrieval():
    old = {"sub_queries": [{"sub_query": "what is a CNN?", "variants": ["what is a CNN?"], "needs_retrieval": False}]}

    assert atr.normalize_plan(old)["needs_retrieval"] is False


def test_unusable_sub_questions_are_skipped_but_usable_ones_are_kept():
    raw = {"needs_retrieval": True, "sub_queries": [{"sub_query": ""}, "junk", plan_item(sub_query="kept")]}

    assert [s["sub_query"] for s in atr.normalize_plan(raw)["sub_queries"]] == ["kept"]


def test_a_fallback_plan_searches_the_papers_once_for_the_question_as_asked():
    plan = atr.fallback_plan("my question")

    assert plan == {"needs_retrieval": True, "sub_queries": [{
        "sub_query": "my question", "expand": False, "variants": ["my question"], "search_mode": "simple",
        "listing": False, "images_required": False, "tables_required": False,
    }]}


# ---------- calling the model ----------

def test_transform_and_route_runs_on_the_query_planner_task(monkeypatch):
    seen = {}

    class R:
        text = '{"sub_queries": []}'

    def fake_generate(system, user, **kwargs):
        seen.update(kwargs)
        return R()

    monkeypatch.setattr(atr.llm_connection, "generate", fake_generate)

    atr.transform_and_route("what is a CNN?")

    assert seen["task"] == "query_planner"


def test_transform_and_route_sends_the_retry_feedback_to_the_model(monkeypatch):
    seen = {}

    class R:
        text = '{"sub_queries": []}'

    monkeypatch.setattr(atr.llm_connection, "generate", lambda system, user, **k: seen.update(user=user) or R())

    atr.transform_and_route("what F1?", feedback={"missing": "F1 numbers", "look_for": "Table 4"})

    assert '"retry_feedback": {"missing": "F1 numbers"' in seen["user"]


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_prompt = None

    def generate_content(self, **kwargs):
        self.last_prompt = kwargs.get("contents")
        return FakeResponse(self.response_text)


class FakeCaches:
    """Simulates caching being unavailable (e.g. no billing enabled) -
    the common free-tier case - so transform_and_route falls back to
    inlining the instructions, same as it would in production."""
    def create(self, model, config):
        raise Exception("caching unavailable in test")


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)
        self.caches = FakeCaches()


def test_transform_and_route_calls_gemini_and_the_answer_normalizes_to_a_plan():
    response_json = (
        '{"needs_retrieval": true, "sub_queries": [{"sub_query": "which papers use CNN", "expand": false, '
        '"variants": ["which papers use CNN"], "search_mode": "hybrid", "listing": true, "images_required": false}]}'
    )
    fake_client = FakeClient(response_json)

    result = atr.transform_and_route("which papers use CNN", client=fake_client)
    plan = atr.normalize_plan(result)

    assert plan["sub_queries"][0]["listing"] is True and plan["sub_queries"][0]["search_mode"] == "hybrid"
    assert "which papers use CNN" in fake_client.models.last_prompt


# ---------- tables_required ----------

def test_the_planner_is_told_about_tables_required_and_the_json_shows_it():
    assert "tables_required: true only if the query asks for a table" in atr.SYSTEM_INSTRUCTION
    assert '"tables_required": false' in atr.SYSTEM_INSTRUCTION


def test_tables_required_is_kept_only_when_it_is_exactly_true():
    for value, expected in ((True, True), (False, False), ("yes", False), (1, False), (None, False)):
        plan = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [plan_item(tables_required=value)]})
        assert plan["sub_queries"][0]["tables_required"] is expected


def test_a_plan_from_the_old_prompt_without_the_field_counts_as_no_table_wanted():
    plan = atr.normalize_plan({"needs_retrieval": True, "sub_queries": [plan_item()]})
    assert plan["sub_queries"][0]["tables_required"] is False
