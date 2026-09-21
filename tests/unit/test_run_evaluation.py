import json

import run_evaluation as re_


def make_result(answer="98% accuracy [a.pdf, p.4]", chunks=None, flags=None):
    if chunks is None:
        chunks = [{"source_pdf": "a.pdf", "page_start": 3, "page_end": 5, "parent_text": "parent A", "text": "child A"}]
    return {
        "final_answer": answer,
        "blocked": False,
        "cache_hit": False,
        "guardrail_flags": flags or [],
        "sub_queries": [{"chunks": chunks}],
    }


RECORD = {
    "input": "What accuracy?",
    "expected_output": "98%",
    "context": ["ctx"],
    "source_pdf": "a.pdf",
    "domain": "lung-cancer",
}


def test_load_golden_set_reads_jsonl_and_respects_limit(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text("\n".join(json.dumps({"input": f"q{i}"}) for i in range(3)) + "\n\n", encoding="utf-8")

    assert len(re_.load_golden_set(path)) == 3
    assert [r["input"] for r in re_.load_golden_set(path, limit=2)] == ["q0", "q1"]


def test_collect_chunks_flattens_all_sub_queries():
    result = {"sub_queries": [{"chunks": [{"a": 1}]}, {"chunks": [{"b": 2}, {"c": 3}]}, {}]}
    assert len(re_.collect_chunks(result)) == 3


def test_collect_contexts_prefers_parent_text_and_dedupes():
    chunks = [
        {"parent_text": "P", "text": "c1"},
        {"parent_text": "P", "text": "c2"},
        {"text": "only child"},
        {"text": ""},
    ]
    assert re_.collect_contexts(chunks) == ["P", "only child"]


def test_deterministic_metrics_source_hit_and_grounded_citation():
    metrics = re_.deterministic_metrics(RECORD, make_result(), latency_s=1.5)

    assert metrics["source_hit"] == 1.0
    assert metrics["citation_verified_rate"] == 1.0
    assert metrics["is_grounded"] == 1.0
    assert metrics["latency_s"] == 1.5
    assert metrics["answer_chars"] == float(len("98% accuracy [a.pdf, p.4]"))


def test_deterministic_metrics_source_miss_and_bad_citation():
    result = make_result(answer="claim [z.pdf, p.9]")
    metrics = re_.deterministic_metrics(RECORD, result, latency_s=0.1)

    assert metrics["citation_verified_rate"] == 0.0
    assert metrics["is_grounded"] == 0.0

    other = {"source_pdf": "other.pdf"}
    assert re_.deterministic_metrics(other, make_result(), 0.1)["source_hit"] == 0.0


def test_deterministic_metrics_no_citations_gives_none_rate():
    metrics = re_.deterministic_metrics(RECORD, make_result(answer="no citations here"), 0.1)
    assert metrics["citation_verified_rate"] is None
    assert metrics["is_grounded"] == 0.0


def test_evaluate_record_builds_flat_row():
    row = re_.evaluate_record(lambda q: make_result(), RECORD)

    assert row["input"] == "What accuracy?"
    assert row["expected_source"] == "a.pdf"
    assert row["answer"] == "98% accuracy [a.pdf, p.4]"
    assert row["contexts"] == ["parent A"]
    assert "source_hit" in row["metrics"]


def test_run_pipeline_invokes_once_per_record():
    seen = []

    def invoke(query):
        seen.append(query)
        return make_result()

    rows = re_.run_pipeline(invoke, [RECORD, {**RECORD, "input": "second"}])

    assert seen == ["What accuracy?", "second"]
    assert len(rows) == 2


def test_aggregate_means_and_skips_none():
    rows = [
        {"metrics": {"source_hit": 1.0, "citation_verified_rate": None}},
        {"metrics": {"source_hit": 0.0, "citation_verified_rate": 0.5}},
    ]
    summary = re_.aggregate(rows)

    assert summary["source_hit"] == 0.5
    assert summary["citation_verified_rate"] == 0.5


def test_aggregate_empty_rows():
    assert re_.aggregate([]) == {}
