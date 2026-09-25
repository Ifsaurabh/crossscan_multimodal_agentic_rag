"""classify_domain: one cheap LLM call per batch of raw text snippets returns each paper's
title and research domain. The model call is faked; no network or quota is used."""
import json
import types

import pytest

import classify_domain as cd


def reply(text):
    return types.SimpleNamespace(text=text)


def install_llm(monkeypatch, replier=None):
    """Replaces the model call. By default it answers every snippet with 'Title N' / 'Lung Cancer'."""
    calls = []

    def default_replier(instruction, content):
        count = len(content.split("\n"))
        return json.dumps([{"title": f"Title {i + 1}", "domain": "Lung Cancer"} for i in range(count)])

    def fake_generate(instruction, content, tier=None, **kwargs):
        calls.append({"instruction": instruction, "content": content, "tier": tier})
        return reply((replier or default_replier)(instruction, content))

    monkeypatch.setattr(cd, "generate", fake_generate)
    return calls


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    monkeypatch.setattr(cd, "TEXT_DIR", text_dir)
    monkeypatch.setattr(cd, "OUTPUT_PATH", tmp_path / "domain_classification.json")
    return tmp_path


def write_text_doc(workspace, stem, blocks, source_pdf=None):
    doc = {"source_pdf": source_pdf or f"{stem}.pdf", "blocks": [{"label": "text", "text": t, "page": 1} for t in blocks]}
    (workspace / "text" / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")


def read_output(workspace):
    return json.loads((workspace / "domain_classification.json").read_text(encoding="utf-8"))


# ---------- slugify_title ----------

def test_slugify_title_produces_kebab_case():
    assert cd.slugify_title("Protecting Context and Prompts: Deterministic Security") == \
        "protecting-context-and-prompts-deterministic"


def test_slugify_title_handles_empty_text():
    assert cd.slugify_title("") == "unknown-domain"


def test_slugify_title_with_only_symbols_is_unknown():
    assert cd.slugify_title("!!! --- ???") == "unknown-domain"


def test_slugify_title_keeps_at_most_five_words_and_drops_punctuation():
    assert cd.slugify_title("Lung, Cancer: Imaging (CT) & Deep Learning Methods") == "lung-cancer-imaging-ct-deep"


# ---------- get_document_snippet ----------

def test_the_snippet_is_the_leading_text_of_all_blocks_in_reading_order(workspace):
    write_text_doc(workspace, "sample", ["OPEN ACCESS", "Deep learning for lung cancer", "Abstract text follows"])

    snippet = cd.get_document_snippet(workspace / "text" / "sample.json")

    assert snippet == "OPEN ACCESS Deep learning for lung cancer Abstract text follows"


def test_the_snippet_is_cut_at_the_default_length(workspace):
    write_text_doc(workspace, "sample", ["x" * 5000])

    assert len(cd.get_document_snippet(workspace / "text" / "sample.json")) == cd.SNIPPET_CHARS == 800


def test_the_snippet_length_can_be_changed(workspace):
    write_text_doc(workspace, "sample", ["abcdefghij"])

    assert cd.get_document_snippet(workspace / "text" / "sample.json", max_chars=4) == "abcd"


def test_a_document_with_no_blocks_has_an_empty_snippet(workspace):
    write_text_doc(workspace, "sample", [])

    assert cd.get_document_snippet(workspace / "text" / "sample.json") == ""


# ---------- classify_snippets_batch ----------

def test_a_batch_is_sent_as_one_numbered_list_on_the_fast_tier(monkeypatch):
    calls = install_llm(monkeypatch)

    cd.classify_snippets_batch(["first snippet", "second snippet"])

    assert len(calls) == 1
    assert calls[0]["content"] == "1. first snippet\n2. second snippet"
    assert calls[0]["tier"] == "fast"
    assert calls[0]["instruction"] == cd.DOMAIN_SYSTEM_INSTRUCTION


def test_titles_are_trimmed_and_domains_become_slugs(monkeypatch):
    install_llm(monkeypatch, lambda i, c: json.dumps([{"title": "  A Real Title  ", "domain": "Land Cover Remote Sensing"}]))

    assert cd.classify_snippets_batch(["s"]) == [{"title": "A Real Title", "domain": "land-cover-remote-sensing"}]


@pytest.mark.parametrize("wrapper", ["```json\n{}\n```", "```\n{}\n```", "{}", "  {}  \n"])
def test_code_fences_and_whitespace_around_the_json_are_tolerated(monkeypatch, wrapper):
    payload = json.dumps([{"title": "T", "domain": "ai security"}])
    install_llm(monkeypatch, lambda i, c: wrapper.replace("{}", payload))

    assert cd.classify_snippets_batch(["s"]) == [{"title": "T", "domain": "ai-security"}]


def test_a_wrong_number_of_results_is_refused(monkeypatch):
    install_llm(monkeypatch, lambda i, c: json.dumps([{"title": "Only one", "domain": "x"}]))

    with pytest.raises(ValueError, match="Batch mismatch: 2 snippets in, 1 results back"):
        cd.classify_snippets_batch(["a", "b"])


def test_a_reply_that_is_not_json_raises(monkeypatch):
    install_llm(monkeypatch, lambda i, c: "Sorry, I cannot help with that.")

    with pytest.raises(json.JSONDecodeError):
        cd.classify_snippets_batch(["a"])


def test_a_result_missing_a_field_raises_instead_of_guessing(monkeypatch):
    install_llm(monkeypatch, lambda i, c: json.dumps([{"title": "No domain here"}]))

    with pytest.raises(KeyError):
        cd.classify_snippets_batch(["a"])


def test_an_empty_domain_becomes_the_unknown_slug(monkeypatch):
    install_llm(monkeypatch, lambda i, c: json.dumps([{"title": "T", "domain": ""}]))

    assert cd.classify_snippets_batch(["a"])[0]["domain"] == "unknown-domain"


# ---------- classify_domain (the whole stage) ----------

def test_every_document_gets_a_title_and_a_domain_in_file_order(workspace, monkeypatch, capsys):
    write_text_doc(workspace, "b_paper", ["Land cover mapping"], source_pdf="B Paper.pdf")
    write_text_doc(workspace, "a_paper", ["Lung nodules"], source_pdf="A Paper.pdf")
    install_llm(monkeypatch)

    cd.classify_domain()

    assert read_output(workspace) == [
        {"source_pdf": "A Paper.pdf", "title": "Title 1", "domain": "lung-cancer"},
        {"source_pdf": "B Paper.pdf", "title": "Title 2", "domain": "lung-cancer"},
    ]
    assert "Found 2 documents to classify" in capsys.readouterr().out


def test_the_output_is_the_list_shape_the_loaders_read(workspace, monkeypatch):
    write_text_doc(workspace, "sample", ["text"], source_pdf="Sample.pdf")
    install_llm(monkeypatch)

    cd.classify_domain()

    domain_map = {d["source_pdf"]: d["domain"] for d in read_output(workspace)}  # exactly what load_vector_db / load_graph_db do
    assert domain_map == {"Sample.pdf": "lung-cancer"}


def test_documents_are_classified_in_batches_and_order_is_preserved(workspace, monkeypatch):
    monkeypatch.setattr(cd, "BATCH_SIZE", 2)
    for i in range(5):
        write_text_doc(workspace, f"doc{i}", [f"snippet {i}"], source_pdf=f"doc{i}.pdf")
    calls = install_llm(monkeypatch)

    cd.classify_domain()

    assert [len(call["content"].split("\n")) for call in calls] == [2, 2, 1]  # three model calls, not five
    assert [r["source_pdf"] for r in read_output(workspace)] == [f"doc{i}.pdf" for i in range(5)]


def test_only_the_snippet_is_sent_to_the_model_not_the_whole_document(workspace, monkeypatch):
    write_text_doc(workspace, "big", ["word " * 5000])
    calls = install_llm(monkeypatch)

    cd.classify_domain()

    assert len(calls[0]["content"]) <= len("1. ") + cd.SNIPPET_CHARS


def test_no_documents_means_no_model_call_and_an_empty_result(workspace, monkeypatch):
    calls = install_llm(monkeypatch)

    cd.classify_domain()

    assert calls == [] and read_output(workspace) == []


def test_a_failed_model_call_stops_the_stage_and_writes_no_partial_file(workspace, monkeypatch):
    write_text_doc(workspace, "sample", ["text"])

    def broken(instruction, content):
        raise RuntimeError("all models failed")

    install_llm(monkeypatch, broken)

    with pytest.raises(RuntimeError):
        cd.classify_domain()

    assert not (workspace / "domain_classification.json").exists()


def test_a_bad_batch_reply_stops_the_stage_and_writes_no_partial_file(workspace, monkeypatch):
    write_text_doc(workspace, "a", ["text a"])
    write_text_doc(workspace, "b", ["text b"])
    install_llm(monkeypatch, lambda i, c: json.dumps([{"title": "Only one", "domain": "x"}]))

    with pytest.raises(ValueError):
        cd.classify_domain()

    assert not (workspace / "domain_classification.json").exists()
