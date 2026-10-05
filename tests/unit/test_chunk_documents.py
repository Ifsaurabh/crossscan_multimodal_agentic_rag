import pytest

from ingestion import chunk_documents as cd
from ingestion import chunking_config as cfg

TOLERANCE = 5  # the splitter can overshoot a cap by a few tokens (separator / join overhead)


def test_config_overlap_is_20_percent():
    assert cfg.PARENT_OVERLAP_TOKENS == round(cfg.PARENT_MAX_TOKENS * 0.20)
    assert cfg.CHILD_OVERLAP_TOKENS == round(cfg.CHILD_MAX_TOKENS * 0.20)


def test_count_tokens_matches_tiktoken_encoding():
    text = "Hello world, this is a test."
    assert cd.count_tokens(text) == len(cd.ENCODING.encode(text))


# ---------- split_text (LangChain's token-aware recursive splitter) ----------

def test_split_text_respects_the_token_cap():
    long_text = "This is a moderately long sentence about lung cancer research. " * 50
    max_tokens = 50

    pieces = cd.split_text(long_text, max_tokens, overlap_tokens=10)

    assert len(pieces) > 1
    for piece in pieces:
        assert cd.count_tokens(piece) <= max_tokens + TOLERANCE


def test_split_text_returns_one_chunk_when_everything_fits():
    pieces = cd.split_text("Short one. Short two.", max_tokens=1000, overlap_tokens=100)

    assert pieces == ["Short one. Short two."]


def test_split_text_carries_an_overlap_into_the_next_chunk():
    text = " ".join(f"Sentence number {i} is right here." for i in range(60))

    pieces = cd.split_text(text, max_tokens=40, overlap_tokens=10)

    assert len(pieces) > 2
    for previous, following in zip(pieces, pieces[1:]):
        opening_words = " ".join(following.split()[:3])
        assert opening_words in previous  # the start of a chunk repeats the end of the one before


def test_split_text_without_overlap_repeats_nothing():
    words = [f"word{i}" for i in range(300)]

    pieces = cd.split_text(" ".join(words), max_tokens=40, overlap_tokens=0)

    assert " ".join(pieces).split() == words  # every word once, in order


def test_split_text_prefers_paragraph_boundaries_over_cutting_mid_paragraph():
    first, second = " ".join(["alpha"] * 20), " ".join(["beta"] * 20)

    pieces = cd.split_text(f"{first}\n\n{second}", max_tokens=30, overlap_tokens=0)

    assert pieces == [first, second]  # each paragraph kept whole


def test_split_text_of_nothing_is_nothing():
    assert cd.split_text("", 100, 10) == []
    assert cd.split_text("   \n\n  ", 100, 10) == []


def test_split_text_keeps_unicode_intact():
    text = "Le réseau détecte le cancer du poumon. 肺癌の検出。" * 5

    pieces = cd.split_text(text, max_tokens=1000, overlap_tokens=0)

    assert pieces == [text.strip()]


# ---------- chunk_document (sections in, parents + children out) ----------

def section(heading="Introduction", text="First paragraph of intro.\n\nSecond paragraph of intro.", start=1, end=1):
    return {"heading": heading, "text": text, "page_start": start, "page_end": end}


def chunk(sections, stem="sample", source_pdf=None):
    return cd.chunk_document(source_pdf or f"{stem}.pdf", sections, stem=stem)


def test_chunk_document_produces_parent_child_structure():
    result = chunk([section()])

    assert result["chunking_version"] == cfg.CHUNKING_VERSION and result["source_pdf"] == "sample.pdf"
    assert len(result["parents"]) >= 1
    assert len(result["parents"][0]["children"]) >= 1
    assert result["parents"][0]["section"] == "Introduction"
    assert result["parents"][0]["page_start"] == 1


def test_parents_and_children_get_predictable_ids_and_link_to_each_other():
    parents = chunk([section("Intro"), section("Methods", text="Methods text here.", start=2, end=3)])["parents"]

    assert [p["chunk_id"] for p in parents] == ["sample_p1", "sample_p2"]
    assert all(p["chunk_type"] == "parent" for p in parents)
    for parent in parents:
        for child in parent["children"]:
            assert child["parent_id"] == parent["chunk_id"] and child["chunk_type"] == "child"


def test_child_ids_keep_counting_across_the_whole_document():
    parents = chunk([section("A", text="First section."), section("B", text="Second section.")])["parents"]

    children = [c for p in parents for c in p["children"]]
    assert [c["chunk_id"] for c in children] == ["sample_c1", "sample_c2"]  # not restarted per parent


def test_the_stem_names_the_chunks_and_defaults_to_the_file_name():
    assert chunk([section()], stem="lung-cancer/paper")["parents"][0]["chunk_id"] == "lung-cancer/paper_p1"
    assert cd.chunk_document("Paper One.pdf", [section()])["parents"][0]["chunk_id"] == "Paper One_p1"


def test_children_inherit_the_papers_section_and_pages():
    child = chunk([section("Results", text="Accuracy was high.", start=12, end=14)], source_pdf="Paper One.pdf")["parents"][0]["children"][0]

    assert (child["source_pdf"], child["section"], child["page_start"], child["page_end"]) == ("Paper One.pdf", "Results", 12, 14)
    assert child["text"] == "Accuracy was high."


def test_a_section_without_page_numbers_passes_none_through():
    parent = chunk([{"heading": "Abstract", "text": "Some text."}])["parents"][0]
    assert parent["page_start"] is None and parent["page_end"] is None


def test_empty_and_whitespace_only_sections_are_skipped_and_use_no_ids():
    parents = chunk([section("Blank", text=""), section("Spaces", text="  \n\n "), section("Real", text="Real text.")])["parents"]
    assert [p["section"] for p in parents] == ["Real"] and parents[0]["chunk_id"] == "sample_p1"


def test_a_long_section_is_split_into_several_capped_parents_and_children():
    long_text = " ".join(f"Sentence {i} discusses lung nodule detection results." for i in range(600))

    parents = chunk([section("Results", text=long_text)])["parents"]

    assert len(parents) > 1
    for parent in parents:
        assert cd.count_tokens(parent["text"]) <= cfg.PARENT_MAX_TOKENS + TOLERANCE
        assert len(parent["children"]) > 1
        for child in parent["children"]:
            assert cd.count_tokens(child["text"]) <= cfg.CHILD_MAX_TOKENS + TOLERANCE


def test_a_document_with_no_sections_gives_no_parents():
    assert chunk([])["parents"] == []
