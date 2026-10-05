import pytest

from ingestion import prepare_documents as pd


def header(text, page, level=1):
    return {"label": "section_header", "level": level, "page": page, "text": text}


def body(text, page):
    return {"label": "text", "level": None, "page": page, "text": text}


def table(text, page):
    return {"label": "table", "level": None, "page": page, "text": text}


# ---------- is_references_heading ----------

def test_is_references_heading_matches_common_variants():
    assert pd.is_references_heading("References")
    assert pd.is_references_heading("REFERENCES")
    assert pd.is_references_heading("6. REFERENCES")
    assert pd.is_references_heading("2.1 References")
    assert pd.is_references_heading("Bibliography")
    assert pd.is_references_heading("Reference List:")


def test_is_references_heading_rejects_non_matches():
    assert not pd.is_references_heading("Introduction")
    assert not pd.is_references_heading("1. Methodology")
    assert not pd.is_references_heading(None)
    assert not pd.is_references_heading("")


def test_a_heading_that_merely_mentions_references_is_kept():
    assert not pd.is_references_heading("References to prior work")
    assert not pd.is_references_heading("Related work and references")


# ---------- split_into_sections: sections ----------

def test_split_into_sections_groups_blocks_and_tracks_pages():
    blocks = [
        {"label": "text", "level": None, "page": 1, "text": "preamble text"},
        {"label": "section_header", "level": 1, "page": 1, "text": "Abstract"},
        {"label": "text", "level": None, "page": 1, "text": "abstract body"},
        {"label": "section_header", "level": 1, "page": 2, "text": "Introduction"},
        {"label": "text", "level": None, "page": 2, "text": "intro part one"},
        {"label": "text", "level": None, "page": 3, "text": "intro part two"},
    ]

    sections, tables = pd.split_into_sections(blocks)

    assert len(sections) == 3
    assert tables == []

    preamble = sections[0]
    assert preamble["heading"] is None
    assert preamble["page_start"] == 1
    assert preamble["page_end"] == 1

    abstract = sections[1]
    assert abstract["heading"] == "Abstract"
    assert "abstract body" in abstract["text"]
    assert abstract["page_start"] == 1
    assert abstract["page_end"] == 1

    intro = sections[2]
    assert intro["heading"] == "Introduction"
    assert "intro part one" in intro["text"]
    assert "intro part two" in intro["text"]
    assert intro["page_start"] == 2
    assert intro["page_end"] == 3


def test_block_text_is_joined_with_blank_lines_and_trimmed():
    sections, _ = pd.split_into_sections([header("Intro", 1), body("first", 1), body("second", 1)])

    assert sections[0]["text"] == "first\n\nsecond"


def test_no_blocks_gives_no_sections_and_no_tables():
    assert pd.split_into_sections([]) == ([], [])


def test_blocks_before_the_first_heading_form_a_headingless_preamble():
    sections, _ = pd.split_into_sections([body("loose text", 1)])

    assert sections[0]["heading"] is None and sections[0]["level"] == 0


def test_a_section_with_no_page_information_has_none_pages():
    sections, _ = pd.split_into_sections([header("Intro", None), body("text", None)])

    assert sections[0]["page_start"] is None and sections[0]["page_end"] is None


def test_the_page_range_is_the_smallest_and_largest_page_seen():
    sections, _ = pd.split_into_sections([header("Results", 5), body("a", 7), body("b", 6)])

    assert (sections[0]["page_start"], sections[0]["page_end"]) == (5, 7)


# ---------- split_into_sections: tables ----------

def test_a_table_block_is_pulled_out_of_the_prose_into_its_own_list():
    blocks = [header("Results", 3), body("Before the table.", 3), table("Model  F1\nVGG16  0.98", 3), body("After the table.", 3)]

    sections, tables = pd.split_into_sections(blocks)

    assert tables == [{"text": "Model  F1\nVGG16  0.98", "caption": "", "page": 3, "section_heading": "Results"}]
    assert "VGG16" not in sections[0]["text"]                       # not glued into the prose
    assert sections[0]["text"] == "Before the table.\n\nAfter the table."


def test_a_table_remembers_the_heading_of_the_section_it_sits_in():
    blocks = [header("Methods", 1), table("t1", 1), header("Results", 2), table("t2", 2)]

    _, tables = pd.split_into_sections(blocks)

    assert [(t["text"], t["section_heading"]) for t in tables] == [("t1", "Methods"), ("t2", "Results")]


def test_a_table_before_any_heading_has_no_section_heading():
    _, tables = pd.split_into_sections([table("early table", 1)])

    assert tables == [{"text": "early table", "caption": "", "page": 1, "section_heading": None}]


def test_a_tables_page_still_counts_towards_its_sections_page_range():
    sections, _ = pd.split_into_sections([header("Results", 5), body("text", 5), table("t", 6)])

    assert (sections[0]["page_start"], sections[0]["page_end"]) == (5, 6)


def test_a_table_can_have_no_page():
    _, tables = pd.split_into_sections([header("Results", 1), table("t", None)])

    assert tables[0]["page"] is None


def test_several_tables_keep_their_reading_order():
    _, tables = pd.split_into_sections([header("A", 1), table("first", 1), table("second", 1), table("third", 2)])

    assert [t["text"] for t in tables] == ["first", "second", "third"]


# ---------- prepare_document (blocks in, sections and tables out) ----------

def test_references_are_dropped_and_the_rest_is_kept():
    result = pd.prepare_document([
        header("Introduction", 1), body("Some intro content.", 1),
        header("References", 5), body("[1] Some citation.", 5),
    ])

    headings = [s["heading"] for s in result["sections"]]
    assert "References" not in headings and "Introduction" in headings
    assert result["references_dropped"] == 1


def test_the_result_has_the_sections_the_tables_and_what_was_dropped():
    result = pd.prepare_document([header("Intro", 1), body("text", 1), table("t", 1)])
    assert set(result) == {"sections", "tables", "references_dropped", "reference_tables_dropped"} and len(result["tables"]) == 1


def test_tables_in_the_references_section_are_dropped_but_others_are_kept():
    result = pd.prepare_document([
        header("Results", 3), table("real results table", 3),
        header("6. References", 9), table("citation list rendered as a table", 9),
    ])

    assert [t["text"] for t in result["tables"]] == ["real results table"]
    assert result["reference_tables_dropped"] == 1


def test_a_section_with_only_a_table_is_dropped_as_prose_but_its_table_survives():
    result = pd.prepare_document([header("Results", 3), table("only a table here", 3)])

    assert result["sections"] == []                       # nothing left to chunk as text
    assert [t["text"] for t in result["tables"]] == ["only a table here"]


def test_empty_sections_are_skipped():
    result = pd.prepare_document([header("Empty heading", 1), header("Real", 2), body("content", 2)])
    assert [s["heading"] for s in result["sections"]] == ["Real"]


def test_a_preamble_before_the_first_heading_is_kept():
    result = pd.prepare_document([body("title and authors", 1), header("Abstract", 1), body("abstract text", 1)])
    assert [s["heading"] for s in result["sections"]] == [None, "Abstract"]


def test_a_document_with_no_blocks_gives_nothing_and_does_not_fail():
    assert pd.prepare_document([]) == {"sections": [], "tables": [], "references_dropped": 0, "reference_tables_dropped": 0}
