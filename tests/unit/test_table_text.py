"""table_text: what is embedded for a table (its caption, then its text cut to the limit)."""
from ingestion import table_text as tt

HEADER = "| model | recall |\n|---|---|"


def table(rows):
    return HEADER + "\n" + "\n".join(f"| m{i} | 0.{i:02d} |" for i in range(rows))


def test_a_small_table_is_left_alone():
    text = table(5)
    assert tt.truncate_to_tokens(text) == (text, 0)


def test_a_huge_table_is_cut_at_a_row_boundary_and_keeps_its_header():
    text = table(2000)

    cut, rows_cut = tt.truncate_to_tokens(text, max_tokens=100)

    assert cut.startswith(HEADER) and rows_cut > 0
    assert cut.count("\n") + 1 + rows_cut == text.count("\n") + 1  # every line is either kept or counted as cut
    assert len(tt.ENCODING.encode(cut)) <= 100 + 8   # within the limit (one line of slack)
    assert all(line.startswith("|") for line in cut.split("\n"))  # no row is cut in half


def test_the_header_is_kept_even_when_the_limit_is_tiny():
    cut, _ = tt.truncate_to_tokens(table(50), max_tokens=1)
    assert cut == HEADER


def test_the_caption_comes_first_and_is_embedded_with_the_table():
    result = tt.table_embedding_text({"caption": "Table 3: recall of each model", "text": table(3)})
    assert result.startswith("Table 3: recall of each model\n| model | recall |")


def test_a_table_without_a_caption_is_embedded_as_it_is():
    assert tt.table_embedding_text({"caption": "", "text": table(3)}) == table(3)
    assert tt.table_embedding_text({"text": table(3)}) == table(3)


def test_the_embedded_text_is_cut_to_the_limit_for_a_large_table(monkeypatch):
    monkeypatch.setattr(tt, "TABLE_MAX_TOKENS", 50)
    result = tt.table_embedding_text({"caption": "Table 1", "text": table(500)})
    assert len(tt.ENCODING.encode(result)) < 80
