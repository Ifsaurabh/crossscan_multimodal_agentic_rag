"""table_context: which tables go into the answer prompt, and how much of each."""
from retrieval import table_context as tc

HEADER = "| model | recall |" + chr(10) + "|---|---|"


def table(rows, table_id="t1"):
    return {"table_id": table_id, "text": HEADER + chr(10) + chr(10).join(f"| m{i} | 0.{i:02d} |" for i in range(rows))}


def test_a_small_table_is_untouched():
    t = table(5)
    assert tc.cut_table(t["text"]) == (t["text"], 0)


def test_a_long_table_is_cut_after_its_header_and_first_rows(monkeypatch):
    cut, rows_cut = tc.cut_table(table(5000)["text"], max_tokens=200)

    assert cut.startswith(HEADER) and rows_cut > 0
    assert tc.estimate_tokens(cut) <= 200 + 10
    assert all(line.startswith("|") for line in cut.split(chr(10)))  # no row is cut in half


def test_the_header_is_kept_even_with_a_tiny_limit():
    assert tc.cut_table(table(50)["text"], max_tokens=1)[0] == HEADER


def test_merging_puts_the_parents_tables_first_and_removes_duplicates():
    merged = tc.merge_tables([table(1, "a"), table(1, "b")], [table(1, "b"), table(1, "c")])
    assert [t["table_id"] for t in merged] == ["a", "b", "c"]


def test_at_most_max_tables_are_kept():
    capped = tc.cap_tables([table(2, f"t{i}") for i in range(9)])
    assert len(capped) == tc.MAX_TABLES == 4 and all(t["rows_cut"] == 0 for t in capped)


def test_one_table_is_cut_to_its_own_limit_and_says_how_many_rows_were_left_out():
    capped = tc.cap_tables([table(20000)])
    assert tc.estimate_tokens(capped[0]["text"]) <= tc.MAX_TABLE_TOKENS + 10 and capped[0]["rows_cut"] > 0


def test_the_total_budget_is_shared_between_the_tables(monkeypatch):
    monkeypatch.setattr(tc, "MAX_TABLES_TOKENS", 300)
    monkeypatch.setattr(tc, "MAX_TABLE_TOKENS", 200)
    capped = tc.cap_tables([table(2000, f"t{i}") for i in range(4)])

    assert sum(tc.estimate_tokens(t["text"]) for t in capped) <= 300 + 20 * len(capped)
    assert len(capped) < 4 or capped[-1]["rows_cut"] > 0


def test_the_original_tables_are_not_changed():
    original = table(5000)
    tc.cap_tables([original])
    assert original["text"] == table(5000)["text"] and "rows_cut" not in original
