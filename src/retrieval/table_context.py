"""table_context: which tables go into the answer prompt, and how much of each.

The tables of the final chunks' parents come first, the tables found by the table search fill the rest, duplicates are
removed by table id. Then the size limits: a table longer than MAX_TABLE_TOKENS is cut after its header and first rows
(its `rows_cut` says how many rows were left out, and the prompt shows a note); at most MAX_TABLES tables and
MAX_TABLES_TOKENS tokens in all. Sizes are estimated at CHARS_PER_TOKEN characters per token.
"""
from retrieval.retrieval_config import CHARS_PER_TOKEN, MAX_TABLE_TOKENS, MAX_TABLES, MAX_TABLES_TOKENS

HEADER_LINES = 2  # a markdown table starts with its header row and the separator line


def estimate_tokens(text: str) -> int:
    return -(-len(text or "") // CHARS_PER_TOKEN)


def cut_table(text: str, max_tokens: int = MAX_TABLE_TOKENS):
    """(text, rows_cut): the table cut at a row boundary to fit max_tokens, the header always kept."""
    if estimate_tokens(text) <= max_tokens:
        return text, 0
    lines = text.split("\n")
    kept, used = [], 0
    for index, line in enumerate(lines):
        cost = estimate_tokens(line) + 1
        if index >= HEADER_LINES and used + cost > max_tokens:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept), len(lines) - len(kept)


def merge_tables(from_parents: list, from_search: list) -> list:
    """The parents' tables first, then the searched ones, each table once (by table id)."""
    seen, merged = set(), []
    for table in list(from_parents or []) + list(from_search or []):
        table_id = table.get("table_id")
        if table_id in seen:
            continue
        seen.add(table_id)
        merged.append(table)
    return merged


def cap_tables(tables: list) -> list:
    """The tables the prompt will carry, each cut to its limit (`rows_cut` set when rows were left out)."""
    capped, budget = [], MAX_TABLES_TOKENS
    for table in tables:
        if len(capped) >= MAX_TABLES or budget <= 0:
            break
        text, rows_cut = cut_table(table.get("text") or "", min(MAX_TABLE_TOKENS, budget))
        capped.append({**table, "text": text, "rows_cut": rows_cut})
        budget -= estimate_tokens(text)
    return capped
