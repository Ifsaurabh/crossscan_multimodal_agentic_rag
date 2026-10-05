"""table_text: the text that is embedded for a table.

A table is embedded as its caption followed by its text, so a question that names the table ("Table 3") or its subject
can find it. The text is cut to TABLE_MAX_TOKENS first (after the header and the first rows, with the same kind of
note the answer prompt gets), so a very large table does not make a poor vector. The table is STORED in full.
"""
from ingestion.chunk_documents import ENCODING

TABLE_MAX_TOKENS = 1500


def truncate_to_tokens(text: str, max_tokens: int = TABLE_MAX_TOKENS):
    """(text, rows_cut): the text cut at a row boundary so that it fits in max_tokens, and how many rows were cut.
    Rows are lines; the header (the first two lines of a markdown table) is always kept."""
    if len(ENCODING.encode(text)) <= max_tokens:
        return text, 0
    lines = text.split("\n")
    kept, used = [], 0
    for index, line in enumerate(lines):
        cost = len(ENCODING.encode(line)) + 1
        if index >= 2 and used + cost > max_tokens:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept), len(lines) - len(kept)


def table_embedding_text(table: dict) -> str:
    """The caption, then the table text cut to the limit."""
    body, _ = truncate_to_tokens(table.get("text") or "", TABLE_MAX_TOKENS)
    caption = (table.get("caption") or "").strip()
    return f"{caption}\n{body}" if caption else body
