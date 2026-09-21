from types import SimpleNamespace

import extract_text as et


class FakeDocument:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        for item in self._items:
            yield item, 0


def make_item(label, text, page=1, level=None):
    prov = [SimpleNamespace(page_no=page)] if page is not None else None
    return SimpleNamespace(label=label, text=text, prov=prov, level=level)


def test_extract_blocks_keeps_only_text_labels():
    items = [
        make_item("text", "Hello world", page=1),
        make_item("picture", None, page=1),
        make_item("section_header", "Intro", page=1, level=1),
        make_item("footnote", "a footnote", page=2),
        make_item("table", "table caption text", page=2),
    ]
    doc = FakeDocument(items)

    blocks = et.extract_blocks(doc)

    labels = [b["label"] for b in blocks]
    assert "text" in labels
    assert "section_header" in labels
    assert "footnote" in labels
    assert "table" not in labels
    assert "picture" not in labels


def test_extract_blocks_drops_items_with_no_text():
    items = [
        make_item("text", "", page=1),
        make_item("text", None, page=1),
        make_item("text", "Real content", page=1),
    ]
    doc = FakeDocument(items)

    blocks = et.extract_blocks(doc)

    assert len(blocks) == 1
    assert blocks[0]["text"] == "Real content"


def test_extract_blocks_preserves_page_and_level():
    items = [make_item("section_header", "Methods", page=3, level=2)]
    doc = FakeDocument(items)

    blocks = et.extract_blocks(doc)

    assert blocks[0]["page"] == 3
    assert blocks[0]["level"] == 2
    assert blocks[0]["text"] == "Methods"


def test_extract_blocks_level_is_none_for_non_headers():
    items = [make_item("text", "Body text", page=1, level=None)]
    doc = FakeDocument(items)

    blocks = et.extract_blocks(doc)

    assert blocks[0]["level"] is None
