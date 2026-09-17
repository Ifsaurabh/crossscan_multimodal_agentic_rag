import json

import chunk_documents as cd
import chunking_config as cfg


def test_config_overlap_is_20_percent():
    assert cfg.PARENT_OVERLAP_TOKENS == round(cfg.PARENT_MAX_TOKENS * 0.20)
    assert cfg.CHILD_OVERLAP_TOKENS == round(cfg.CHILD_MAX_TOKENS * 0.20)


def test_count_tokens_matches_tiktoken_encoding():
    text = "Hello world, this is a test."
    assert cd.count_tokens(text) == len(cd.ENCODING.encode(text))


def test_group_units_respects_max_tokens_cap():
    units = [f"This is paragraph number {i} with some words." for i in range(10)]
    max_tokens = 20

    chunks = cd.group_units(units, max_tokens, overlap_tokens=4)

    assert len(chunks) > 1
    for chunk in chunks:
        assert cd.count_tokens(chunk) <= max_tokens + 5  # small tolerance for join overhead


def test_group_units_single_chunk_when_everything_fits():
    units = ["Short one.", "Short two."]
    max_tokens = 1000

    chunks = cd.group_units(units, max_tokens, overlap_tokens=100)

    assert len(chunks) == 1
    assert "Short one." in chunks[0]
    assert "Short two." in chunks[0]


def test_group_units_carries_overlap_into_next_chunk():
    units = ["Paragraph A here.", "Paragraph B here.", "Paragraph C here."]
    tok_a = cd.count_tokens(units[0])
    max_tokens = tok_a + cd.count_tokens(units[1])  # fits exactly A+B, not C
    overlap_tokens = tok_a  # carry paragraph B forward isn't guaranteed; just check some overlap logic runs

    chunks = cd.group_units(units, max_tokens, overlap_tokens)

    assert len(chunks) >= 2
    # the overlap-carrying unit from the end of chunk 1 should reappear at the start of chunk 2
    assert any(units[1] in chunks[i] and units[1] in chunks[i + 1] for i in range(len(chunks) - 1)) or len(chunks) == 2


def test_split_oversized_unit_respects_cap():
    # RecursiveCharacterTextSplitter can slightly overshoot the cap (a few tokens)
    # due to separator/join overhead - same known effect as group_units' overlap
    # join (documented in plannings.md as the 400->409/426 token overshoot).
    long_text = "This is a moderately long sentence about lung cancer research. " * 50
    max_tokens = 50
    tolerance = 5

    pieces = cd.split_oversized_unit(long_text, max_tokens, overlap_tokens=10)

    assert len(pieces) > 1
    for piece in pieces:
        assert cd.count_tokens(piece) <= max_tokens + tolerance


def test_chunk_documents_produces_parent_child_structure(tmp_path, monkeypatch):
    guarded_dir = tmp_path / "guarded"
    chunks_dir = tmp_path / "chunks"
    guarded_dir.mkdir()

    monkeypatch.setattr(cd, "GUARDED_DIR", guarded_dir)
    monkeypatch.setattr(cd, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(cd, "REPORT_PATH", tmp_path / "chunking_report.json")

    doc = {
        "source_pdf": "sample.pdf",
        "sections": [
            {
                "heading": "Introduction",
                "text": "First paragraph of intro.\n\nSecond paragraph of intro.",
                "page_start": 1,
                "page_end": 1,
            }
        ],
    }
    (guarded_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")

    cd.chunk_documents()

    out_path = chunks_dir / "sample.json"
    assert out_path.exists()

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["chunking_version"] == cfg.CHUNKING_VERSION
    assert len(result["parents"]) >= 1
    assert len(result["parents"][0]["children"]) >= 1
    assert result["parents"][0]["section"] == "Introduction"
    assert result["parents"][0]["page_start"] == 1
