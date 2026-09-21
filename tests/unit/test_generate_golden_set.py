import json

from google.genai.errors import ClientError

import generate_golden_set as ggs


class FakeGolden:
    def __init__(self, input, expected_output, context, source_file):
        self.input = input
        self.expected_output = expected_output
        self.context = context
        self.source_file = source_file


class FakeSynthesizer:
    def __init__(self, goldens_by_source=None, fail_on=None):
        self.goldens_by_source = goldens_by_source or {}
        self.fail_on = fail_on or set()
        self.calls = []

    def generate_goldens_from_contexts(self, **kwargs):
        source = kwargs["source_files"][0]
        self.calls.append(source)
        if source in self.fail_on:
            raise ClientError(
                429,
                {"error": {"message": "RESOURCE_EXHAUSTED: quota exceeded"}},
                None,
            )
        return self.goldens_by_source.get(source, [])


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql):
        return FakeCursor(self.rows)

    def close(self):
        pass


SAMPLE_ROWS = [
    ("paper_a.pdf", "Introduction", 1, 2, "medical", "Full text of paper A's richest parent chunk."),
    ("paper_b.pdf", "Methodology", 3, 4, "remote_sensing", "Full text of paper B's richest parent chunk."),
]


def test_load_representative_chunks_maps_columns(monkeypatch):
    monkeypatch.setattr(ggs, "get_connection", lambda: FakeConn(SAMPLE_ROWS))

    chunks = ggs.load_representative_chunks()

    assert len(chunks) == 2
    assert chunks[0]["source_pdf"] == "paper_a.pdf"
    assert chunks[0]["domain"] == "medical"
    assert chunks[1]["text"] == "Full text of paper B's richest parent chunk."


def test_load_completed_source_pdfs_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ggs, "GOLDEN_SET_PATH", str(tmp_path / "golden_set.jsonl"))
    assert ggs.load_completed_source_pdfs() == set()


def test_load_completed_source_pdfs_reads_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "golden_set.jsonl"
    path.write_text(
        json.dumps({"source_pdf": "paper_a.pdf", "input": "Q"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ggs, "GOLDEN_SET_PATH", str(path))

    assert ggs.load_completed_source_pdfs() == {"paper_a.pdf"}


def test_generate_golden_set_writes_incrementally_per_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(ggs, "GOLDEN_SET_PATH", str(tmp_path / "golden_set.jsonl"))

    chunks = [
        {"source_pdf": "paper_a.pdf", "domain": "medical", "text": "context a"},
        {"source_pdf": "paper_b.pdf", "domain": "remote_sensing", "text": "context b"},
    ]
    synthesizer = FakeSynthesizer(goldens_by_source={
        "paper_a.pdf": [FakeGolden("What is X?", "X is Y.", ["context a"], "paper_a.pdf")],
        "paper_b.pdf": [FakeGolden("What is Z?", "Z is W.", ["context b"], "paper_b.pdf")],
    })

    result = ggs.generate_golden_set(chunks=chunks, synthesizer=synthesizer)

    assert len(result) == 2
    assert synthesizer.calls == ["paper_a.pdf", "paper_b.pdf"]

    with open(str(tmp_path / "golden_set.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 2
    assert records[0]["input"] == "What is X?"
    assert records[0]["source_pdf"] == "paper_a.pdf"
    assert records[0]["domain"] == "medical"


def test_generate_golden_set_skips_already_completed_papers(tmp_path, monkeypatch):
    path = tmp_path / "golden_set.jsonl"
    path.write_text(
        json.dumps({"source_pdf": "paper_a.pdf", "input": "existing"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ggs, "GOLDEN_SET_PATH", str(path))

    chunks = [
        {"source_pdf": "paper_a.pdf", "domain": "medical", "text": "context a"},
        {"source_pdf": "paper_b.pdf", "domain": "remote_sensing", "text": "context b"},
    ]
    synthesizer = FakeSynthesizer(goldens_by_source={
        "paper_b.pdf": [FakeGolden("What is Z?", "Z is W.", ["context b"], "paper_b.pdf")],
    })

    ggs.generate_golden_set(chunks=chunks, synthesizer=synthesizer)

    # only the not-yet-completed paper was sent to the synthesizer
    assert synthesizer.calls == ["paper_b.pdf"]


def test_generate_golden_set_stops_cleanly_on_quota_exhaustion(tmp_path, monkeypatch):
    monkeypatch.setattr(ggs, "GOLDEN_SET_PATH", str(tmp_path / "golden_set.jsonl"))

    chunks = [
        {"source_pdf": "paper_a.pdf", "domain": "medical", "text": "context a"},
        {"source_pdf": "paper_b.pdf", "domain": "remote_sensing", "text": "context b"},
        {"source_pdf": "paper_c.pdf", "domain": "medical", "text": "context c"},
    ]
    synthesizer = FakeSynthesizer(
        goldens_by_source={
            "paper_a.pdf": [FakeGolden("Q1", "A1", ["context a"], "paper_a.pdf")],
        },
        fail_on={"paper_b.pdf"},
    )

    result = ggs.generate_golden_set(chunks=chunks, synthesizer=synthesizer)

    # stopped after paper_a succeeded and paper_b hit quota - never reached paper_c
    assert len(result) == 1
    assert synthesizer.calls == ["paper_a.pdf", "paper_b.pdf"]

    with open(str(tmp_path / "golden_set.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["source_pdf"] == "paper_a.pdf"


def test_generate_golden_set_reraises_non_quota_client_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(ggs, "GOLDEN_SET_PATH", str(tmp_path / "golden_set.jsonl"))

    chunks = [{"source_pdf": "paper_a.pdf", "domain": "medical", "text": "context a"}]

    class FailingSynthesizer:
        def generate_goldens_from_contexts(self, **kwargs):
            raise ClientError(400, {"error": {"message": "INVALID_ARGUMENT: bad request"}}, None)

    try:
        ggs.generate_golden_set(chunks=chunks, synthesizer=FailingSynthesizer())
        assert False, "expected ClientError to propagate"
    except ClientError:
        pass


def test_generate_golden_set_noop_when_all_papers_done(tmp_path, monkeypatch):
    path = tmp_path / "golden_set.jsonl"
    path.write_text(
        json.dumps({"source_pdf": "paper_a.pdf", "input": "existing"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ggs, "GOLDEN_SET_PATH", str(path))

    chunks = [{"source_pdf": "paper_a.pdf", "domain": "medical", "text": "context a"}]
    synthesizer = FakeSynthesizer()

    result = ggs.generate_golden_set(chunks=chunks, synthesizer=synthesizer)

    assert result == []
    assert synthesizer.calls == []
