import load_data as ld


def test_load_dataset_copies_pdfs_from_cache(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    (cache_dir / "paper1.pdf").write_bytes(b"%PDF-1.4 fake content")
    (cache_dir / "paper2.pdf").write_bytes(b"%PDF-1.4 fake content")
    (cache_dir / "readme.txt").write_text("not a pdf")

    monkeypatch.setattr(ld, "RAW_DIR", raw_dir)
    monkeypatch.setattr(ld.kagglehub, "dataset_download", lambda dataset: str(cache_dir))

    ld.load_dataset()

    copied = sorted(p.name for p in raw_dir.glob("*.pdf"))
    assert copied == ["paper1.pdf", "paper2.pdf"]
    assert not (raw_dir / "readme.txt").exists()


def test_load_dataset_uses_configured_dataset_id(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    seen = {}

    def fake_download(dataset):
        seen["dataset"] = dataset
        return str(cache_dir)

    monkeypatch.setattr(ld, "RAW_DIR", raw_dir)
    monkeypatch.setattr(ld.kagglehub, "dataset_download", fake_download)

    ld.load_dataset()

    assert seen["dataset"] == ld.DATASET
