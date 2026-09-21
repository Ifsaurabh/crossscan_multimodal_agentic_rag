import hashlib

import experiment_tracking as et


def test_golden_set_version_uses_dvc_md5_when_present(tmp_path):
    golden = tmp_path / "golden_set.jsonl"
    golden.write_text("data", encoding="utf-8")
    (tmp_path / "golden_set.jsonl.dvc").write_text(
        "outs:\n- md5: 323fe8a767aeb7bc6138c0b5a6d33cc3\n  size: 4\n  path: golden_set.jsonl\n",
        encoding="utf-8",
    )

    assert et.golden_set_version(golden) == "323fe8a767aeb7bc6138c0b5a6d33cc3"


def test_golden_set_version_hashes_file_when_not_dvc_tracked(tmp_path):
    golden = tmp_path / "golden_set.jsonl"
    golden.write_bytes(b"hello")

    assert et.golden_set_version(golden) == hashlib.md5(b"hello").hexdigest()


def test_retrieval_config_snapshot_has_uppercase_constants_only():
    snapshot = et.retrieval_config_snapshot()

    assert snapshot["MAX_RETRY_ATTEMPTS"] == 3
    assert "GEMINI_MODEL" in snapshot
    assert all(k.isupper() for k in snapshot)


def test_collect_versions_flattens_golden_prompts_and_config(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_bytes(b"x")

    versions = et.collect_versions(golden)

    assert versions["golden_set_md5"] == hashlib.md5(b"x").hexdigest()
    assert "prompt_transform-route" in versions
    assert versions["config_max_retry_attempts"] == 3


def test_log_run_writes_params_metrics_and_artifact(tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient

    monkeypatch.setattr(et, "MLRUNS_DIR", tmp_path / "mlruns")
    (tmp_path / "mlruns").mkdir()

    run_id = et.log_run(
        "test-run",
        params={"golden_set_md5": "abc", "n_examples": 2},
        metrics={"source_hit": 0.5, "skipped": None},
        results_rows=[{"input": "q", "answer": "a"}],
        tags={"kind": "unit-test"},
    )

    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())
    run = client.get_run(run_id)
    assert run.data.params["golden_set_md5"] == "abc"
    assert run.data.metrics["source_hit"] == 0.5
    assert "skipped" not in run.data.metrics
    assert run.data.tags["kind"] == "unit-test"
    assert [a.path for a in client.list_artifacts(run_id, "results")]
