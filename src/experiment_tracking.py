import hashlib
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "crossscan-rag-eval"


def golden_set_version(path) -> str:
    """The DVC content hash of the golden set when it's DVC-tracked (so the
    value matches `dvc` and git history), else a hash of the file itself."""
    path = Path(path)
    dvc_file = Path(str(path) + ".dvc")
    if dvc_file.exists():
        match = re.search(r"md5:\s*([0-9a-f]{32})", dvc_file.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return hashlib.md5(path.read_bytes()).hexdigest()


def retrieval_config_snapshot() -> dict:
    """Every UPPER_CASE constant in retrieval_config.py."""
    import retrieval_config

    return {k: getattr(retrieval_config, k) for k in dir(retrieval_config) if k.isupper()}


def collect_versions(golden_path) -> dict:
    """Everything an eval run depends on, flattened for MLflow params:
    golden-set hash, per-prompt content hashes, retrieval config values."""
    import prompt_registry

    versions = {"golden_set_md5": golden_set_version(golden_path)}
    for name, digest in prompt_registry.local_prompt_versions().items():
        versions[f"prompt_{name}"] = digest
    for key, value in retrieval_config_snapshot().items():
        versions[f"config_{key.lower()}"] = value

    import llm_connection

    versions.update(llm_connection.config_snapshot())
    return versions


def log_run(run_name: str, params: dict, metrics: dict, results_rows: list, tags: dict = None) -> str:
    """Logs one evaluation run to the local MLflow file store and returns the
    run id."""
    import mlflow

    # MLflow 3.x treats the local file store as maintenance-mode and refuses it
    # unless explicitly allowed. We keep it on purpose (no server, one folder);
    # switch to a sqlite:/// URI if it is ever removed.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        if tags:
            mlflow.set_tags(tags)

        results_path = MLRUNS_DIR / f"_results_{run.info.run_id}.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for row in results_rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        mlflow.log_artifact(str(results_path), artifact_path="results")
        os.remove(results_path)

        return run.info.run_id
