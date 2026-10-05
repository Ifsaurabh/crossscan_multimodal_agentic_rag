import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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


def git_commit() -> str:
    """The commit the run was made from (GITHUB_SHA in CI, else git), or "unknown"."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=PROJECT_ROOT)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def corpus_version():
    """(a short hash, number of documents) of what is ingested: the active (document, content hash) rows of the manifest, so
    a run can say which corpus it was measured on. ("unknown", 0) when the database cannot be read."""
    try:
        from shared.db import SCHEMA_NAME, connection

        with connection() as conn:
            rows = conn.execute(
                f"SELECT source_pdf, content_hash FROM {SCHEMA_NAME}.ingestion_manifest WHERE status = 'active' ORDER BY 1, 2"
            ).fetchall()
        digest = hashlib.md5("\n".join(f"{name}:{content_hash}" for name, content_hash in rows).encode("utf-8")).hexdigest()
        return digest[:12], len(rows)
    except Exception:
        return "unknown", 0


def retrieval_config_snapshot() -> dict:
    """Every UPPER_CASE constant in retrieval_config.py."""
    from retrieval import retrieval_config

    return {k: getattr(retrieval_config, k) for k in dir(retrieval_config) if k.isupper()}


def collect_versions(golden_path) -> dict:
    """Everything an eval run depends on, flattened for MLflow params:
    golden-set hash, per-prompt content hashes, retrieval config values."""
    from retrieval import prompt_sync

    versions = {"golden_set_md5": golden_set_version(golden_path), "git_commit": git_commit()}
    versions["corpus_version"], versions["corpus_documents"] = corpus_version()
    for name, digest in prompt_sync.local_prompt_versions().items():
        versions[f"prompt_{name}"] = digest
    for key, value in retrieval_config_snapshot().items():
        versions[f"config_{key.lower()}"] = value

    from shared import llm_connection

    versions.update(llm_connection.config_snapshot())
    return versions


def database_tracking_uri(database_url: str) -> str:
    """The MLflow tracking URI for the project's Postgres, made from DATABASE_URL: the same database, the SQLAlchemy driver name
    added (`postgresql://...` -> `postgresql+psycopg2://...`). MLflow creates its own tables (experiments, runs, metrics...)
    in that database's default schema, `public`, next to the app's `rag_*` schemas and apart from them."""
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg2://" + database_url[len(prefix):]
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg2://" + database_url[len(prefix):]
    raise ValueError("DATABASE_URL is not a postgresql:// URL")


def tracking_uri() -> str:
    """Where runs are stored: MLFLOW_TRACKING_URI when set; else, with MLFLOW_STORE=database, the project's Postgres
    (from DATABASE_URL); else the local ./mlruns folder."""
    explicit = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit:
        return explicit
    if os.environ.get("MLFLOW_STORE") == "database":
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("MLFLOW_STORE=database needs DATABASE_URL")
        return database_tracking_uri(database_url)
    return MLRUNS_DIR.as_uri()


def log_run(run_name: str, params: dict, metrics: dict, results_rows: list, tags: dict = None) -> str:
    """Logs one evaluation run and returns the run id.

    Where it goes (see tracking_uri): MLFLOW_STORE=database for the project's Postgres, or MLFLOW_TRACKING_URI for any store,
    and MLFLOW_ARTIFACT_ROOT for the files (a bucket folder, for example gs://crossscan-docs-upload-here/mlflow), so a run from
    your laptop and the weekly CI run land in the same store. Without them, the local file store in
    ./mlruns, as before."""
    import mlflow

    uri = tracking_uri()
    if uri.startswith("file:"):
        # MLflow 3.x treats the local file store as maintenance-mode and refuses it
        # unless explicitly allowed. It stays the default (no server, one folder).
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if artifact_root and mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root)  # fixed when the experiment is created
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        if tags:
            mlflow.set_tags(tags)

        with tempfile.TemporaryDirectory() as folder:
            results_path = Path(folder) / f"_results_{run.info.run_id}.jsonl"
            with open(results_path, "w", encoding="utf-8") as f:
                for row in results_rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            mlflow.log_artifact(str(results_path), artifact_path="results")

        return run.info.run_id
