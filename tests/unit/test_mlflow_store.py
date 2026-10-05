"""Where an evaluation run is stored: the local folder by default, the project's Postgres with MLFLOW_STORE=database."""
import pytest

from evaluation import experiment_tracking as et

URL = "postgresql://user:p%40ss@ep-example.eu-west-1.aws.neon.tech/db?sslmode=require"


@pytest.mark.parametrize("given,expected", [
    (URL, "postgresql+psycopg2://user:p%40ss@ep-example.eu-west-1.aws.neon.tech/db?sslmode=require"),
    ("postgres://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
    ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
    ("postgresql+psycopg2://u:p@h/db", "postgresql+psycopg2://u:p@h/db"),
])
def test_the_database_url_becomes_a_sqlalchemy_url_for_mlflow(given, expected):
    assert et.database_tracking_uri(given) == expected


@pytest.mark.parametrize("bad", ["mysql://u:p@h/db", "sqlite:///x.db", "", "file:///tmp/x"])
def test_anything_that_is_not_a_postgres_url_is_refused(bad):
    with pytest.raises(ValueError):
        et.database_tracking_uri(bad)


@pytest.fixture
def clean(monkeypatch):
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_STORE", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_by_default_runs_stay_in_the_local_folder(clean):
    assert et.tracking_uri() == et.MLRUNS_DIR.as_uri()


def test_the_database_store_uses_the_projects_postgres(clean, monkeypatch):
    monkeypatch.setenv("MLFLOW_STORE", "database")
    monkeypatch.setenv("DATABASE_URL", URL)
    assert et.tracking_uri().startswith("postgresql+psycopg2://user:p%40ss@ep-example")


def test_an_explicit_tracking_uri_wins_over_everything(clean, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example:5000")
    monkeypatch.setenv("MLFLOW_STORE", "database")
    monkeypatch.setenv("DATABASE_URL", URL)
    assert et.tracking_uri() == "http://mlflow.example:5000"


def test_the_database_store_without_a_database_url_says_so(clean, monkeypatch):
    monkeypatch.setenv("MLFLOW_STORE", "database")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        et.tracking_uri()


def test_any_other_store_value_means_the_local_folder(clean, monkeypatch):
    monkeypatch.setenv("MLFLOW_STORE", "")
    monkeypatch.setenv("DATABASE_URL", URL)
    assert et.tracking_uri() == et.MLRUNS_DIR.as_uri()
