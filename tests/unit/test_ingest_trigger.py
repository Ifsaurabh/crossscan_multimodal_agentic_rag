"""ingest_trigger: which messages start the worker, and what it calls. The Google client is a fake; no network."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trigger():
    # functions_framework is not needed to test the logic: a stand-in decorator is enough when it is not installed
    if importlib.util.find_spec("functions_framework") is None:
        stub = types.ModuleType("functions_framework")
        stub.cloud_event = lambda fn: fn
        sys.modules["functions_framework"] = stub
    spec = importlib.util.spec_from_file_location("ingest_trigger_main", ROOT / "functions" / "ingest_trigger" / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upload(object_id="incoming/lung-cancer/paper.pdf", event="OBJECT_FINALIZE"):
    return {"eventType": event, "objectId": object_id, "bucketId": "docs"}


@pytest.mark.parametrize("attributes,expected", [
    (upload(), True),
    (upload("incoming/x.pdf"), True),                        # no domain folder: the worker rejects it, but it must run
    (upload("incoming/lung-cancer/"), False),                # the placeholder of an empty folder
    (upload("processed/lung-cancer/paper.pdf"), False),      # the worker's own moves
    (upload("failed/paper.pdf"), False),
    (upload("images/lung-cancer/paper/a.png"), False),
    (upload(event="OBJECT_DELETE"), False),
    (upload(event="OBJECT_METADATA_UPDATE"), False),
    ({}, False), (None, False), ({"eventType": "OBJECT_FINALIZE"}, False),
])
def test_only_the_upload_of_a_file_under_incoming_starts_the_worker(trigger, attributes, expected):
    assert trigger.should_start(attributes) is expected


def test_the_rule_is_the_same_as_the_workers(trigger):
    from ingestion import ingest_worker

    for attributes in (upload(), upload("incoming/"), upload("processed/a/b.pdf"), upload(event="OBJECT_DELETE"), {}):
        assert trigger.should_start(attributes) == (ingest_worker.parse_notification(attributes) is not None)


def test_the_job_is_started_by_its_full_name_and_not_waited_for(trigger):
    calls = []

    class Client:
        def run_job(self, request):
            calls.append(request)
            return types.SimpleNamespace(metadata=types.SimpleNamespace(name="executions/abc"))

    name = trigger.start_job("proj", "europe-west1", "crossscan-ingest-worker", client=Client())

    assert calls == [{"name": "projects/proj/locations/europe-west1/jobs/crossscan-ingest-worker"}]
    assert name == "executions/abc"


def event(attributes):
    return types.SimpleNamespace(data={"message": {"attributes": attributes}})


def test_an_upload_message_starts_one_execution_and_another_event_starts_none(trigger, monkeypatch, capsys):
    started = []
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("JOB", "my-job")
    monkeypatch.setenv("REGION", "europe-west1")
    monkeypatch.setattr(trigger, "start_job", lambda project, region, job: started.append((project, region, job)) or "exec-1")

    trigger.trigger(event(upload()))
    trigger.trigger(event(upload(event="OBJECT_DELETE")))
    trigger.trigger(types.SimpleNamespace(data=None))

    assert started == [("proj", "europe-west1", "my-job")]
    assert "started exec-1" in capsys.readouterr().out


def test_a_failure_to_start_the_job_is_not_hidden_so_pubsub_delivers_the_message_again(trigger, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")

    def broken(project, region, job):
        raise RuntimeError("permission denied")

    monkeypatch.setattr(trigger, "start_job", broken)
    with pytest.raises(RuntimeError):
        trigger.trigger(event(upload()))


def test_the_function_has_its_own_requirements_and_no_dependency_on_the_app_code():
    requirements = (ROOT / "functions" / "ingest_trigger" / "requirements.txt").read_text()
    assert "functions-framework" in requirements and "google-cloud-run" in requirements
    source = (ROOT / "functions" / "ingest_trigger" / "main.py").read_text()
    assert "from shared" not in source and "from ingestion" not in source and "from retrieval" not in source


# ---------- the project id (a second-generation function is not given GOOGLE_CLOUD_PROJECT) ----------

def test_the_project_comes_from_the_environment_first(trigger):
    assert trigger.project_id({"GOOGLE_CLOUD_PROJECT": "from-env", "GCP_PROJECT": "other"}, default_project="from-credentials") == "from-env"
    assert trigger.project_id({"GCP_PROJECT": "legacy"}, default_project="from-credentials") == "legacy"


def test_without_it_in_the_environment_the_project_of_the_credentials_is_used(trigger):
    assert trigger.project_id({}, default_project="from-credentials") == "from-credentials"


def test_an_unknown_project_is_an_error_not_a_wrong_job_name(trigger, monkeypatch):
    fake = types.SimpleNamespace(default=lambda: (None, None))
    monkeypatch.setitem(sys.modules, "google.auth", fake)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(auth=fake))
    with pytest.raises(RuntimeError, match="project id is unknown"):
        trigger.project_id({})


def test_a_real_upload_starts_the_job_even_when_the_environment_has_no_project(trigger, monkeypatch):
    started = []
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setattr(trigger, "project_id", lambda *a, **k: "sagar-crossscan")
    monkeypatch.setattr(trigger, "start_job", lambda project, region, job, client=None: started.append((project, region, job)) or "exec-1")
    event = types.SimpleNamespace(data={"message": {"attributes": upload()}})

    trigger.trigger(event)

    assert started == [("sagar-crossscan", trigger.DEFAULT_REGION, trigger.DEFAULT_JOB)]
