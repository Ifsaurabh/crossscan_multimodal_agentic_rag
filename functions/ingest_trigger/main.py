"""ingest_trigger: starts the ingestion worker when a document is uploaded.

The bucket sends a message to the Pub/Sub topic `doc-events` for every new object under `incoming/`. This function is subscribed
to that topic (by `gcloud functions deploy --trigger-topic`, which gives it a subscription of its own, next to the worker's
`doc-events-sub`), and for every real upload it starts one execution of the Cloud Run job.

It does not wait for the job and does not decide anything else: many uploads at once start many executions, and the
worker's lease lets exactly one of them work while the others exit within seconds (ingestion/worker_lease.py).

Settings (environment variables of the function): JOB (default crossscan-ingest-worker), REGION (default europe-west1) and
GOOGLE_CLOUD_PROJECT (set by Google).
"""
import os

import functions_framework

DEFAULT_JOB = "crossscan-ingest-worker"
DEFAULT_REGION = "europe-west1"
INCOMING = "incoming/"


def should_start(attributes: dict) -> bool:
    """True for the upload of a file under incoming/ (the same rule as ingestion/ingest_worker.parse_notification):
    only a finished upload counts, not a deletion or another event, and not the placeholder object that the console
    creates for an empty folder."""
    attributes = attributes or {}
    object_id = attributes.get("objectId") or ""
    return (attributes.get("eventType") == "OBJECT_FINALIZE" and object_id.startswith(INCOMING)
            and not object_id.endswith("/"))


def start_job(project: str, region: str, job: str, client=None) -> str:
    """Starts one execution of the job and returns its name, without waiting for it to finish."""
    if client is None:
        from google.cloud import run_v2

        client = run_v2.JobsClient()
    operation = client.run_job(request={"name": f"projects/{project}/locations/{region}/jobs/{job}"})
    return getattr(getattr(operation, "metadata", None), "name", "") or "started"


@functions_framework.cloud_event
def trigger(cloud_event):
    """The entry point: one Pub/Sub message in, at most one job execution out."""
    message = (cloud_event.data or {}).get("message", {})
    attributes = message.get("attributes", {})
    if not should_start(attributes):
        print(f"ignored: {attributes.get('eventType')} {attributes.get('objectId')}")
        return
    execution = start_job(os.environ["GOOGLE_CLOUD_PROJECT"], os.environ.get("REGION", DEFAULT_REGION),
                          os.environ.get("JOB", DEFAULT_JOB))
    print(f"started {execution} for {attributes.get('objectId')}")
