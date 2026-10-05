"""ingest_worker: takes uploaded documents off the queue one at a time.

The flow for one message:
    1. read the message (a bucket notification: which file was uploaded)
    2. check the path is incoming/<domain>/<file> (the folder is the domain), then download the file
    3. intake check (intake_check.py): what is it, is it new, changed or unchanged, what do its pages hold
    4. rejected   -> the file moves to failed/ with the reason, the message is acknowledged, the next one follows
       unchanged  -> the file moves to archived/ (a duplicate), acknowledged
       accepted   -> the ingestion pipeline runs (injected, see `pipeline`); then the file moves to
                     processed/ and the message is acknowledged. If the pipeline finds the document
                     unusable (DocumentRejected: too empty, a lost table...) it is handled like a rejection
    5. an unexpected error -> the message is NOT acknowledged (it is delivered again after a back-off);
       on the last delivery attempt the file is also moved to failed/, and Pub/Sub then forwards the
       message to the dead-letter topic

Only one message is handled at a time: the worker pulls a single message, and the next is not
delivered until this one is acknowledged. While it works, a background thread keeps extending the
message's acknowledgement deadline, so a long document (a scan costs about 40 s a page) is not
delivered a second time.

Handle the queue with `--run`. Try the checks on real uploads, without moving anything, with:
    PYTHONPATH=src python -m ingestion.ingest_worker --check incoming/some-file.pdf
which downloads and checks the named files and changes nothing (no moves, no queue).
"""
import argparse
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from ingestion import doc_storage
from ingestion import intake_check
from ingestion.errors import DocumentRejected

DEFAULT_PROJECT = "sagar-crossscan"
DEFAULT_SUBSCRIPTION = "doc-events-sub"

ACK_DEADLINE_SECONDS = 60          # the subscription's deadline, renewed this long each time
LEASE_INTERVAL_SECONDS = 30        # how often the deadline is renewed while a document is being handled
MAX_LEASE_SECONDS = 6 * 60 * 60    # give up renewing after this long (the message then comes back)
MAX_DELIVERY_ATTEMPTS = 5          # the subscription's dead-letter limit
PULL_TIMEOUT_SECONDS = 30
EMPTY_POLLS_TO_STOP = 2            # the queue counts as empty after this many empty pulls in a row

# What happened to a message.
INGESTED = "ingested"
REJECTED = "rejected"
DUPLICATE = "duplicate"
MISSING = "missing"      # the file is no longer in incoming/ (an earlier delivery already moved it)
IGNORED = "ignored"      # not an upload of a document (a folder placeholder, another prefix, another event)
FAILED_AGAIN = "failed"  # an error: the message was not acknowledged
CHECKED = "checked"      # --check only: nothing was changed


class PipelineNotAvailable(Exception):
    """The ingestion pipeline is not wired in yet."""


@dataclass
class Notification:
    bucket: str
    object_name: str
    event_type: str
    generation: str = None


@dataclass
class Outcome:
    status: str
    object_name: str = None
    detail: dict = field(default_factory=dict)


def parse_notification(attributes: dict):
    """The upload a message announces, or None if the message is not about a document waiting in
    incoming/. Cloud Storage puts the event type, bucket and object name in the message attributes."""
    attributes = attributes or {}
    event_type, object_name = attributes.get("eventType"), attributes.get("objectId")
    if event_type != "OBJECT_FINALIZE" or not doc_storage.is_document_object(object_name):
        return None
    return Notification(
        bucket=attributes.get("bucketId"), object_name=object_name, event_type=event_type,
        generation=attributes.get("objectGeneration"),
    )


def review_metadata(result: intake_check.IntakeResult, now: datetime = None) -> dict:
    """What is stored on a rejected file, so the review list can say why it is there."""
    return {
        "intake_outcome": result.outcome,
        "intake_reason_code": result.reason_code,
        "intake_reason": result.reason,
        "content_hash": result.content_hash,
        "size_bytes": result.size_bytes,
        "checked_at": (now or datetime.now(timezone.utc)).isoformat(),
    }


def manifest_active_hash(name: str):
    """The content hash of the version of `name` already ingested, or None. Needs the database."""
    from ingestion import ingestion_manifest
    from shared.db import connection

    with connection() as conn:
        return ingestion_manifest.get_active_hash(conn, name)


def process_document(object_name: str, storage, lookup_active_hash, pipeline=None, work_dir=None,
                     dry_run: bool = False, now: datetime = None, record_intake=None) -> Outcome:
    """Downloads one document, checks it, and moves it according to the result.

    `lookup_active_hash(name)` gives the hash of the version already ingested (or None).
    `pipeline(local_path, intake_result)` ingests an accepted document; it is not wired in yet.
    `record_intake(intake_result)` stores the check (intake_log), before anything is moved.
    With `dry_run` nothing is recorded or moved and the pipeline is not run."""
    name = doc_storage.source_name(object_name)  # `<domain>/<file>`
    try:
        doc_storage.parse_upload(object_name)
    except doc_storage.BadUploadPath as bad:
        # no domain folder, or nested folders: rejected without downloading anything
        result = intake_check.IntakeResult(file_name=name, outcome=intake_check.REJECT, reason_code=bad.code,
                                           reason=bad.reason)
        if dry_run:
            return Outcome(CHECKED, object_name, result.to_dict())
        if record_intake is not None:
            record_intake(result)
        storage.mark_failed(object_name, review_metadata(result, now))
        return Outcome(REJECTED, object_name, result.to_dict())
    own_dir = None
    if work_dir is None:
        own_dir = tempfile.TemporaryDirectory(prefix="ingest_")
        work_dir = own_dir.name
    try:
        try:
            path = storage.download(object_name, work_dir)
        except doc_storage.ObjectNotFound:
            return Outcome(MISSING, object_name, {"reason": "the file is no longer in incoming/"})

        result = intake_check.check_document(path, source_name=name, active_hash=lookup_active_hash(name))
        detail = result.to_dict()
        if record_intake is not None and not dry_run:
            record_intake(result)  # recorded first, so the history is complete even if the move fails

        if result.outcome == intake_check.REJECT:
            if dry_run:
                return Outcome(CHECKED, object_name, detail)
            storage.mark_failed(object_name, review_metadata(result, now))
            return Outcome(REJECTED, object_name, detail)

        if result.outcome == intake_check.SKIP_UNCHANGED:
            if dry_run:
                return Outcome(CHECKED, object_name, detail)
            storage.mark_duplicate(object_name, {"note": result.reason, "content_hash": result.content_hash})
            return Outcome(DUPLICATE, object_name, detail)

        if dry_run:
            return Outcome(CHECKED, object_name, detail)
        if pipeline is None:
            raise PipelineNotAvailable("the ingestion pipeline is not wired in yet")
        try:
            report = pipeline(path, result)
        except DocumentRejected as rejected:
            # the document was read but cannot be ingested: it goes to failed/ and into the review list
            result = replace(result, outcome=intake_check.REJECT, reason_code=rejected.reason_code, reason=rejected.reason)
            if record_intake is not None:
                record_intake(result)
            storage.mark_failed(object_name, review_metadata(result, now))
            return Outcome(REJECTED, object_name, {**result.to_dict(), "details": rejected.details})
        storage.mark_processed(object_name, {"document_label": result.document_label, "content_hash": result.content_hash})
        if hasattr(report, "to_dict"):
            detail = {**detail, "ingested": report.to_dict()}
        return Outcome(INGESTED, object_name, detail)
    finally:
        if own_dir is not None:
            own_dir.cleanup()


class LeaseKeeper:
    """While a document is being handled, keeps extending the message's acknowledgement deadline
    (in a background thread), so the message is not delivered again before the work is done."""

    def __init__(self, subscriber, subscription_path: str, ack_id: str, deadline: int = ACK_DEADLINE_SECONDS,
                 interval: float = LEASE_INTERVAL_SECONDS, max_seconds: float = MAX_LEASE_SECONDS):
        self.subscriber, self.subscription_path, self.ack_id = subscriber, subscription_path, ack_id
        self.deadline, self.interval, self.max_seconds = deadline, interval, max_seconds
        self.extensions = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lease-keeper", daemon=True)

    def _run(self):
        waited = 0.0
        while not self._stop.wait(self.interval):
            waited += self.interval
            if waited > self.max_seconds:
                print("   (lease: gave up extending the deadline; the message will be delivered again)")
                return
            try:
                self.subscriber.modify_ack_deadline(request={
                    "subscription": self.subscription_path, "ack_ids": [self.ack_id],
                    "ack_deadline_seconds": self.deadline,
                })
                self.extensions += 1
            except Exception as e:
                print(f"   (lease: could not extend the deadline: {e})")
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False


def _ack(subscriber, subscription_path, ack_id):
    subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": [ack_id]})


def _nack(subscriber, subscription_path, ack_id):
    """Gives the message back at once; Pub/Sub delivers it again after the retry back-off."""
    subscriber.modify_ack_deadline(request={"subscription": subscription_path, "ack_ids": [ack_id], "ack_deadline_seconds": 0})


def handle_received(subscriber, subscription_path: str, received, process, storage=None,
                    lease_interval: float = LEASE_INTERVAL_SECONDS) -> Outcome:
    """Handles one pulled message: ignore it, process it, or give it back after an error.
    `process(object_name)` returns an Outcome. `storage` is used to flag a document on the last attempt."""
    ack_id = received.ack_id
    message = received.message
    notification = parse_notification(dict(message.attributes or {}))
    if notification is None:
        _ack(subscriber, subscription_path, ack_id)
        return Outcome(IGNORED, (message.attributes or {}).get("objectId"), {"reason": "not an upload to incoming/"})

    try:
        with LeaseKeeper(subscriber, subscription_path, ack_id, interval=lease_interval):
            outcome = process(notification.object_name)
    except Exception as e:
        attempt = getattr(received, "delivery_attempt", 0) or 0
        print(f"   ERROR on {notification.object_name} (delivery {attempt or '?'} of {MAX_DELIVERY_ATTEMPTS}): {type(e).__name__}: {e}")
        if attempt >= MAX_DELIVERY_ATTEMPTS and storage is not None:
            # The last attempt: flag the file for review now. The message is then given back once
            # more, and Pub/Sub forwards it to the dead-letter topic.
            try:
                storage.mark_failed(notification.object_name, {
                    "worker_error": f"{type(e).__name__}: {e}", "deliveries": attempt,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as move_error:
                print(f"   could not move the file to failed/: {move_error}")
        _nack(subscriber, subscription_path, ack_id)
        return Outcome(FAILED_AGAIN, notification.object_name, {"error": f"{type(e).__name__}: {e}", "delivery_attempt": attempt})

    _ack(subscriber, subscription_path, ack_id)
    return outcome


def run_worker(subscriber, subscription_path: str, process, storage=None, max_messages: int = None,
               empty_polls_to_stop: int = EMPTY_POLLS_TO_STOP, pull_timeout: float = PULL_TIMEOUT_SECONDS,
               lease_interval: float = LEASE_INTERVAL_SECONDS, should_stop=None) -> list:
    """Pulls and handles ONE message at a time until the queue is empty (or max_messages were handled, or `should_stop()` says
    to stop: the worker lease was lost). Returns the Outcome of every message handled."""
    outcomes, empty_polls = [], 0
    while max_messages is None or len(outcomes) < max_messages:
        if should_stop is not None and should_stop():
            print("   (the worker lease was lost: not taking more messages)")
            break
        try:
            response = subscriber.pull(request={"subscription": subscription_path, "max_messages": 1}, timeout=pull_timeout)
        except Exception as e:
            if type(e).__name__ != "DeadlineExceeded":  # an empty queue can make a pull time out
                raise
            response = None
        received_messages = list(response.received_messages) if response is not None else []
        if not received_messages:
            empty_polls += 1
            if empty_polls >= empty_polls_to_stop:
                break
            continue
        empty_polls = 0
        outcome = handle_received(subscriber, subscription_path, received_messages[0], process, storage, lease_interval)
        print(f"   {outcome.status}: {outcome.object_name}")
        outcomes.append(outcome)
    return outcomes


def default_pipeline(storage):
    """The ingestion pipeline for one document (models load on first use)."""
    from ingestion.pipeline import Pipeline

    return Pipeline(storage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", nargs="+", metavar="OBJECT",
                        help="download and check these objects (for example incoming/a.pdf); nothing is moved, the queue is not used")
    parser.add_argument("--with-manifest", action="store_true", help="with --check: look up the ingested version in the database")
    parser.add_argument("--run", action="store_true", help="handle the queued uploads one at a time")
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args(argv)

    storage = doc_storage.DocumentStorage()

    if args.check:
        lookup = manifest_active_hash if args.with_manifest else (lambda name: None)
        for object_name in args.check:
            outcome = process_document(object_name, storage, lookup, dry_run=True)
            print(json.dumps({"object": object_name, "status": outcome.status, **outcome.detail}, indent=2))
        return

    if args.run:
        from ingestion import worker_lease

        # Only one run handles the queue at a time. An execution that does not get the lease exits at once, before it
        # loads any model, so starting many executions (one per upload) costs almost nothing.
        with worker_lease.hold() as lease:
            if not lease.acquired:
                print("Another worker run holds the lease: exiting without touching the queue.")
                return
            pipeline = default_pipeline(storage)
            from google.cloud import pubsub_v1

            from ingestion import intake_log
            from shared.db import connection

            def record_intake(result):
                with connection() as conn:
                    intake_log.record_check(conn, result)

            subscriber = pubsub_v1.SubscriberClient()
            path = subscriber.subscription_path(os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT),
                                                os.environ.get("DOCS_SUBSCRIPTION", DEFAULT_SUBSCRIPTION))

            def process(object_name):
                return process_document(object_name, storage, manifest_active_hash, pipeline, record_intake=record_intake)

            run_worker(subscriber, path, process, storage, max_messages=args.max_messages,
                       should_stop=lease.lost.is_set)
            return

    parser.print_help()


if __name__ == "__main__":
    main()
