"""ingest_worker: one document at a time off the queue - read the message, download, check, move, acknowledge.
The bucket and the subscription are fakes; the intake check is the real one, run on generated PDFs."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from ingestion import doc_storage
from ingestion import ingest_worker as iw
from ingestion import intake_check as ic
from ingestion.errors import DocumentRejected

NOW = datetime(2026, 10, 3, 10, 15, 30, tzinfo=timezone.utc)
SUB = "projects/p/subscriptions/doc-events-sub"


# ---------- generated documents ----------

def pdf_bytes(kind="text"):
    doc = pymupdf.open()
    page = doc.new_page()
    if kind == "text":
        page.insert_text((72, 72), "lung nodule detection results")
    if kind == "locked":
        page.insert_text((72, 72), "secret")
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    return doc.tobytes()  # "blank": a page with nothing on it


# ---------- fakes ----------

class FakeStorage:
    """Stands in for DocumentStorage: files in memory, every move recorded."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.moves = []
        self.fail_marking = False
        self.downloaded_to = []

    def download(self, object_name, dest_dir):
        if object_name not in self.files:
            raise doc_storage.ObjectNotFound(object_name)
        path = Path(dest_dir) / Path(object_name).name
        path.write_bytes(self.files[object_name])
        self.downloaded_to.append(path)
        return path

    def _move(self, kind, object_name, metadata):
        if self.fail_marking:
            raise RuntimeError("bucket unavailable")
        self.moves.append((kind, object_name, metadata))
        return f"{kind}/{doc_storage.source_name(object_name)}"

    def mark_processed(self, object_name, metadata=None):
        return self._move("processed", object_name, metadata)

    def mark_failed(self, object_name, metadata=None):
        return self._move("failed", object_name, metadata)

    def mark_duplicate(self, object_name, metadata=None):
        return self._move("archived", object_name, metadata)


def message(object_name="incoming/lung-cancer/paper.pdf", event="OBJECT_FINALIZE", attempt=1, ack_id=None, bucket="docs"):
    attributes = {"eventType": event, "bucketId": bucket, "objectId": object_name, "objectGeneration": "1"}
    return SimpleNamespace(ack_id=ack_id or f"ack-{object_name}", delivery_attempt=attempt,
                           message=SimpleNamespace(attributes=attributes, data=b"{}"))


class FakeSubscriber:
    def __init__(self, messages=(), log=None):
        self.queue = list(messages)
        self.acked, self.deadlines, self.pulls = [], [], 0
        self.log = log if log is not None else []

    def pull(self, request, timeout=None):
        self.pulls += 1
        self.log.append("pull")
        assert request["max_messages"] == 1  # always one at a time
        batch, self.queue = self.queue[:1], self.queue[1:]
        return SimpleNamespace(received_messages=batch)

    def acknowledge(self, request):
        self.log.append("ack")
        self.acked.extend(request["ack_ids"])

    def modify_ack_deadline(self, request):
        self.log.append(f"deadline:{request['ack_deadline_seconds']}")
        self.deadlines.append((request["ack_ids"][0], request["ack_deadline_seconds"]))


def never_ingested(name):
    return None


# ---------- reading the message ----------

def test_an_upload_to_incoming_is_a_notification():
    note = iw.parse_notification({"eventType": "OBJECT_FINALIZE", "bucketId": "docs", "objectId": "incoming/lung-cancer/a.pdf", "objectGeneration": "17"})

    assert (note.bucket, note.object_name, note.event_type, note.generation) == ("docs", "incoming/lung-cancer/a.pdf", "OBJECT_FINALIZE", "17")


@pytest.mark.parametrize("attributes", [
    {"eventType": "OBJECT_DELETE", "objectId": "incoming/lung-cancer/a.pdf"},
    {"eventType": "OBJECT_METADATA_UPDATE", "objectId": "incoming/lung-cancer/a.pdf"},
    {"eventType": "OBJECT_FINALIZE", "objectId": "incoming/"},             # the folder placeholder
    {"eventType": "OBJECT_FINALIZE", "objectId": "processed/a.pdf"},
    {"eventType": "OBJECT_FINALIZE", "objectId": "failed/a.pdf"},
    {"eventType": "OBJECT_FINALIZE"},
    {"objectId": "incoming/lung-cancer/a.pdf"},
    {},
    None,
])
def test_anything_that_is_not_a_new_document_in_incoming_is_not_a_notification(attributes):
    assert iw.parse_notification(attributes) is None


def test_the_review_metadata_says_why_a_file_was_rejected():
    result = ic.IntakeResult(file_name="a.pdf", outcome=ic.REJECT, reason_code=ic.BLANK, reason="Every page is blank.",
                             content_hash="abc", size_bytes=12)

    assert iw.review_metadata(result, NOW) == {
        "intake_outcome": "reject", "intake_reason_code": "blank", "intake_reason": "Every page is blank.",
        "content_hash": "abc", "size_bytes": 12, "checked_at": "2026-10-03T10:15:30+00:00",
    }


# ---------- one document ----------

def test_an_accepted_document_runs_the_pipeline_and_moves_to_processed(tmp_path):
    data = pdf_bytes("text")
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": data})
    seen = {}

    def pipeline(path, result):
        seen["bytes"] = Path(path).read_bytes()
        seen["label"] = result.document_label

    outcome = iw.process_document("incoming/lung-cancer/paper.pdf", storage, never_ingested, pipeline)

    assert outcome.status == iw.INGESTED and seen["label"] == "text" and seen["bytes"] == data
    kind, name, metadata = storage.moves[0]
    assert (kind, name) == ("processed", "incoming/lung-cancer/paper.pdf")
    assert metadata["document_label"] == "text" and len(metadata["content_hash"]) == 64


def test_the_temporary_copy_is_removed_afterwards():
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": pdf_bytes("text")})

    iw.process_document("incoming/lung-cancer/paper.pdf", storage, never_ingested, pipeline=lambda path, result: None)

    assert storage.downloaded_to and not storage.downloaded_to[0].exists()


def test_a_given_work_folder_is_used_and_left_alone(tmp_path):
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": pdf_bytes("text")})

    iw.process_document("incoming/lung-cancer/paper.pdf", storage, never_ingested, pipeline=lambda p, r: None, work_dir=tmp_path)

    assert storage.downloaded_to == [tmp_path / "paper.pdf"] and (tmp_path / "paper.pdf").exists()


@pytest.mark.parametrize("kind,code", [("blank", ic.BLANK), ("locked", ic.PASSWORD_PROTECTED)])
def test_a_rejected_document_moves_to_failed_with_the_reason_and_never_reaches_the_pipeline(kind, code):
    storage = FakeStorage({"incoming/lung-cancer/bad.pdf": pdf_bytes(kind)})

    outcome = iw.process_document("incoming/lung-cancer/bad.pdf", storage, never_ingested,
                                  pipeline=lambda *a: pytest.fail("must not run the pipeline"), now=NOW)

    assert outcome.status == iw.REJECTED and outcome.detail["reason_code"] == code
    kind_moved, name, metadata = storage.moves[0]
    assert (kind_moved, name) == ("failed", "incoming/lung-cancer/bad.pdf")
    assert metadata["intake_reason_code"] == code and metadata["checked_at"] == "2026-10-03T10:15:30+00:00"


def test_an_unchanged_upload_is_archived_as_a_duplicate_and_not_ingested_again():
    data = pdf_bytes("text")  # one copy: every call builds a slightly different PDF, so the hash must come from these bytes
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": data})
    same = hashlib.sha256(data).hexdigest()

    outcome = iw.process_document("incoming/lung-cancer/paper.pdf", storage, lambda name: same,
                                  pipeline=lambda *a: pytest.fail("must not ingest a duplicate"))

    assert outcome.status == iw.DUPLICATE and outcome.detail["version_status"] == "unchanged"
    kind, name, metadata = storage.moves[0]
    assert (kind, name) == ("archived", "incoming/lung-cancer/paper.pdf") and metadata["content_hash"] == same


def test_a_changed_document_is_ingested_as_a_new_version():
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": pdf_bytes("text")})
    ingested = []

    outcome = iw.process_document("incoming/lung-cancer/paper.pdf", storage, lambda name: "0" * 64, lambda path, result: ingested.append(result))

    assert outcome.status == iw.INGESTED and ingested[0].version_status == "changed"


def test_the_manifest_is_asked_about_the_document_name_with_its_domain():
    storage = FakeStorage({"incoming/Land Cover/paper.pdf": pdf_bytes("text")})
    asked = []

    iw.process_document("incoming/Land Cover/paper.pdf", storage, lambda name: asked.append(name), lambda p, r: None)

    assert asked == ["land-cover/paper.pdf"]  # the folder is the domain, normalised


@pytest.mark.parametrize("object_name,code", [
    ("incoming/paper.pdf", doc_storage.NO_DOMAIN),
    ("incoming/medical/lung/paper.pdf", doc_storage.NESTED_FOLDER),
    ("incoming/!!!/paper.pdf", doc_storage.INVALID_DOMAIN),
])
def test_a_file_without_a_usable_domain_folder_is_rejected_without_being_downloaded(object_name, code):
    storage = FakeStorage({object_name: pdf_bytes("text")})
    recorded = []

    outcome = iw.process_document(object_name, storage, lambda name: pytest.fail("no manifest lookup"),
                                  pipeline=lambda *a: pytest.fail("no pipeline"), now=NOW, record_intake=recorded.append)

    assert outcome.status == iw.REJECTED and outcome.detail["reason_code"] == code
    assert storage.downloaded_to == []
    assert [m[0] for m in storage.moves] == ["failed"] and storage.moves[0][2]["intake_reason_code"] == code
    assert [r.reason_code for r in recorded] == [code]


def test_a_bad_path_on_a_dry_run_is_reported_and_nothing_moves():
    storage = FakeStorage({"incoming/paper.pdf": pdf_bytes("text")})
    outcome = iw.process_document("incoming/paper.pdf", storage, lambda name: None, dry_run=True)
    assert outcome.status == iw.CHECKED and outcome.detail["reason_code"] == doc_storage.NO_DOMAIN
    assert storage.moves == []


def test_a_file_that_is_no_longer_in_incoming_is_reported_missing_and_nothing_else_happens():
    storage = FakeStorage()

    outcome = iw.process_document("incoming/lung-cancer/gone.pdf", storage, lambda name: pytest.fail("no lookup for a missing file"))

    assert outcome.status == iw.MISSING and storage.moves == []


def test_a_document_is_left_in_place_when_there_is_no_pipeline_to_ingest_it():
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": pdf_bytes("text")})

    with pytest.raises(iw.PipelineNotAvailable):
        iw.process_document("incoming/lung-cancer/paper.pdf", storage, never_ingested, pipeline=None)

    assert storage.moves == []


def test_a_document_the_pipeline_rejects_goes_to_failed_and_into_the_review_list():
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": pdf_bytes("text")})
    recorded = []

    def pipeline(path, result):
        raise DocumentRejected("too_empty", "60% of the pages are nearly empty", {"pages": 5})

    outcome = iw.process_document("incoming/lung-cancer/paper.pdf", storage, never_ingested, pipeline, now=NOW,
                                  record_intake=recorded.append)

    assert outcome.status == iw.REJECTED
    assert outcome.detail["reason_code"] == "too_empty" and outcome.detail["details"] == {"pages": 5}
    assert [m[0] for m in storage.moves] == ["failed"]
    assert storage.moves[0][2]["intake_reason_code"] == "too_empty"
    # the check is recorded twice: first as accepted, then again as the rejection that replaces it
    assert [r.outcome for r in recorded] == [ic.ACCEPT, ic.REJECT]
    assert recorded[-1].reason_code == "too_empty"


def test_the_pipeline_report_is_kept_in_the_outcome():
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": pdf_bytes("text")})
    outcome = iw.process_document("incoming/lung-cancer/paper.pdf", storage, never_ingested,
                                  pipeline=lambda p, r: SimpleNamespace(to_dict=lambda: {"children": 7}))

    assert outcome.status == iw.INGESTED and outcome.detail["ingested"] == {"children": 7}


def test_a_pipeline_error_leaves_the_file_where_it_is_and_reaches_the_caller():
    storage = FakeStorage({"incoming/lung-cancer/paper.pdf": pdf_bytes("text")})

    def broken(path, result):
        raise RuntimeError("extraction failed")

    with pytest.raises(RuntimeError, match="extraction failed"):
        iw.process_document("incoming/lung-cancer/paper.pdf", storage, never_ingested, broken)

    assert storage.moves == []


@pytest.mark.parametrize("kind,had_hash", [("text", None), ("blank", None), ("text", "same")])
def test_a_dry_run_checks_but_moves_nothing_and_runs_no_pipeline(kind, had_hash):
    data = pdf_bytes(kind)
    storage = FakeStorage({"incoming/lung-cancer/a.pdf": data})
    lookup = (lambda name: hashlib.sha256(data).hexdigest()) if had_hash else never_ingested

    outcome = iw.process_document("incoming/lung-cancer/a.pdf", storage, lookup, pipeline=lambda *a: pytest.fail("no pipeline"), dry_run=True)

    assert outcome.status == iw.CHECKED and storage.moves == []
    assert "outcome" in outcome.detail and "page_labels" in outcome.detail


def test_a_dry_run_works_without_a_pipeline_even_for_an_accepted_document():
    storage = FakeStorage({"incoming/lung-cancer/a.pdf": pdf_bytes("text")})

    outcome = iw.process_document("incoming/lung-cancer/a.pdf", storage, never_ingested, pipeline=None, dry_run=True)

    assert outcome.status == iw.CHECKED and outcome.detail["outcome"] == "accept"


# ---------- recording the check (intake_log) ----------

def test_every_check_is_recorded_before_anything_is_moved():
    for kind, lookup_hash in (("text", None), ("blank", None), ("text", "same")):
        data = pdf_bytes(kind)
        storage = FakeStorage({"incoming/lung-cancer/a.pdf": data})
        recorded = []

        def record(result, storage=storage, recorded=recorded):
            assert storage.moves == []  # nothing moved yet
            recorded.append(result)

        lookup = (lambda name, data=data: hashlib.sha256(data).hexdigest()) if lookup_hash else never_ingested
        iw.process_document("incoming/lung-cancer/a.pdf", storage, lookup, pipeline=lambda p, r: None, record_intake=record)

        assert len(recorded) == 1 and recorded[0].file_name == "lung-cancer/a.pdf"
        assert recorded[0].outcome in (ic.ACCEPT, ic.REJECT, ic.SKIP_UNCHANGED) and storage.moves


def test_the_recorded_check_is_the_one_the_decision_was_made_on():
    storage = FakeStorage({"incoming/lung-cancer/bad.pdf": pdf_bytes("locked")})
    recorded = []

    outcome = iw.process_document("incoming/lung-cancer/bad.pdf", storage, never_ingested, record_intake=recorded.append)

    assert recorded[0].reason_code == ic.PASSWORD_PROTECTED and outcome.detail["reason_code"] == ic.PASSWORD_PROTECTED


def test_a_dry_run_records_nothing():
    storage = FakeStorage({"incoming/lung-cancer/a.pdf": pdf_bytes("text")})

    iw.process_document("incoming/lung-cancer/a.pdf", storage, never_ingested, dry_run=True,
                        record_intake=lambda r: pytest.fail("a dry run must not write to the database"))


def test_a_missing_file_records_nothing():
    iw.process_document("incoming/lung-cancer/gone.pdf", FakeStorage(), never_ingested,
                        record_intake=lambda r: pytest.fail("there was no check to record"))


def test_if_recording_fails_nothing_is_moved_and_the_error_reaches_the_caller():
    storage = FakeStorage({"incoming/lung-cancer/a.pdf": pdf_bytes("text")})

    def broken(result):
        raise ConnectionError("database down")

    with pytest.raises(ConnectionError):
        iw.process_document("incoming/lung-cancer/a.pdf", storage, never_ingested, pipeline=lambda p, r: None, record_intake=broken)

    assert storage.moves == []  # the file stays in incoming/ and the message is retried


# ---------- keeping the message alive ----------

def test_the_lease_keeper_extends_the_deadline_while_it_runs_and_stops_when_it_exits():
    subscriber = FakeSubscriber()

    with iw.LeaseKeeper(subscriber, SUB, "ack-1", interval=0.01):
        time.sleep(0.15)
    calls_while_running = len(subscriber.deadlines)
    time.sleep(0.05)

    assert calls_while_running >= 3
    assert all(call == ("ack-1", iw.ACK_DEADLINE_SECONDS) for call in subscriber.deadlines)
    assert len(subscriber.deadlines) == calls_while_running  # nothing after it stopped


def test_the_lease_keeper_stops_quietly_when_the_deadline_can_no_longer_be_extended():
    class Failing(FakeSubscriber):
        def modify_ack_deadline(self, request):
            super().modify_ack_deadline(request)
            raise RuntimeError("ack id expired")

    subscriber = Failing()

    with iw.LeaseKeeper(subscriber, SUB, "ack-1", interval=0.01):
        time.sleep(0.1)

    assert len(subscriber.deadlines) == 1  # tried once, then gave up without crashing


def test_the_lease_keeper_gives_up_after_the_maximum_time():
    subscriber = FakeSubscriber()

    with iw.LeaseKeeper(subscriber, SUB, "ack-1", interval=0.01, max_seconds=0.035):
        time.sleep(0.2)

    assert 1 <= len(subscriber.deadlines) <= 4


# ---------- one message ----------

def test_a_message_that_is_not_a_document_upload_is_acknowledged_and_ignored():
    subscriber = FakeSubscriber()
    received = message("incoming/", event="OBJECT_FINALIZE")

    outcome = iw.handle_received(subscriber, SUB, received, lambda name: pytest.fail("nothing to process"))

    assert outcome.status == iw.IGNORED and subscriber.acked == [received.ack_id]


def test_a_handled_message_is_acknowledged_once_and_never_given_back():
    subscriber = FakeSubscriber()
    received = message("incoming/lung-cancer/a.pdf")

    outcome = iw.handle_received(subscriber, SUB, received, lambda name: iw.Outcome(iw.INGESTED, name))

    assert outcome.status == iw.INGESTED and subscriber.acked == [received.ack_id]
    assert [d for d in subscriber.deadlines if d[1] == 0] == []


@pytest.mark.parametrize("status", [iw.REJECTED, iw.DUPLICATE, iw.MISSING])
def test_a_rejected_duplicate_or_missing_document_is_acknowledged_so_the_queue_moves_on(status):
    subscriber = FakeSubscriber()
    received = message("incoming/lung-cancer/a.pdf")

    iw.handle_received(subscriber, SUB, received, lambda name: iw.Outcome(status, name))

    assert subscriber.acked == [received.ack_id]


def test_the_deadline_is_extended_while_the_document_is_being_worked_on():
    subscriber = FakeSubscriber()
    received = message("incoming/lung-cancer/a.pdf")

    def slow(name):
        time.sleep(0.1)
        return iw.Outcome(iw.INGESTED, name)

    iw.handle_received(subscriber, SUB, received, slow, lease_interval=0.01)

    extensions = [e for e in subscriber.log if e == f"deadline:{iw.ACK_DEADLINE_SECONDS}"]
    assert len(extensions) >= 3 and subscriber.log[-1] == "ack"  # extended during the work, acknowledged only at the end


def test_an_error_gives_the_message_back_instead_of_acknowledging_it():
    subscriber = FakeSubscriber()
    storage = FakeStorage()
    received = message("incoming/lung-cancer/a.pdf", attempt=2)

    def broken(name):
        raise RuntimeError("database down")

    outcome = iw.handle_received(subscriber, SUB, received, broken, storage, lease_interval=60)

    assert outcome.status == iw.FAILED_AGAIN and outcome.detail["delivery_attempt"] == 2
    assert "database down" in outcome.detail["error"]
    assert subscriber.acked == [] and subscriber.deadlines[-1] == (received.ack_id, 0)  # given back at once
    assert storage.moves == []  # not the last attempt: the file stays in incoming/


def test_on_the_last_attempt_the_file_is_flagged_for_review_before_the_message_is_given_back():
    subscriber = FakeSubscriber()
    storage = FakeStorage()
    received = message("incoming/lung-cancer/a.pdf", attempt=iw.MAX_DELIVERY_ATTEMPTS)

    def broken(name):
        raise ValueError("cannot parse")

    outcome = iw.handle_received(subscriber, SUB, received, broken, storage, lease_interval=60)

    assert outcome.status == iw.FAILED_AGAIN and subscriber.acked == []
    kind, name, metadata = storage.moves[0]
    assert (kind, name) == ("failed", "incoming/lung-cancer/a.pdf")
    assert metadata["worker_error"] == "ValueError: cannot parse" and metadata["deliveries"] == iw.MAX_DELIVERY_ATTEMPTS
    assert subscriber.deadlines[-1] == (received.ack_id, 0)  # then given back, so Pub/Sub forwards it to the dead-letter topic


def test_failing_to_move_the_file_on_the_last_attempt_does_not_stop_the_message_being_given_back():
    subscriber = FakeSubscriber()
    storage = FakeStorage()
    storage.fail_marking = True
    received = message("incoming/lung-cancer/a.pdf", attempt=iw.MAX_DELIVERY_ATTEMPTS)

    def broken(name):
        raise ValueError("cannot parse")

    outcome = iw.handle_received(subscriber, SUB, received, broken, storage, lease_interval=60)

    assert outcome.status == iw.FAILED_AGAIN and subscriber.deadlines[-1] == (received.ack_id, 0)


def test_an_unknown_delivery_attempt_is_not_treated_as_the_last():
    subscriber = FakeSubscriber()
    storage = FakeStorage()
    received = message("incoming/lung-cancer/a.pdf", attempt=0)  # Pub/Sub reports 0 when it does not track attempts

    iw.handle_received(subscriber, SUB, received, lambda name: (_ for _ in ()).throw(RuntimeError("x")), storage, lease_interval=60)

    assert storage.moves == []


# ---------- the loop ----------

def test_messages_are_handled_one_at_a_time_each_acknowledged_before_the_next_is_pulled():
    log = []
    subscriber = FakeSubscriber([message("incoming/lung-cancer/a.pdf"), message("incoming/lung-cancer/b.pdf")], log)

    def process(name):
        log.append(f"process:{name}")
        return iw.Outcome(iw.INGESTED, name)

    outcomes = iw.run_worker(subscriber, SUB, process, pull_timeout=1, lease_interval=60)

    assert [o.object_name for o in outcomes] == ["incoming/lung-cancer/a.pdf", "incoming/lung-cancer/b.pdf"]
    assert log == ["pull", "process:incoming/lung-cancer/a.pdf", "ack", "pull", "process:incoming/lung-cancer/b.pdf", "ack", "pull", "pull"]


def test_the_worker_stops_after_two_empty_pulls_in_a_row():
    subscriber = FakeSubscriber()

    assert iw.run_worker(subscriber, SUB, lambda name: pytest.fail("nothing to do")) == []
    assert subscriber.pulls == iw.EMPTY_POLLS_TO_STOP


def test_a_message_after_an_empty_pull_resets_the_count():
    class Flaky(FakeSubscriber):
        def pull(self, request, timeout=None):
            self.pulls += 1
            if self.pulls == 1:
                return SimpleNamespace(received_messages=[])  # one empty pull first
            return super().pull(request, timeout)

    subscriber = Flaky([message("incoming/lung-cancer/a.pdf")])

    outcomes = iw.run_worker(subscriber, SUB, lambda name: iw.Outcome(iw.INGESTED, name), lease_interval=60)

    assert len(outcomes) == 1


def test_the_worker_can_be_limited_to_a_number_of_messages():
    subscriber = FakeSubscriber([message(f"incoming/{n}.pdf") for n in "abc"])

    outcomes = iw.run_worker(subscriber, SUB, lambda name: iw.Outcome(iw.INGESTED, name), max_messages=2, lease_interval=60)

    assert len(outcomes) == 2 and len(subscriber.queue) == 1


def test_a_pull_that_times_out_counts_as_an_empty_queue():
    class DeadlineExceeded(Exception):
        pass

    class TimingOut(FakeSubscriber):
        def pull(self, request, timeout=None):
            self.pulls += 1
            raise DeadlineExceeded("no messages")

    subscriber = TimingOut()

    assert iw.run_worker(subscriber, SUB, lambda name: None) == [] and subscriber.pulls == iw.EMPTY_POLLS_TO_STOP


def test_any_other_pull_error_is_not_swallowed():
    class Down(FakeSubscriber):
        def pull(self, request, timeout=None):
            raise ConnectionError("pub/sub unreachable")

    with pytest.raises(ConnectionError):
        iw.run_worker(Down(), SUB, lambda name: None)


def test_a_failing_message_does_not_stop_the_loop_it_comes_back_later():
    subscriber = FakeSubscriber([message("incoming/lung-cancer/bad.pdf"), message("incoming/lung-cancer/good.pdf")])

    def process(name):
        if "bad" in name:
            raise RuntimeError("boom")
        return iw.Outcome(iw.INGESTED, name)

    outcomes = iw.run_worker(subscriber, SUB, process, FakeStorage(), lease_interval=60)

    assert [o.status for o in outcomes] == [iw.FAILED_AGAIN, iw.INGESTED]
    assert subscriber.acked == ["ack-incoming/lung-cancer/good.pdf"]


# ---------- the command line ----------

def test_the_default_pipeline_is_the_real_one_for_this_storage():
    from ingestion.pipeline import Pipeline

    storage = FakeStorage()
    pipeline = iw.default_pipeline(storage)

    assert isinstance(pipeline, Pipeline) and pipeline.storage is storage


def test_check_downloads_and_checks_the_named_files_without_changing_anything(monkeypatch, capsys):
    storage = FakeStorage({"incoming/lung-cancer/a.pdf": pdf_bytes("text"), "incoming/lung-cancer/b.pdf": pdf_bytes("blank")})
    monkeypatch.setattr(iw.doc_storage, "DocumentStorage", lambda *a, **k: storage)

    iw.main(["--check", "incoming/lung-cancer/a.pdf", "incoming/lung-cancer/b.pdf"])

    output = capsys.readouterr().out
    reports = [json.loads(chunk) for chunk in output.replace("}\n{", "}\x00{").split("\x00")]
    assert [(r["object"], r["status"], r["outcome"]) for r in reports] == [
        ("incoming/lung-cancer/a.pdf", "checked", "accept"), ("incoming/lung-cancer/b.pdf", "checked", "reject")]
    assert storage.moves == []


def test_check_asks_the_manifest_only_when_told_to(monkeypatch, capsys):
    storage = FakeStorage({"incoming/lung-cancer/a.pdf": pdf_bytes("text")})
    monkeypatch.setattr(iw.doc_storage, "DocumentStorage", lambda *a, **k: storage)
    asked = []
    monkeypatch.setattr(iw, "manifest_active_hash", lambda name: asked.append(name))

    iw.main(["--check", "incoming/lung-cancer/a.pdf"])
    assert asked == []
    iw.main(["--check", "incoming/lung-cancer/a.pdf", "--with-manifest"])
    assert asked == ["lung-cancer/a.pdf"]


def test_with_no_arguments_the_help_is_shown(capsys):
    iw.main([])

    assert "--check" in capsys.readouterr().out
