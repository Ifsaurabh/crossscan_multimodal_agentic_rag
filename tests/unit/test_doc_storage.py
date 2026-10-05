"""doc_storage: downloading from and moving files inside the documents bucket. The bucket is a fake held in memory."""
from datetime import datetime, timezone

import pytest

from ingestion import doc_storage as ds

NOW = datetime(2026, 10, 3, 10, 15, 30, tzinfo=timezone.utc)
STAMP = "20261003T101530Z"


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = None
        self.patched = 0

    def exists(self):
        return self.name in self.bucket.objects

    def download_to_filename(self, filename):
        with open(filename, "wb") as f:
            f.write(self.bucket.objects[self.name]["data"])

    def upload_from_string(self, data, content_type=None):
        self.bucket.objects[self.name] = {"data": data, "metadata": {}, "content_type": content_type,
                                          "cache_control": getattr(self, "cache_control", None)}

    def delete(self):
        del self.bucket.objects[self.name]

    def patch(self):
        self.patched += 1
        self.bucket.objects[self.name]["metadata"] = dict(self.metadata or {})


class FakeBucket:
    def __init__(self, objects=None):
        self.objects = {name: {"data": data, "metadata": {}} for name, data in (objects or {}).items()}
        self.renames = []

    def blob(self, name):
        return FakeBlob(self, name)

    def list_blobs(self, prefix=""):
        return [FakeBlob(self, n) for n in list(self.objects) if n.startswith(prefix)]

    def rename_blob(self, blob, new_name):
        self.renames.append((blob.name, new_name))
        self.objects[new_name] = self.objects.pop(blob.name)
        return FakeBlob(self, new_name)


class FakeClient:
    def __init__(self, bucket):
        self._bucket = bucket
        self.asked_for = []

    def bucket(self, name):
        self.asked_for.append(name)
        return self._bucket


def storage_with(objects=None):
    bucket = FakeBucket(objects)
    client = FakeClient(bucket)
    return ds.DocumentStorage("docs", client=client, clock=lambda: NOW), bucket, client


# ---------- names ----------

def test_the_source_name_is_the_normalised_domain_and_the_file():
    assert ds.source_name("incoming/lung-cancer/paper.pdf") == "lung-cancer/paper.pdf"
    assert ds.source_name("incoming/Land Cover/paper v2.pdf") == "land-cover/paper v2.pdf"
    assert ds.source_name("processed/ai-security/a.pdf") == "ai-security/a.pdf"


def test_a_path_that_is_not_a_valid_upload_keeps_only_its_file_name():
    assert ds.source_name("incoming/paper.pdf") == "paper.pdf"
    assert ds.source_name("incoming/sub/dir/paper v2.pdf") == "paper v2.pdf"
    assert ds.source_name("paper.pdf") == "paper.pdf"


@pytest.mark.parametrize("folder,domain", [("Lung Cancer", "lung-cancer"), ("  Land_cover!! ", "land-cover"),
                                           ("AI-Security", "ai-security"), ("!!!", ""), ("", ""), (None, "")])
def test_normalise_domain(folder, domain):
    assert ds.normalise_domain(folder) == domain


def test_parse_upload_reads_the_domain_folder_and_file():
    up = ds.parse_upload("incoming/Lung Cancer/paper.pdf")
    assert (up.domain, up.file_name, up.name) == ("lung-cancer", "paper.pdf", "lung-cancer/paper.pdf")


@pytest.mark.parametrize("name,code", [("incoming/paper.pdf", ds.NO_DOMAIN),
                                       ("incoming/a/b/paper.pdf", ds.NESTED_FOLDER),
                                       ("incoming/!!!/paper.pdf", ds.INVALID_DOMAIN)])
def test_parse_upload_rejects_a_missing_nested_or_unusable_folder(name, code):
    with pytest.raises(ds.BadUploadPath) as bad:
        ds.parse_upload(name)
    assert bad.value.code == code and bad.value.reason


@pytest.mark.parametrize("name,expected", [
    ("incoming/paper.pdf", True),
    ("incoming/sub/paper.pdf", True),
    ("incoming/", False),              # the folder placeholder made by "Create folder" in the console
    ("incoming/sub/", False),
    ("processed/paper.pdf", False),
    ("failed/paper.pdf", False),
    ("paper.pdf", False),
    ("", False),
    (None, False),
])
def test_only_a_file_waiting_in_incoming_is_a_document(name, expected):
    assert ds.is_document_object(name) is expected


def test_the_bucket_name_comes_from_the_environment_or_the_default(monkeypatch):
    monkeypatch.delenv("DOCS_BUCKET", raising=False)
    assert ds.bucket_name_from_env() == "crossscan-docs-upload-here"
    monkeypatch.setenv("DOCS_BUCKET", "other-bucket")
    assert ds.bucket_name_from_env() == "other-bucket"
    assert ds.DocumentStorage(client=FakeClient(FakeBucket())).bucket_name == "other-bucket"


def test_the_client_and_bucket_are_only_created_when_first_needed():
    bucket = FakeBucket({"incoming/lung-cancer/a.pdf": b"x"})
    client = FakeClient(bucket)
    storage = ds.DocumentStorage("docs", client=client)

    assert client.asked_for == []  # nothing yet
    storage.exists("incoming/lung-cancer/a.pdf")
    storage.exists("incoming/lung-cancer/a.pdf")
    assert client.asked_for == ["docs"]  # once, then reused


# ---------- download ----------

def test_download_saves_the_object_under_its_file_name(tmp_path):
    storage, _, _ = storage_with({"incoming/lung-cancer/paper.pdf": b"%PDF content"})

    path = storage.download("incoming/lung-cancer/paper.pdf", tmp_path)

    assert path == tmp_path / "paper.pdf" and path.read_bytes() == b"%PDF content"


def test_downloading_an_object_that_is_not_there_says_so(tmp_path):
    storage, _, _ = storage_with()

    with pytest.raises(ds.ObjectNotFound, match="incoming/lung-cancer/gone.pdf"):
        storage.download("incoming/lung-cancer/gone.pdf", tmp_path)


def test_exists_reports_whether_the_object_is_in_the_bucket():
    storage, _, _ = storage_with({"incoming/lung-cancer/a.pdf": b"x"})

    assert storage.exists("incoming/lung-cancer/a.pdf") and not storage.exists("incoming/lung-cancer/b.pdf")


# ---------- moving ----------

def test_a_processed_document_moves_to_processed():
    storage, bucket, _ = storage_with({"incoming/lung-cancer/paper.pdf": b"data"})

    target = storage.mark_processed("incoming/lung-cancer/paper.pdf")

    assert target == "processed/lung-cancer/paper.pdf"
    assert set(bucket.objects) == {"processed/lung-cancer/paper.pdf"} and bucket.objects["processed/lung-cancer/paper.pdf"]["data"] == b"data"


def test_a_failed_document_moves_to_failed_with_the_reason_as_metadata():
    storage, bucket, _ = storage_with({"incoming/lung-cancer/locked.pdf": b"data"})

    target = storage.mark_failed("incoming/lung-cancer/locked.pdf", {"intake_reason_code": "password_protected", "intake_reason": "locked"})

    assert target == "failed/lung-cancer/locked.pdf" and set(bucket.objects) == {"failed/lung-cancer/locked.pdf"}
    assert bucket.objects["failed/lung-cancer/locked.pdf"]["metadata"] == {"intake_reason_code": "password_protected", "intake_reason": "locked"}


def test_a_move_without_metadata_does_not_touch_it():
    storage, bucket, _ = storage_with({"incoming/lung-cancer/a.pdf": b"x"})

    storage.mark_processed("incoming/lung-cancer/a.pdf")

    assert bucket.objects["processed/lung-cancer/a.pdf"]["metadata"] == {}


def test_metadata_values_are_text_and_long_ones_are_cut():
    storage, bucket, _ = storage_with({"incoming/lung-cancer/a.pdf": b"x"})

    storage.mark_failed("incoming/lung-cancer/a.pdf", {"count": 5, "nothing": None, "long": "x" * 5000})

    metadata = bucket.objects["failed/lung-cancer/a.pdf"]["metadata"]
    assert metadata["count"] == "5" and metadata["nothing"] == ""
    assert len(metadata["long"]) == ds.MAX_METADATA_VALUE_CHARS


def test_a_replaced_version_is_archived_not_overwritten():
    storage, bucket, _ = storage_with({"incoming/lung-cancer/paper.pdf": b"new", "processed/lung-cancer/paper.pdf": b"old"})

    storage.mark_processed("incoming/lung-cancer/paper.pdf")

    assert bucket.objects["processed/lung-cancer/paper.pdf"]["data"] == b"new"
    assert bucket.objects[f"archived/lung-cancer/{STAMP}_paper.pdf"]["data"] == b"old"
    assert bucket.renames == [("processed/lung-cancer/paper.pdf", f"archived/lung-cancer/{STAMP}_paper.pdf"), ("incoming/lung-cancer/paper.pdf", "processed/lung-cancer/paper.pdf")]


def test_an_earlier_failure_of_the_same_name_is_archived_too():
    storage, bucket, _ = storage_with({"incoming/lung-cancer/a.pdf": b"second", "failed/lung-cancer/a.pdf": b"first"})

    storage.mark_failed("incoming/lung-cancer/a.pdf")

    assert bucket.objects["failed/lung-cancer/a.pdf"]["data"] == b"second"
    assert bucket.objects[f"archived/lung-cancer/{STAMP}_a.pdf"]["data"] == b"first"


def test_a_duplicate_upload_goes_to_archived_under_a_time_stamped_name():
    storage, bucket, _ = storage_with({"incoming/lung-cancer/paper.pdf": b"same", "processed/lung-cancer/paper.pdf": b"same"})

    target = storage.mark_duplicate("incoming/lung-cancer/paper.pdf", {"note": "same content as the ingested version"})

    assert target == f"archived/lung-cancer/{STAMP}_paper.pdf"
    assert set(bucket.objects) == {"processed/lung-cancer/paper.pdf", target}  # the ingested copy is left alone
    assert bucket.objects[target]["metadata"] == {"note": "same content as the ingested version"}


def test_moving_an_object_that_is_not_there_says_so():
    storage, _, _ = storage_with()

    with pytest.raises(ds.ObjectNotFound):
        storage.mark_processed("incoming/lung-cancer/gone.pdf")


def test_the_domain_folder_is_kept_when_a_file_is_moved_and_the_same_name_in_two_domains_never_meets():
    storage, bucket, _ = storage_with({"incoming/Land Cover/report.pdf": b"a", "incoming/ai-security/report.pdf": b"b"})

    assert storage.mark_processed("incoming/Land Cover/report.pdf") == "processed/land-cover/report.pdf"
    assert storage.mark_processed("incoming/ai-security/report.pdf") == "processed/ai-security/report.pdf"
    assert set(bucket.objects) == {"processed/land-cover/report.pdf", "processed/ai-security/report.pdf"}


def test_moving_to_the_same_place_does_nothing():
    storage, bucket, _ = storage_with({"processed/lung-cancer/a.pdf": b"x"})

    assert storage.mark_processed("processed/lung-cancer/a.pdf") == "processed/lung-cancer/a.pdf"
    assert bucket.renames == []


# ---------- images ----------

def test_upload_bytes_stores_data_with_type_and_cache_control():
    storage, bucket, _ = storage_with()
    storage.upload_bytes("images/p/a.png", b"png", "image/png", "public, max-age=3600")
    obj = bucket.objects["images/p/a.png"]
    assert obj["data"] == b"png" and obj["content_type"] == "image/png" and obj["cache_control"] == "public, max-age=3600"


def test_delete_prefix_removes_everything_except_kept():
    storage, bucket, _ = storage_with({"images/p/a.png": b"1", "images/p/b.png": b"2", "images/q/c.png": b"3"})
    assert storage.delete_prefix("images/p/", keep=["images/p/a.png"]) == 1
    assert set(bucket.objects) == {"images/p/a.png", "images/q/c.png"}
