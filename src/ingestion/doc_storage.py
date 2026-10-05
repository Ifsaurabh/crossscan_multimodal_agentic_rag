"""doc_storage: the documents bucket.

Documents are uploaded to `incoming/<domain>/<file>`: the folder is the document's DOMAIN.
A new folder is a new domain. A file with no folder, or inside a folder inside a folder, is rejected.
The document's name everywhere (manifest, database, moved files) is `<domain>/<file>`, so two files with the
same name in different domains never meet. The worker downloads one, checks it, and moves it:
    processed/   ingested
    failed/      rejected by a check, or still failing after the retries (flagged for review)
    archived/    an older version of a document that was replaced, or an upload that duplicated one

A move never overwrites: if the target name already holds a file, that file is first moved to
`archived/` under a time-stamped name. (The bucket also keeps soft-deleted objects for 60 days.)
The reason a file was rejected is stored on the moved object as custom metadata.

Credentials are Google's Application Default Credentials: the service account on Cloud Run, or
`gcloud auth application-default login` on a laptop. The bucket name comes from the DOCS_BUCKET
environment variable.
"""
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEFAULT_BUCKET = "crossscan-docs-upload-here"

INCOMING = "incoming/"
PROCESSED = "processed/"
FAILED = "failed/"
ARCHIVED = "archived/"

MAX_METADATA_VALUE_CHARS = 1000  # custom metadata is limited in size; a long reason is cut
MAX_DOMAIN_CHARS = 60

NO_DOMAIN = "no_domain"
NESTED_FOLDER = "nested_folder"
INVALID_DOMAIN = "invalid_domain"


class ObjectNotFound(Exception):
    """The object is not in the bucket (already moved, deleted, or never uploaded)."""


def bucket_name_from_env() -> str:
    return os.environ.get("DOCS_BUCKET", DEFAULT_BUCKET)


class BadUploadPath(Exception):
    """The object is not in `incoming/<domain>/<file>`; `code` and `reason` are for the review list."""

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass
class Upload:
    domain: str       # the normalised folder name
    file_name: str
    name: str         # `<domain>/<file>`: the document's name everywhere


def normalise_domain(folder: str) -> str:
    """`Lung Cancer` -> `lung-cancer`: lowercase letters and digits, words joined by hyphens. '' if nothing is left."""
    return "-".join(re.findall(r"[a-z0-9]+", (folder or "").lower()))[:MAX_DOMAIN_CHARS].strip("-")


def _after_prefix(object_name: str) -> str:
    for prefix in (INCOMING, PROCESSED, FAILED, ARCHIVED):
        if object_name.startswith(prefix):
            return object_name[len(prefix):]
    return object_name


def parse_upload(object_name: str) -> Upload:
    """The domain and file of an object in incoming/ (or the same path under processed/, failed/, archived/).
    Raises BadUploadPath for a file with no folder, in nested folders, or in a folder with no usable name."""
    parts = [p for p in _after_prefix(object_name).split("/")]
    if len(parts) == 1:
        raise BadUploadPath(NO_DOMAIN, "The file was uploaded without a domain folder: upload to incoming/<domain>/<file>.")
    if len(parts) > 2:
        raise BadUploadPath(NESTED_FOLDER, "The file is in a folder inside a folder: only one level, incoming/<domain>/<file>, is read.")
    domain = normalise_domain(parts[0])
    if not domain or not parts[1]:
        raise BadUploadPath(INVALID_DOMAIN, f"The folder name {parts[0]!r} has no letters or digits to make a domain from.")
    return Upload(domain=domain, file_name=parts[1], name=f"{domain}/{parts[1]}")


IMAGES_PREFIX = "images/"


def image_prefix(name: str) -> str:
    """The bucket folder that holds the pictures of the document `<domain>/<file>`: images/<domain>/<file stem>/."""
    domain, _, file_name = name.partition("/")
    return f"{IMAGES_PREFIX}{domain}/{PurePosixPath(file_name).stem}/"


def source_name(object_name: str) -> str:
    """The document's name in the manifest: `<domain>/<file>`. For a path that is not a valid upload (it is being
    rejected), the file name alone."""
    try:
        return parse_upload(object_name).name
    except BadUploadPath:
        return PurePosixPath(object_name).name


def is_document_object(object_name: str) -> bool:
    """True for a file waiting in incoming/. False for the folder placeholder (`incoming/`, created
    when someone makes a folder in the console) and for anything outside incoming/."""
    return bool(object_name) and object_name.startswith(INCOMING) and not object_name.endswith("/")


def _stamp(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ")


def _clean_metadata(metadata: dict) -> dict:
    return {str(key): "" if value is None else str(value)[:MAX_METADATA_VALUE_CHARS] for key, value in metadata.items()}


class DocumentStorage:
    """Everything the worker does with the bucket. `client` is a google.cloud.storage.Client
    (tests pass a fake); `clock` returns the current time (tests pass a fixed one)."""

    def __init__(self, bucket_name: str = None, client=None, clock=None):
        self.bucket_name = bucket_name or bucket_name_from_env()
        self._client = client
        self._bucket = None
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def bucket(self):
        if self._bucket is None:
            if self._client is None:
                from google.cloud import storage  # imported here so tests and tools need no credentials
                self._client = storage.Client()
            self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket

    def exists(self, object_name: str) -> bool:
        return self.bucket.blob(object_name).exists()

    def download(self, object_name: str, dest_dir) -> Path:
        """Downloads the object into dest_dir, under its file name, and returns the local path."""
        blob = self.bucket.blob(object_name)
        if not blob.exists():
            raise ObjectNotFound(f"gs://{self.bucket_name}/{object_name} is not in the bucket")
        path = Path(dest_dir) / PurePosixPath(object_name).name
        blob.download_to_filename(str(path))
        return path

    def _rename(self, object_name: str, target: str, metadata: dict = None) -> str:
        """Moves object_name to target, first archiving whatever already sits at target."""
        if target == object_name:
            return target
        source = self.bucket.blob(object_name)
        if not source.exists():
            raise ObjectNotFound(f"gs://{self.bucket_name}/{object_name} is not in the bucket")
        existing = self.bucket.blob(target)
        if existing.exists():
            self.bucket.rename_blob(existing, self._archive_name(target))
        moved = self.bucket.rename_blob(source, target)
        if metadata:
            moved.metadata = _clean_metadata(metadata)
            moved.patch()
        return target

    def _archive_name(self, name_or_object: str) -> str:
        """archived/<domain>/<time>_<file> (archived/<time>_<file> when there is no domain)."""
        name = source_name(name_or_object)
        folder, _, file_name = name.rpartition("/")
        return f"{ARCHIVED}{folder + '/' if folder else ''}{_stamp(self._clock())}_{file_name}"

    def mark_processed(self, object_name: str, metadata: dict = None) -> str:
        """The document was ingested: processed/<name>. A previous version there goes to archived/."""
        return self._rename(object_name, PROCESSED + source_name(object_name), metadata)

    def mark_failed(self, object_name: str, metadata: dict = None) -> str:
        """The document was rejected or kept failing: failed/<name>, with the reason as metadata."""
        return self._rename(object_name, FAILED + source_name(object_name), metadata)

    def mark_duplicate(self, object_name: str, metadata: dict = None) -> str:
        """An upload with the same content as the ingested version: archived/<time>_<name>."""
        return self._rename(object_name, self._archive_name(object_name), metadata)

    def upload_bytes(self, object_name: str, data: bytes, content_type: str = "application/octet-stream",
                     cache_control: str = None) -> str:
        """Stores bytes at object_name (used for the images of a document), replacing what is there."""
        blob = self.bucket.blob(object_name)
        if cache_control:
            blob.cache_control = cache_control
        blob.upload_from_string(data, content_type=content_type)
        return object_name

    def delete_prefix(self, prefix: str, keep=()) -> int:
        """Deletes every object under prefix except the names in `keep`. Returns how many were deleted."""
        keep = set(keep)
        deleted = 0
        for blob in list(self.bucket.list_blobs(prefix=prefix)):
            if blob.name not in keep:
                blob.delete()
                deleted += 1
        return deleted
