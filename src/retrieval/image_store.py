"""image_store: where the app gets the picture files it shows next to an answer.

The pictures of a document are written to the documents bucket by the ingestion worker, under `images/<image_file>`
(`image_file` is `<domain>/<file stem>/page3_img1.png`, as stored in the `images` table). The app reads them from
there, so a newly ingested document's pictures show without a redeploy.

For development and tests, setting IMAGES_LOCAL_DIR makes it read `<that folder>/<image_file>` instead (no Google
Cloud needed). Credentials in the cloud are the Cloud Run service account (it needs read access to the bucket).
A picture that cannot be fetched is simply not shown (None), never an error for the user.
"""
import logging
import os
from pathlib import Path, PurePosixPath

DEFAULT_BUCKET = "crossscan-docs-upload-here"
IMAGES_PREFIX = "images/"
MAX_IMAGE_BYTES = 15 * 1024 * 1024   # a picture larger than this is not loaded into the page

log = logging.getLogger(__name__)
_client = None


def _safe_name(image_file: str):
    """The image name as a clean relative path, or None for anything that could point outside the images folder."""
    if not image_file or not isinstance(image_file, str):
        return None
    path = PurePosixPath(image_file)
    if path.is_absolute() or ".." in path.parts or "\\" in image_file:
        return None
    return str(path)


def _bucket():
    global _client
    from google.cloud import storage  # imported here so the app and the tests need no credentials until a picture is shown

    if _client is None:
        _client = storage.Client()
    return _client.bucket(os.environ.get("DOCS_BUCKET", DEFAULT_BUCKET))


def fetch_image(image_file: str, client_bucket=None):
    """The picture's bytes, or None when it is not there, is too large, or cannot be read."""
    name = _safe_name(image_file)
    if name is None:
        return None
    try:
        local = os.environ.get("IMAGES_LOCAL_DIR")
        if local:
            path = Path(local) / name
            if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
                return None
            return path.read_bytes()
        blob = (client_bucket or _bucket()).get_blob(IMAGES_PREFIX + name)
        if blob is None or (blob.size or 0) > MAX_IMAGE_BYTES:
            return None
        return blob.download_as_bytes()
    except Exception as e:  # a missing credential, a network error: the answer is still shown, only the picture is not
        log.warning("could not fetch image %s (%s: %s)", image_file, type(e).__name__, str(e)[:200])
        return None
