"""image_store: where the app gets the pictures it shows. A fake bucket and a temporary folder; no network."""
import pytest

from retrieval import image_store as store


class FakeBlob:
    def __init__(self, data, size=None):
        self.data, self.size = data, len(data) if size is None else size

    def download_as_bytes(self):
        return self.data


class FakeBucket:
    def __init__(self, blobs):
        self.blobs, self.asked = blobs, []

    def get_blob(self, name):
        self.asked.append(name)
        return self.blobs.get(name)


@pytest.fixture(autouse=True)
def no_local_dir(monkeypatch):
    monkeypatch.delenv("IMAGES_LOCAL_DIR", raising=False)


def test_a_picture_is_read_from_the_bucket_under_the_images_prefix():
    bucket = FakeBucket({"images/lung-cancer/a/page3_img1.png": FakeBlob(b"PNG")})
    assert store.fetch_image("lung-cancer/a/page3_img1.png", client_bucket=bucket) == b"PNG"
    assert bucket.asked == ["images/lung-cancer/a/page3_img1.png"]


def test_a_picture_that_is_not_in_the_bucket_is_none():
    assert store.fetch_image("x/y/z.png", client_bucket=FakeBucket({})) is None


def test_a_picture_that_is_too_large_is_not_loaded():
    bucket = FakeBucket({"images/a/b.png": FakeBlob(b"x", size=store.MAX_IMAGE_BYTES + 1)})
    assert store.fetch_image("a/b.png", client_bucket=bucket) is None


@pytest.mark.parametrize("name", ["", None, "../secret.png", "a/../../b.png", "/etc/passwd", "a" + chr(92) + "b.png", 5])
def test_a_name_that_could_leave_the_images_folder_is_refused_without_asking_the_bucket(name):
    bucket = FakeBucket({})
    assert store.fetch_image(name, client_bucket=bucket) is None and bucket.asked == []


def test_a_bucket_error_never_reaches_the_user():
    class Broken:
        def get_blob(self, name):
            raise RuntimeError("no credentials")

    assert store.fetch_image("a/b.png", client_bucket=Broken()) is None


def test_a_local_folder_is_used_when_set(monkeypatch, tmp_path):
    (tmp_path / "lung-cancer" / "a").mkdir(parents=True)
    (tmp_path / "lung-cancer" / "a" / "p1.png").write_bytes(b"LOCAL")
    monkeypatch.setenv("IMAGES_LOCAL_DIR", str(tmp_path))

    assert store.fetch_image("lung-cancer/a/p1.png", client_bucket=FakeBucket({})) == b"LOCAL"
    assert store.fetch_image("lung-cancer/a/missing.png") is None
    assert store.fetch_image("../outside.png") is None
