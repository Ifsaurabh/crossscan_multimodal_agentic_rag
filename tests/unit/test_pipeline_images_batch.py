"""pipeline: the pictures of a document are embedded in batches."""
from test_pipeline import FakeStorage, build, db, extracted, intake  # noqa: F401  (the same fakes the pipeline tests use)

IMAGES = [{"page": 1, "filename": f"p1_img{i}.png", "ext": "png", "bytes": f"img{i}".encode()} for i in range(1, 4)]


def test_all_the_pictures_of_a_document_go_to_the_batch_embedder_in_one_call(db):
    calls = []
    pipeline, storage = build(db, images=IMAGES, image_batch_embedder=lambda datas: calls.append(list(datas)) or [[0.0, 1.0]] * len(datas))

    report = pipeline("paper.pdf", intake())

    assert calls == [[b"img1", b"img2", b"img3"]] and report.images == 3
    assert sorted(storage.uploads) == [f"images/lung-cancer/paper/p1_img{i}.png" for i in (1, 2, 3)]


def test_a_picture_the_batch_could_not_decode_is_skipped_and_counted(db):
    pipeline, storage = build(db, images=IMAGES, image_batch_embedder=lambda datas: [[0.0, 1.0], None, [0.0, 1.0]])

    report = pipeline("paper.pdf", intake())

    assert report.images == 2 and report.images_skipped == 1
    assert "images/lung-cancer/paper/p1_img2.png" not in storage.uploads


def test_without_a_batch_embedder_each_picture_is_embedded_on_its_own_and_a_bad_one_is_skipped(db):
    def one(data):
        if data == b"img2":
            raise ValueError("not an image")
        return [0.0, 1.0]

    pipeline, _ = build(db, images=IMAGES, image_embedder=one)

    report = pipeline("paper.pdf", intake())

    assert report.images == 2 and report.images_skipped == 1


def test_the_stored_bytes_are_counted(db):
    pipeline, _ = build(db, images=IMAGES, image_batch_embedder=lambda datas: [[0.0, 1.0]] * len(datas))
    assert pipeline("paper.pdf", intake()).image_bytes == 12
