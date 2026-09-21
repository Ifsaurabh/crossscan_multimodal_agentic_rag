import json
import time
from pathlib import Path

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from embedding_config import EMBEDDING_VERSION, IMAGE_MODEL_NAME

IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"
METADATA_PATH = IMAGES_DIR / "metadata.json"
EMBEDDINGS_PATH = Path(__file__).parent.parent / "data" / "embeddings" / "image_embeddings.json"
REPORT_PATH = Path(__file__).parent.parent / "data" / "image_embedding_report.json"


def embed_images():
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading image embedding model: {IMAGE_MODEL_NAME}")
    model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"Found {len(metadata)} images to embed")

    records = []
    failed = []
    start_time = time.perf_counter()

    for item in metadata:
        image_path = IMAGES_DIR / item["image_file"]
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                # model.get_image_features() returns the wrong object (raw vision
                # encoder output, not the projected 512-dim embedding) in this
                # transformers version - call the vision encoder + projection
                # layer directly instead, which is what get_image_features is
                # supposed to do internally.
                vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
                embedding = model.visual_projection(vision_outputs.pooler_output)[0]
                embedding = embedding / embedding.norm()
        except Exception as e:
            failed.append({"image_file": item["image_file"], "error": str(e)})
            continue

        records.append({
            "image_file": item["image_file"],
            "source_pdf": item["source_pdf"],
            "page": item["page"],
            "embedding": embedding.tolist(),
        })

    elapsed = time.perf_counter() - start_time

    EMBEDDINGS_PATH.write_text(
        json.dumps({
            "embedding_version": EMBEDDING_VERSION,
            "embedding_dim": len(records[0]["embedding"]) if records else 0,
            "records": records,
        }, indent=2),
        encoding="utf-8",
    )

    report = {
        "embedding_version": EMBEDDING_VERSION,
        "model": IMAGE_MODEL_NAME,
        "images_embedded": len(records),
        "images_failed": len(failed),
        "failed_details": failed,
        "latency_seconds": round(elapsed, 3),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{len(records)} images embedded, {len(failed)} failed, {round(elapsed, 3)}s")
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    embed_images()
