import json
from pathlib import Path

from PIL import Image
from transformers import pipeline

IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"
METADATA_PATH = IMAGES_DIR / "metadata.json"
REPORT_PATH = Path(__file__).parent.parent / "data" / "image_guardrail_report.json"

UNSAFE_SCORE_THRESHOLD = 0.5


def run_image_guardrails():
    print("Loading NSFW image classifier (Falconsai/nsfw_image_detection)...")
    classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"Checking {len(metadata)} images...")

    kept = []
    flagged = []

    for item in metadata:
        image_path = IMAGES_DIR / item["image_file"]

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                predictions = classifier(img)
        except Exception as e:
            flagged.append({**item, "reason": f"unreadable_image: {e}"})
            continue

        nsfw_score = next((p["score"] for p in predictions if p["label"] == "nsfw"), 0.0)

        if nsfw_score >= UNSAFE_SCORE_THRESHOLD:
            flagged.append({**item, "reason": "nsfw_flagged", "nsfw_score": round(nsfw_score, 4)})
        else:
            kept.append(item)

    for entry in flagged:
        image_path = IMAGES_DIR / entry["image_file"]
        if image_path.exists():
            image_path.unlink()

    METADATA_PATH.write_text(json.dumps(kept, indent=2), encoding="utf-8")

    report = {
        "images_checked": len(metadata),
        "images_flagged": len(flagged),
        "images_remaining": len(kept),
        "flagged_details": flagged,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{len(flagged)}/{len(metadata)} images flagged and removed.")
    for entry in flagged:
        print(f" - {entry['image_file']} ({entry['source_pdf']}): {entry['reason']}")
    print(f"\n{len(kept)} images remain. Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_image_guardrails()
