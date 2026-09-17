import shutil
from pathlib import Path

import kagglehub

DATASET = "saibhossain/rag-practice"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET}")
    cache_path = Path(kagglehub.dataset_download(DATASET))

    files = sorted(cache_path.glob("*.pdf"))
    print(f"Copying {len(files)} files to {RAW_DIR}")

    for f in files:
        shutil.copy2(f, RAW_DIR / f.name)
        print(f" - {f.name}")

    print(f"\nDone. {len(files)} files in {RAW_DIR}")


if __name__ == "__main__":
    load_dataset()