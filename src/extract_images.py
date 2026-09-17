import json
from pathlib import Path

import pymupdf

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"


def extract_images():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")

    all_metadata = []

    for pdf_path in pdf_files:
        doc = pymupdf.open(pdf_path)
        pdf_out_dir = IMAGES_DIR / pdf_path.stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]

                image_filename = f"page{page_index + 1}_img{img_index + 1}.{ext}"
                image_path = pdf_out_dir / image_filename
                image_path.write_bytes(image_bytes)

                all_metadata.append({
                    "source_pdf": pdf_path.name,
                    "page": page_index + 1,
                    "image_file": str(image_path.relative_to(IMAGES_DIR)),
                })
                count += 1

        doc.close()
        print(f"Extracted {count} images: {pdf_path.name}")

    metadata_path = IMAGES_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")
    print(f"Total images: {len(all_metadata)} -> metadata saved to {metadata_path}")


if __name__ == "__main__":
    extract_images()
