import pymupdf



def extract_pdf_images(pdf_path) -> list:
    """The embedded pictures of ONE PDF: [{"page", "filename", "ext", "bytes"}], page 1 first.
    The file name is `page<page>_img<n>.<ext>`."""
    found = []
    doc = pymupdf.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc[page_index].get_images(full=True)):
                base_image = doc.extract_image(img[0])
                found.append({
                    "page": page_index + 1,
                    "filename": f"page{page_index + 1}_img{img_index + 1}.{base_image['ext']}",
                    "ext": base_image["ext"],
                    "bytes": base_image["image"],
                })
    finally:
        doc.close()
    return found


