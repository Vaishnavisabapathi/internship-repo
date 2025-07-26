from typing import Dict, List
from PIL import Image
import fitz  # PyMuPDF

processed_pages: Dict[str, List[Image.Image]] = {}

async def handle_file_upload(file):
    pdf_data = await file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")

    pages = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)

    # ✅ Extract filename from file object
    original_filename = file.filename
    clean_filename = original_filename.strip().replace(" ", "_").lower()

    processed_pages[clean_filename] = pages
    print("Stored filename in processed_pages:", clean_filename)

    return {
        "message": f"{clean_filename} uploaded successfully!",
        "page_count": len(pages)
    }
