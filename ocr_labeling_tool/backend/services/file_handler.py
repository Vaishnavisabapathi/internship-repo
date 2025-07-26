from typing import List
from PIL import Image
import fitz  # PyMuPDF
import io

def convert_pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(image)
    return images

def convert_image_to_pil(file_bytes: bytes) -> List[Image.Image]:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return [image]  # Return as a list to match PDF output format
