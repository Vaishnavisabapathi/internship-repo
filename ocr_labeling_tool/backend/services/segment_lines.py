# backend/services/segment_lines.py

from services.upload_file import processed_pages
from services.ocr import extract_text_with_tr_ocr, pil_to_base64

def segment_all_pages(filename: str, offset: int = 0, limit: int = 5):
    clean_filename = filename.strip().replace(" ", "_").lower()
    pages = processed_pages.get(clean_filename)

    print("Requested filename lookup:", clean_filename)

    if pages is None:
        raise ValueError(f"No uploaded pages found for filename: {clean_filename}")

    results = []
    current_line = 0
    start = offset
    end = offset + limit

    for page_index, pil_img in enumerate(pages):
        img_height = pil_img.height
        line_height = 50  # You can tweak this value based on actual line spacing
        max_lines_on_page = img_height // line_height

        for line_num in range(max_lines_on_page):
            if current_line >= end:
                return results
            if current_line >= start:
                top = line_num * line_height
                bottom = (line_num + 1) * line_height
                line_image = pil_img.crop((0, top, pil_img.width, bottom))

                ocr_text = extract_text_with_tr_ocr(line_image)
                line_id = f"{clean_filename.replace('.pdf', '')}_page{page_index+1:03d}_line{line_num+1:03d}"

                results.append({
                    "line_id": line_id,
                    "image": line_image,  # Actual PIL image (not base64)
                    "ocr_text": ocr_text
                })

            current_line += 1

    return results
