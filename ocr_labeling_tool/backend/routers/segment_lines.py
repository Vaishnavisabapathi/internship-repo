# backend/routers/lines.py

from fastapi import APIRouter, HTTPException, Query
from services.segment_lines import segment_all_pages
from services.ocr import perform_ocr_on_image, pil_to_base64

router = APIRouter(prefix="/api/lines", tags=["Lines"])

@router.get("/")
async def get_lines(filename: str = Query(...), offset: int = Query(0)):
    try:
        all_lines = segment_all_pages(filename)
        lines_batch = all_lines[offset:offset+5]

        result = []
        for line in lines_batch:
            ocr_text = perform_ocr_on_image(line["image"])
            result.append({
                "line_id": line["line_id"],
                "image_base64": pil_to_base64(line["image"]),
                "ocr_text": ocr_text.strip()
            })

        return {
            "filename": filename,
            "offset": offset,
            "lines": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
