# backend/routers/lines.py

from fastapi import APIRouter, HTTPException, Query
from services.segment_lines import segment_all_pages
import traceback

router = APIRouter(prefix="/api/lines", tags=["Lines"])

@router.get("/")
async def get_lines(filename: str = Query(...), offset: int = Query(0)):
    try:
        # Call segmentation with offset and limit 5
        lines_batch = segment_all_pages(filename, offset=offset, limit=5)

        return {
            "filename": filename,
            "offset": offset,
            "lines": lines_batch
        }

    except Exception as e:
        traceback.print_exc()  # Logs full traceback to terminal
        raise HTTPException(status_code=500, detail=str(e))
