# backend/routers/segment.py

from fastapi import APIRouter, HTTPException, Query
from services.segment_lines import segment_all_pages

router = APIRouter(prefix="/api/segment", tags=["Segment"])

@router.post("/")
async def segment_lines(
    filename: str = Query(...),
    offset: int = Query(0),
    limit: int = Query(5)
):
    try:
        result = segment_all_pages(filename, offset=offset, limit=limit)
        return {
            "message": f"Segmentation completed for {filename}",
            "total_lines_returned": len(result),
            "line_ids": [r["line_id"] for r in result]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
