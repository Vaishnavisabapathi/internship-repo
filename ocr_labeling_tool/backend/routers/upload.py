from fastapi import APIRouter, File, UploadFile
from services.upload_file import handle_file_upload

router = APIRouter(prefix="/api/upload")

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file)
