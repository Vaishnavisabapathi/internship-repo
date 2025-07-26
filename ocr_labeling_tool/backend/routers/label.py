from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from services.file_handler import convert_pdf_to_images, convert_image_to_pil

router = APIRouter(prefix="/api")

