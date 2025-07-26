# backend/services/ocr.py

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import io
import base64

# Load the TR-OCR model and processor once globally
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Ensure model is in evaluation mode and uses CPU
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_text_with_tr_ocr(pil_image: Image.Image) -> str:
    """
    Performs OCR using TR-OCR on a PIL image.
    """
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def pil_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL image to base64 string (for sending in API responses).
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
