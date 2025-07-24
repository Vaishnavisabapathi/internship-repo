import os
import cv2
import io
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pdf2image import convert_from_bytes
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------- Model Loading ----------------
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, model = load_model()

# ---------------- Line Segmentation ----------------
def segment_lines(image, proj_threshold=15, min_line_height=10):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    projection = np.sum(binary, axis=1)

    lines = []
    in_line = False
    start = 0

    for y in range(binary.shape[0]):
        if projection[y] > proj_threshold:
            if not in_line:
                start = y
                in_line = True
        else:
            if in_line:
                end = y
                in_line = False
                if end - start > min_line_height:
                    lines.append((start, end))
    if in_line:
        end = binary.shape[0]
        if end - start > min_line_height:
            lines.append((start, end))

    line_images = []
    for top, bottom in lines:
        margin = 5
        y1 = max(0, top - margin)
        y2 = min(img.shape[0], bottom + margin)
        cropped = Image.fromarray(img[y1:y2, :])
        line_images.append(cropped)

    return line_images

# ---------------- OCR Function ----------------
@st.cache_data(show_spinner=False)
def run_ocr(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="OCR Labeling", layout="wide")
st.title("📄 Handwritten Line-Level OCR + Labeling Tool")

uploaded_files = st.file_uploader(
    "📁 Upload PDFs or Images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    labeled_data = []
    line_queue = []

    for file in uploaded_files:
        filename = file.name
        if filename.lower().endswith(".pdf"):
            pages = convert_from_bytes(file.read())
            for i, page in enumerate(pages):
                lines = segment_lines(page)
                for j, line in enumerate(lines):
                    line_id = f"{os.path.splitext(filename)[0]}_page{i+1:03}_line{j+1:03}"
                    line_queue.append((line_id, line))
        else:
            image = Image.open(file).convert("RGB")
            lines = segment_lines(image)
            for j, line in enumerate(lines):
                line_id = f"{os.path.splitext(filename)[0]}_line{j+1:03}"
                line_queue.append((line_id, line))

    # Pre-fill OCR for the first 5 lines
    ocr_results = {}
    for idx in range(min(5, len(line_queue))):
        line_id, image = line_queue[idx]
        ocr_results[line_id] = run_ocr(image)

    placeholder = st.empty()

    for idx, (line_id, image) in enumerate(line_queue):
        st.image(image, caption=line_id, width=600)

        # Run OCR for upcoming line if not already
        if idx + 5 < len(line_queue) and line_queue[idx + 5][0] not in ocr_results:
            next_id, next_image = line_queue[idx + 5]
            ocr_results[next_id] = run_ocr(next_image)

        default_text = ocr_results.get(line_id, "")
        corrected_text = st.text_area(f"✏️ Edit Text for {line_id}", value=default_text, height=100)
        labeled_data.append({"line_id": line_id, "text": corrected_text})

    if st.button("✅ Save All Labels"):
        df = pd.DataFrame(labeled_data)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download labeled_data.csv", csv_data, file_name="labeled_data.csv", mime="text/csv")
        st.success("Labels saved successfully!")

else:
    st.info("Please upload some segmented images or PDFs to begin.")
