
import os
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import tempfile

=== Load TrOCR model and processor ===

@st.cache_resource
def load_model():
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to("cuda" if torch.cuda.is_available() else "cpu")
return processor, model

processor, model = load_model()

=== Line segmentation using projection profile ===

def segment_lines_from_image(image: Image.Image):
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  

kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)  

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))  
dilated = cv2.dilate(closed, kernel_dilate, iterations=1)  

projection = np.sum(dilated, axis=1)  
height = dilated.shape[0]  

lines = []  
in_line = False  
start = 0  

for y in range(height):  
    if projection[y] > 0:  
        if not in_line:  
            start = y  
            in_line = True  
    else:  
        if in_line:  
            end = y  
            in_line = False  
            if end - start > 10:  
                lines.append((start, end))  
if in_line:  
    end = height  
    if end - start > 10:  
        lines.append((start, end))  

segmented_lines = []  
for idx, (top, bottom) in enumerate(lines):  
    margin = 5  
    y1 = max(0, top - margin)  
    y2 = min(img.shape[0], bottom + margin)  
    line_img = img[y1:y2, :]  

    max_width = 1000  
    h, w = line_img.shape[:2]  
    if w > max_width:  
        new_w = max_width  
        new_h = int(h * (new_w / w))  
        line_img = cv2.resize(line_img, (new_w, new_h), interpolation=cv2.INTER_AREA)  

    segmented_lines.append(Image.fromarray(line_img))  

return segmented_lines

=== OCR for a single image ===

def run_ocr(image):
pixel_values = processor(images=image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
generated_ids = model.generate(pixel_values, max_length=128)
return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

=== Streamlit App ===

st.set_page_config(page_title="Handwritten OCR Labeling Tool", layout="wide")
st.title("📄 OCR & Labeling Tool")

uploaded_files = st.file_uploader("📤 Upload PDFs or Images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
if "file_index" not in st.session_state:
st.session_state.file_index = 0
if "page_index" not in st.session_state:
st.session_state.page_index = 0
if "labeled_data" not in st.session_state:
st.session_state.labeled_data = []
if "ocr_cache" not in st.session_state:
st.session_state.ocr_cache = {}
if "pdf_images_cache" not in st.session_state:
st.session_state.pdf_images_cache = {}

current_file = uploaded_files[st.session_state.file_index]  
file_name = os.path.splitext(current_file.name)[0]  

if current_file.name not in st.session_state.pdf_images_cache:  
    if current_file.type == "application/pdf":  
        with st.spinner(f"📖 Converting PDF: {current_file.name}"):  
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:  
                tmp_file.write(current_file.read())  
                tmp_pdf_path = tmp_file.name  
            images = convert_from_path(tmp_pdf_path, fmt='png', dpi=300)  
    else:  
        images = [Image.open(current_file).convert("RGB")]  
    st.session_state.pdf_images_cache[current_file.name] = images  

pdf_images = st.session_state.pdf_images_cache[current_file.name]  
total_pages = len(pdf_images)  
current_page = st.session_state.page_index  

st.markdown(f"### 📄 File {st.session_state.file_index + 1} of {len(uploaded_files)} — Page {current_page + 1} of {total_pages}")  
page_img = pdf_images[current_page]  

line_images = segment_lines_from_image(page_img)  
st.info(f"✂️ {len(line_images)} lines found on Page {current_page + 1}")  

for line_num, line_img in enumerate(line_images, 1):  
    line_id = f"{file_name}_page{current_page + 1:03}_line{line_num:03}"  
    st.image(line_img, caption=line_id, use_container_width=True)  

    if line_id in st.session_state.ocr_cache:  
        extracted_text = st.session_state.ocr_cache[line_id]  
    else:  
        with st.spinner("🔍 Running OCR..."):  
            extracted_text = run_ocr(line_img)  
            st.session_state.ocr_cache[line_id] = extracted_text  

    edited_text = st.text_area(  
        f"✏️ Edit OCR - {line_id}", value=extracted_text, height=100, key=line_id  
    )  

    if not any(d["line_id"] == line_id for d in st.session_state.labeled_data):  
        st.session_state.labeled_data.append({  
            "line_id": line_id,  
            "text": edited_text  
        })  

col1, col2 = st.columns(2)  

if col1.button("➡️ Continue to Next Page"):  
    if st.session_state.page_index + 1 < total_pages:  
        st.session_state.page_index += 1  
        st.rerun()  
    else:  
        if st.session_state.file_index + 1 < len(uploaded_files):  
            st.session_state.file_index += 1  
            st.session_state.page_index = 0  
            st.rerun()  
        else:  
            st.success("✅ All files and pages have been processed!")  

            df = pd.DataFrame(st.session_state.labeled_data)  
            csv_data = df.to_csv(index=False).encode("utf-8")  

            st.download_button("⬇️ Download Labeled CSV", csv_data, file_name="labeled_data.csv", mime="text/csv")  

            if st.button("🔁 Reset & Start Over"):  
                for key in ["file_index", "page_index", "labeled_data", "ocr_cache", "pdf_images_cache"]:  
                    st.session_state.pop(key, None)  
                st.rerun()  

if col2.button("🔁 Reset All"):  
    for key in ["file_index", "page_index", "labeled_data", "ocr_cache", "pdf_images_cache"]:  
        st.session_state.pop(key, None)  
    st.rerun()

else:
st.info("👈 Upload your PDF or image files to begin.")

