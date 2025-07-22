import os
import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import pandas as pd
import io

# Load model and processor
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, model = load_model()

st.title("📝 Handwritten OCR Labeling Tool")

# Upload segmented line images
uploaded_images = st.file_uploader("📁 Upload segmented line images (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)


if uploaded_images:
    labeled_data = []

    for uploaded_file in uploaded_images:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, width=500)

        with st.spinner("Running OCR..."):
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
            generated_ids = model.generate(pixel_values)
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        edited_text = st.text_area(f"✏️ Correct OCR for {uploaded_file.name}", value=extracted_text, height=100)
        labeled_data.append({"filename": uploaded_file.name, "text": edited_text})

    if st.button("✅ Save All Labels"):
        df = pd.DataFrame(labeled_data)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download labeled_data.csv", csv_data, file_name="labeled_data.csv", mime="text/csv")
        st.success("Labels saved and ready to download!")

else:
    st.info("Please upload your segmented line images.")
