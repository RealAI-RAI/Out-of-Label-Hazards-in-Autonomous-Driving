import streamlit as st
import os
import tempfile
import pandas as pd
import cv2
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

# Title
st.title("🚗 Hazard Detection & Driver State Analysis in Autonomous Driving")

# Upload video
uploaded_video = st.file_uploader("Upload a driving video", type=["mp4", "avi", "mov"])

# Load YOLO model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # Use your trained model path if available

# Load BLIP model
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# LSTM model placeholder (add your model loading here)
def load_lstm_model():
    # Example: return torch.load("lstm_model.pth", map_location=torch.device('cpu'))
    return None

# Process frame with YOLO and BLIP
def process_frame(frame, yolo_model, blip_processor, blip_model):
    results = yolo_model(frame)
    annotated = results[0].plot()
    st.image(annotated, caption="YOLO Detection", use_column_width=True)

    # Caption the image with BLIP
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(images=img_pil, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

if uploaded_video is not None:
    st.video(uploaded_video)

    st.markdown("### Processing Video... Please wait ⏳")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    yolo_model = load_yolo()
    blip_processor, blip_model = load_blip()
    lstm_model = load_lstm_model()

    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 100:  # Limit frames for demo
            break

        caption = process_frame(frame, yolo_model, blip_processor, blip_model)
        results.append({
            "Frame_ID": frame_count,
            "Caption": caption,
            "Driver_State_Changed": "N/A"  # Add logic here if LSTM is loaded
        })
        frame_count += 1

    cap.release()
    df = pd.DataFrame(results)
    st.markdown("### Processed Results")
    st.dataframe(df)

    csv_download = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv_download, "results.csv", "text/csv")
