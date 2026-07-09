import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="Student Behavior Detection", layout="wide")

st.title("🎓 Student Behavior Detection")

st.write("### Step 1: Upload your trained model (.pt)")
uploaded_model = st.file_uploader(
    "Choose your YOLO model",
    type=["pt"]
)

st.write("### Step 2: Upload a classroom video")
uploaded_video = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_model and uploaded_video:

    # Save uploaded model
    model_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    model_temp.write(uploaded_model.read())
    model_temp.close()

    # Load model
    with st.spinner("Loading model..."):
        model = YOLO(model_temp.name)

    st.success("✅ Model loaded!")

    # Save uploaded video
    video_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_temp.write(uploaded_video.read())
    video_temp.close()

    cap = cv2.VideoCapture(video_temp.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)

    # Placeholder for displaying frames
    video_placeholder = st.empty()

    frame_number = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # YOLO inference
        results = model(frame)

        # Draw detections
        annotated_frame = results[0].plot()

        # Display in Streamlit
        video_placeholder.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True
        )

        frame_number += 1

        if total_frames > 0:
            progress_bar.progress(frame_number / total_frames)

    cap.release()

    st.success("🎉 Detection completed!")

    # Delete temporary files
    os.remove(model_temp.name)
    os.remove(video_temp.name)
