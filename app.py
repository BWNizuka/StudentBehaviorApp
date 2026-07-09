import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="Student Behavior Detection", layout="wide")

st.title("🎓 Student Behavior Detection using YOLO")

st.markdown("### Step 1: Upload your trained YOLO model (.pt)")
uploaded_model = st.file_uploader(
    "Upload best.pt",
    type=["pt"]
)

st.markdown("### Step 2: Upload a classroom video")
uploaded_video = st.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov"]
)

if uploaded_model is not None and uploaded_video is not None:

    with st.spinner("Loading model..."):

        # Save uploaded model temporarily
        model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        model_file.write(uploaded_model.read())
        model_file.close()

        # Load YOLO model
        model = YOLO(model_file.name)

    st.success("✅ Model loaded successfully!")

    with st.spinner("Processing video..."):

        # Save uploaded video temporarily
        video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_file.write(uploaded_video.read())
        video_file.close()

        cap = cv2.VideoCapture(video_file.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            fps = 30

        output_path = "output.mp4"

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        frame_count = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            # YOLO prediction
            results = model(frame)

            # Draw detections
            annotated = results[0].plot()

            writer.write(annotated)

            frame_count += 1

            if total_frames > 0:
                progress.progress(frame_count / total_frames)

        cap.release()
        writer.release()

    st.success("🎉 Detection completed!")

    st.video(output_path)

    with open(output_path, "rb") as file:
        st.download_button(
            "📥 Download Processed Video",
            file,
            file_name="student_behavior_detection.mp4",
            mime="video/mp4"
        )

    # Remove temporary files
    os.remove(model_file.name)
    os.remove(video_file.name)
