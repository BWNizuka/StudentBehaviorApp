import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="Student Behavior Detection", layout="wide")

st.title("🎓 Student Behavior Detection")

# Upload model
uploaded_model = st.file_uploader(
    "Upload YOLO model (.pt)",
    type=["pt"]
)

# Upload video
uploaded_video = st.file_uploader(
    "Upload classroom video",
    type=["mp4", "avi", "mov"]
)

if uploaded_model and uploaded_video:

    # Save model
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    model_file.write(uploaded_model.read())
    model_file.close()

    # Load model
    with st.spinner("Loading model..."):
        model = YOLO(model_file.name)

    # Save uploaded video
    video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.write(uploaded_video.read())
    video_file.close()

    cap = cv2.VideoCapture(video_file.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    output_path = "detected_video.mp4"

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    status = st.empty()

    frame = 0

    with st.spinner("Running detection..."):

        while cap.isOpened():

            success, image = cap.read()

            if not success:
                break

            results = model.predict(
                image,
                conf=0.5,
                verbose=False
            )

            annotated = results[0].plot()

            writer.write(annotated)

            frame += 1

            if total_frames > 0:
                progress.progress(frame / total_frames)

            status.text(f"Processing frame {frame}/{total_frames}")

    cap.release()
    writer.release()

    st.success("✅ Detection completed!")

    st.subheader("Detected Video")

    st.video(output_path)

    # Optional download button
    with open(output_path, "rb") as f:
        st.download_button(
            "Download detected video",
            f,
            file_name="detected_video.mp4"
        )

    os.remove(model_file.name)
    os.remove(video_file.name)
