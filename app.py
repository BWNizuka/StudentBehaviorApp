import streamlit as st
import tempfile
import cv2
import os
import subprocess
from ultralytics import YOLO

st.set_page_config(page_title="Student Behavior Detection", layout="wide")

st.title("🎓 Student Behavior Detection using YOLO")

# Upload model
uploaded_model = st.file_uploader(
    "Upload YOLO Model (.pt)",
    type=["pt"]
)

# Upload video
uploaded_video = st.file_uploader(
    "Upload Classroom Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_model is not None and uploaded_video is not None:

    # Save uploaded model
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    model_file.write(uploaded_model.read())
    model_file.close()

    # Save uploaded video
    video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.write(uploaded_video.read())
    video_file.close()

    # Load YOLO model
    with st.spinner("Loading YOLO model..."):
        model = YOLO(model_file.name)

    st.success("Model loaded successfully!")

    cap = cv2.VideoCapture(video_file.name)

    if not cap.isOpened():
        st.error("Cannot open uploaded video.")
        st.stop()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_output = "detected_raw.mp4"
    final_output = "detected_video.mp4"

    writer = cv2.VideoWriter(
        raw_output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        st.error("Failed to create output video.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    frame_number = 0

    with st.spinner("Detecting student behaviors..."):

        while True:

            success, frame = cap.read()

            if not success:
                break

            results = model.predict(
                frame,
                conf=0.5,
                verbose=False
            )

            annotated = results[0].plot()

            writer.write(annotated)

            frame_number += 1

            if total_frames > 0:
                progress.progress(min(frame_number / total_frames, 1.0))

            status.text(f"Processing frame {frame_number}/{total_frames}")

    cap.release()
    writer.release()

    status.text("Converting video to browser-compatible format...")

    try:

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_output,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                final_output,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        st.success("Detection completed!")

        st.subheader("Detected Video")

        with open(final_output, "rb") as f:
            st.video(f.read())

    except subprocess.CalledProcessError as e:

        st.error("FFmpeg failed to convert the video.")
        st.text(e.stderr.decode())

    # Clean temporary uploaded files
    os.remove(model_file.name)
    os.remove(video_file.name)
