import streamlit as st
import tempfile
import cv2
import imageio.v2 as imageio
import os
from ultralytics import YOLO

st.set_page_config(page_title="Student Behavior Detection", layout="wide")

st.title("🎓 Student Behavior Detection")

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

    with st.spinner("Loading YOLO model..."):
        model = YOLO(model_file.name)

    cap = cv2.VideoCapture(video_file.name)

    if not cap.isOpened():
        st.error("Cannot open video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = "detected_video.mp4"

    writer = imageio.get_writer(
        output_video,
        fps=fps
    )

    progress = st.progress(0)
    status = st.empty()

    frame = 0

    while True:

        ret, image = cap.read()

        if not ret:
            break

        results = model.predict(
            image,
            conf=0.5,
            verbose=False
        )

        annotated = results[0].plot()

        # Convert BGR → RGB
        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        writer.append_data(annotated)

        frame += 1

        if total_frames > 0:
            progress.progress(frame / total_frames)

        status.text(f"Processing frame {frame}/{total_frames}")

    cap.release()
    writer.close()

    st.success("✅ Detection completed!")

    st.subheader("Detected Video")

    with open(output_video, "rb") as f:
        st.video(f.read())

    os.remove(model_file.name)
    os.remove(video_file.name)
