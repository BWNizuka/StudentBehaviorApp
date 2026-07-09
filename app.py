import streamlit as st
import tempfile
import os
import glob
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

if uploaded_model and uploaded_video:

    # Save uploaded model
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    model_file.write(uploaded_model.read())
    model_file.close()

    # Save uploaded video
    video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.write(uploaded_video.read())
    video_file.close()

    with st.spinner("Loading model..."):
        model = YOLO(model_file.name)

    progress = st.progress(0)
    status = st.empty()

    status.write("Running detection...")

    # Run YOLO on the whole video
    results = model.predict(
        source=video_file.name,
        save=True,
        conf=0.5,
        project="runs",
        name="streamlit",
        exist_ok=True,
        verbose=False
    )

    progress.progress(100)

    # Find output video
    output_folder = "runs/streamlit"

    video_extensions = ("*.mp4", "*.avi", "*.mov")

    output_video = None

    for ext in video_extensions:
        files = glob.glob(os.path.join(output_folder, ext))
        if files:
            output_video = files[0]
            break

    if output_video:

        st.success("✅ Detection Completed!")

        st.subheader("Detected Video")

        with open(output_video, "rb") as f:
            st.video(f.read())

    else:
        st.error("No output video found.")

    os.remove(model_file.name)
    os.remove(video_file.name)
