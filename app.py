import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Student Behavior Detection")

st.title("🎓 Student Behavior Detection")

# Load model
model = YOLO("best.pt")

uploaded_file = st.file_uploader(
    "Upload a classroom video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:

    # Save uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = "output.mp4"

    writer = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    progress = st.progress(0)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        results = model(frame)

        annotated = results[0].plot()

        writer.write(annotated)

        current += 1
        progress.progress(current / total)

    cap.release()
    writer.release()

    st.success("Finished!")

    st.video(output_file)
