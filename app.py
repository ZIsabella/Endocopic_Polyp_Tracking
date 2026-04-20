import streamlit as st
import os
import tempfile
import cv2
from ultralytics import YOLO

MODEL_PATH = r"D:\Endocopic_polyp_tracking\yolov8n.pt"

from tracking_sort_yolo import track_video_sort_yolo
from tracking_klt_yolo import track_video_klt_yolo
from tracking_deepsort_yolo import track_video_deepsort_yolo
from tracking_opticalflow_yolo import track_video_opticalflow_yolo

st.title("Polyp Detection & Tracking App")

uploaded_file = st.file_uploader(
    "Upload a video (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov", "mpeg4"]
)

tracking_method = st.selectbox(
    "Select tracking method",
    ["SORT", "KLT", "DeepSORT", "Optical Flow"]
)

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 20.0

    temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_mp4, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()

    output_video_path = os.path.join(tempfile.gettempdir(), "tracked_video.mp4")
    output_json_path  = os.path.join(tempfile.gettempdir(), "tracked_data.json")

    try:
        yolo_model = YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        st.stop()

    st.info("🔄 Tracking in progress... please wait.")

    try:
        if tracking_method == "SORT":
            track_video_sort_yolo(temp_mp4, yolo_model, output_video_path, output_json_path)
        elif tracking_method == "KLT":
            track_video_klt_yolo(temp_mp4, yolo_model, output_video_path, output_json_path)
        elif tracking_method == "DeepSORT":
            track_video_deepsort_yolo(temp_mp4, yolo_model, output_video_path, output_json_path)
        elif tracking_method == "Optical Flow":
            track_video_opticalflow_yolo(temp_mp4, yolo_model, output_video_path, output_json_path)
    except Exception as e:
        st.error(f"❌ Tracking failed: {e}")
        st.stop()

    st.success("✅ Tracking Finished!")

    if os.path.exists(output_json_path):
        st.write(f"📄 JSON saved at: {output_json_path}")
        with open(output_json_path, "rb") as f:
            st.download_button("Download JSON", f, "tracked_data.json")
    else:
        st.error("❌ JSON file not created.")

    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        st.success("🎬 Showing tracked video")
        with open(output_video_path, "rb") as f:
            st.video(f.read())
    else:
        st.error("❌ Tracked video not created.")