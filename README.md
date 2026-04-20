# Polyp Detection & Tracking App

An advanced application for **detecting and tracking polyps in medical videos** using **YOLOv8** and multiple tracking methods including **SORT, DeepSORT, KLT, and Optical Flow**.

This project is built with **Streamlit** and allows you to upload a video, select a tracking method, visualize the tracking, and download JSON data containing tracking information.

---

## 🔹 Features

- **Polyp Detection:** Using YOLOv8, load either a pre-trained or custom-trained model.
- **Multiple Tracking Methods:**  
  - SORT (Simple Online and Realtime Tracking)  
  - DeepSORT  
  - KLT (Lucas-Kanade Optical Flow)  
  - Simple Optical Flow
- **Output Formats:**  
  - Tracked video with object IDs displayed  
  - JSON containing bounding boxes and track IDs  
- **User-Friendly Interface:** Streamlit-based web app, no coding required.

---

## 📦 Prerequisites

- Python >= 3.08
- GPU recommended for YOLOv8 inference

### Required Python Packages

All required packages are listed in `requirements.txt`:

```bash
pip install -r requirements.txt

⚡ Installation & Running
Clone the repository
git clone https://github.com/username/polyp-tracking-app.git
cd polyp-tracking-app

Install dependencies
pip install -r requirements.txt

Run the app
streamlit run app.py

Open in browser
Streamlit will display a local URL (e.g., http://localhost:8501).

🖼️ How to Use
Upload your medical video (MP4, AVI, MOV).
Select the desired tracking method (SORT, DeepSORT, KLT, Optical Flow).
Click the Track Video button.
After processing, the tracked video will be displayed and the JSON file is available for download.

📁 Project Structure
polyp-tracking-app/
│
├─ app.py                        # Streamlit interface
├─ tracking_sort_yolo.py          # SORT tracking function
├─ tracking_deepsort_yolo.py     # DeepSORT tracking function
├─ tracking_klt_yolo.py           # KLT tracking function
├─ tracking_opticalflow_yolo.py  # Optical Flow tracking function
├─ sort.py                        # SORT implementation
├─ requirements.txt               # Python dependencies
├─ yolov8n.pt                     # YOLO model (if pre-trained)
└─ dataset.yaml                   # Dataset configuration for YOLO training

📊 YOLOv8 Model Training
To train your own YOLO model:

from train_yolo import train_yolo
train_yolo()

🔧 Technical Notes
FPS and Video Codec:
Videos are read with their original FPS. For writing, mp4v is recommended for cross-platform compatibility.
Track IDs:
All tracking methods assign a unique ID for each polyp.
JSON Output:
Contains per-frame tracking information:
[
  {
    "track_id": 1,
    "bbox": [x1, y1, x2, y2]
  },
  ...
]

📝 License

This project is licensed under the MIT License.
