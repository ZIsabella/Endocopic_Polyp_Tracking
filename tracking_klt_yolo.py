import cv2
import json
import numpy as np

def track_video_klt_yolo(input_video, model, output_video, json_output):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_video}")

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = None
    lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_video, fourcc, 20.0, (w, h))
            if not out.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {output_video}")

        results = model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        frame_predictions = []

        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            frame_predictions.append({"bbox": [int(x1), int(y1), int(x2), int(y2)]})

        all_frames.append(frame_predictions)
        out.write(frame)

    cap.release()
    out.release()

    with open(json_output, "w") as f:
        json.dump(all_frames, f, indent=2)

    return output_video, json_output