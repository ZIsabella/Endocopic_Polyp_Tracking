import cv2
import json
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def track_video_deepsort_yolo(input_video, model, output_video, json_output):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_video}")

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = None
    tracker = DeepSort(max_age=30)
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
        scores = results.boxes.conf.cpu().numpy()
        dets = [(b, s) for b, s in zip(boxes, scores)] if len(boxes) > 0 else []

        tracked_objects = tracker.update_tracks(dets, frame=frame)
        frame_predictions = []

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.to_ltrb()
            tid = obj.track_id
            frame_predictions.append({"id": int(tid), "bbox": [int(x1), int(y1), int(x2), int(y2)]})
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

        all_frames.append(frame_predictions)
        out.write(frame)

    cap.release()
    out.release()

    with open(json_output, "w") as f:
        json.dump(all_frames, f, indent=2)

    return output_video, json_output