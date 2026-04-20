import cv2
import json
import numpy as np

def track_video_opticalflow_yolo(input_video, model, output_video, json_output):

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = None

    lk_params = dict(
        winSize=(15,15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
    )

    prev_gray = None
    prev_pts = None

    all_frames = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()

        frame_predictions = []

        centers = []

        for x1,y1,x2,y2 in boxes:

            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            centers.append([cx,cy])

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)

            frame_predictions.append({
                "bbox":[x1,y1,x2,y2]
            })

        if len(centers) > 0:
            prev_pts = np.array(centers,dtype=np.float32)

        prev_gray = gray

        all_frames.append(frame_predictions)

        out.write(frame)

    cap.release()
    out.release()

    with open(json_output,"w") as f:
        json.dump(all_frames,f,indent=2)

    return output_video, json_output
