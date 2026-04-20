def track_video_sort_yolo(input_video, model, output_video, output_json):
    import cv2
    import numpy as np
    from sort import Sort

    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    tracker = Sort(max_age=12, min_hits=2, iou_threshold=0.2)

    all_tracks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.03)[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                score = scores[i]
                detections.append([x1, y1, x2, y2, score])

        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        tracks = tracker.update(dets)

        frame_tracks = []

        for trk in tracks:
            try:
                track_id, bbox = trk
                x1, y1, x2, y2 = bbox.astype(int)
            except:
                x1, y1, x2, y2, track_id = trk.astype(int)

            frame_tracks.append({
                "track_id": int(track_id),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        all_tracks.append(frame_tracks)
        out.write(frame)

    cap.release()
    out.release()

    import json
    with open(output_json, "w") as f:
        json.dump(all_tracks, f)
