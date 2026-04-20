from ultralytics import YOLO
import cv2
import numpy as np


def yolo_detect(frame, model):
    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # [N,4]
    scores = results.boxes.conf.cpu().numpy()  # [N]

    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        detections.append([x1, y1, x2, y2, score])
    if len(detections) == 0:
        return np.empty((0, 5))
    return np.array(detections)


def yolo_infer(video_path, model_path="runs/detect/train/weights/best.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_detect(frame, model)


        for det in detections:
            x1, y1, x2, y2, conf = det.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
