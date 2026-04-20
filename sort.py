# sort.py
import numpy as np

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.track_id = track_id
        self.hits = 1
        self.no_losses = 0
class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.track_id_count = 0

    def update(self, detections):

        assignments = []

        unmatched_tracks = set(range(len(self.trackers)))
        for det in detections:
            assigned = False
            for idx, trk in enumerate(self.trackers):
                if self.iou(det[:4], trk.bbox) > self.iou_threshold:
                    trk.bbox = det[:4]
                    trk.hits += 1
                    trk.no_losses = 0
                    assignments.append((trk.track_id, det[:4]))
                    assigned = True
                    unmatched_tracks.discard(idx)
                    break

            if not assigned:
                self.track_id_count += 1
                new_trk = Track(det[:4], self.track_id_count)
                self.trackers.append(new_trk)
                assignments.append((new_trk.track_id, det[:4]))

        for idx in unmatched_tracks:
            self.trackers[idx].no_losses += 1

        self.trackers = [trk for trk in self.trackers if trk.no_losses <= self.max_age]

        return assignments

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA + 1)
        interH = max(0, yB - yA + 1)
        interArea = interW * interH

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou