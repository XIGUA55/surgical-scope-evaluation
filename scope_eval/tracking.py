from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    age: int = 0


class IoUTracker:
    """Deterministic association for the single-instrument setting."""
    def __init__(self, iou_threshold: float = 0.25, maximum_age: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.maximum_age = maximum_age
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, boxes: np.ndarray) -> list[int]:
        for track in self.tracks:
            track.age += 1
        assigned_tracks, assigned_boxes = set(), set()
        candidates = sorted(
            ((box_iou(track.box, box), ti, bi) for ti, track in enumerate(self.tracks) for bi, box in enumerate(boxes)),
            reverse=True,
        )
        result = [-1] * len(boxes)
        for iou, track_index, box_index in candidates:
            if iou < self.iou_threshold or track_index in assigned_tracks or box_index in assigned_boxes:
                continue
            track = self.tracks[track_index]
            track.box, track.age = boxes[box_index].copy(), 0
            result[box_index] = track.track_id
            assigned_tracks.add(track_index)
            assigned_boxes.add(box_index)
        for index, box in enumerate(boxes):
            if result[index] < 0:
                self.tracks.append(Track(self.next_id, box.copy()))
                result[index] = self.next_id
                self.next_id += 1
        self.tracks = [track for track in self.tracks if track.age <= self.maximum_age]
        return result

