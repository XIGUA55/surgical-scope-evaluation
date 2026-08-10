import math

import pandas as pd

from scope_eval.metrics import compute_segment_metrics


SETTINGS = {
    "target_class": "Harmonic0", "confidence_threshold": 0.25,
    "center_radius": 0.2, "maximum_interpolation_gap_seconds": 0.6,
    "smoothing_window_seconds": 0.6,
}


def _row(frame, confidence=0.9, x1=40, y1=40, x2=60, y2=60):
    return {
        "case_id": "VID001", "segment_id": "VID001_B", "frame_idx": frame,
        "time_s": frame / 5, "track_id": 1, "class_name": "Harmonic0",
        "confidence": confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "frame_width": 100, "frame_height": 100,
    }


def test_stationary_centered_track_has_zero_motion():
    result = compute_segment_metrics(pd.DataFrame([_row(i) for i in range(10)]), SETTINGS)
    assert result["detection_rate"] == 1
    assert result["centered_rate_detected"] == 1
    assert abs(result["mean_center_distance"]) < 1e-12
    assert abs(result["mean_speed"]) < 1e-12


def test_single_missing_frame_does_not_crash():
    rows = [_row(0), _row(1, confidence=0, x1=float("nan"), y1=float("nan"), x2=float("nan"), y2=float("nan")), _row(2)]
    result = compute_segment_metrics(pd.DataFrame(rows), SETTINGS)
    assert math.isclose(result["detection_rate"], 2 / 3)
    assert result["detection_gap_count"] == 1


def test_metrics_follow_persistent_track_not_framewise_false_positive():
    rows = [_row(i) for i in range(5)]
    false_positive = _row(2, confidence=0.99, x1=0, y1=0, x2=10, y2=10)
    false_positive["track_id"] = 99
    rows.append(false_positive)
    result = compute_segment_metrics(pd.DataFrame(rows), SETTINGS)
    assert result["centered_rate_detected"] == 1
    assert result["mean_center_distance"] < 1e-12
