import pandas as pd

from scope_eval.temporal import TemporalParameters, select_temporal_path


def test_temporal_path_prefers_consistent_candidate_sequence():
    rows = []
    distractors = [(60, 60, 80, 80), (5, 60, 25, 80), (60, 5, 80, 25)]
    for frame, distractor in enumerate(distractors):
        rows.append({"segment_id": "S", "frame_idx": frame, "score": 0.80, "x1": 10, "y1": 10, "x2": 30, "y2": 30, "candidate": "target"})
        rows.append({"segment_id": "S", "frame_idx": frame, "score": 0.90, "x1": distractor[0], "y1": distractor[1], "x2": distractor[2], "y2": distractor[3], "candidate": "distractor"})
    selected = select_temporal_path(pd.DataFrame(rows), TemporalParameters(overlap_weight=2.0, motion_weight=0.3))
    assert selected.candidate.tolist() == ["target", "target", "target"]


def test_temporal_path_handles_frame_gaps_and_segments():
    rows = [
        {"segment_id": segment, "frame_idx": frame, "score": 0.8, "x1": 10, "y1": 10, "x2": 20, "y2": 20}
        for segment in ("A", "B") for frame in (0, 3)
    ]
    selected = select_temporal_path(pd.DataFrame(rows), TemporalParameters())
    assert len(selected) == 4
    assert selected.groupby("segment_id").frame_idx.apply(list).to_dict() == {"A": [0, 3], "B": [0, 3]}
