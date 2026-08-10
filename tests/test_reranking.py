import pandas as pd

from scope_eval.reranking import candidate_features


def test_candidate_features_reward_neighbor_overlap():
    rows = []
    for frame in (0, 1):
        rows.extend([
            {"frame_idx": frame, "candidate_rank": 1, "score": 0.8, "x1": 10, "y1": 10, "x2": 30, "y2": 30},
            {"frame_idx": frame, "candidate_rank": 2, "score": 0.7, "x1": 70 - 50 * frame, "y1": 70, "x2": 90 - 50 * frame, "y2": 90},
        ])
    features = candidate_features(pd.DataFrame(rows), 100, 100)
    assert features.loc[0, "next_overlap"] == 1
    assert features.loc[1, "next_overlap"] == 0
