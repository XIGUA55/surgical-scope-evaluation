from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .temporal import BOX_COLUMNS, _pairwise_iou, box_iou_rows, metrics_at_threshold, threshold_sweep
from .utils import write_json


def candidate_features(candidates: pd.DataFrame, frame_width: float, frame_height: float) -> pd.DataFrame:
    """Label-free candidate features, including bidirectional local track support."""
    boxes = candidates[BOX_COLUMNS].to_numpy(float)
    width = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6) / frame_width
    height = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6) / frame_height
    score = np.clip(candidates.score.to_numpy(float), 1e-6, 1 - 1e-6)
    result = pd.DataFrame(index=candidates.index, data={
        "score": score,
        "score_logit": np.log(score / (1 - score)),
        "rank_reciprocal": 1 / candidates.candidate_rank.to_numpy(float),
        "center_x": (boxes[:, 0] + boxes[:, 2]) / (2 * frame_width),
        "center_y": (boxes[:, 1] + boxes[:, 3]) / (2 * frame_height),
        "box_width": width, "box_height": height,
        "log_area": np.log(width * height + 1e-8),
        "log_aspect": np.log(width / height),
        "previous_support": 0.0, "next_support": 0.0,
        "previous_overlap": 0.0, "next_overlap": 0.0,
    })
    grouped = [(int(frame), group.index.to_numpy()) for frame, group in candidates.groupby("frame_idx", sort=True)]
    for position, (frame, indices) in enumerate(grouped):
        for neighbor_position, support_name, overlap_name in (
            (position - 1, "previous_support", "previous_overlap"),
            (position + 1, "next_support", "next_overlap"),
        ):
            if neighbor_position < 0 or neighbor_position >= len(grouped):
                continue
            neighbor_frame, neighbor_indices = grouped[neighbor_position]
            gap = abs(neighbor_frame - frame)
            if gap > 5:
                continue
            overlap = _pairwise_iou(boxes[indices], boxes[neighbor_indices])
            neighbor_scores = score[neighbor_indices]
            result.loc[indices, overlap_name] = overlap.max(axis=1) / np.sqrt(max(1, gap))
            result.loc[indices, support_name] = (overlap * neighbor_scores[None, :]).max(axis=1) / np.sqrt(max(1, gap))
    return result


def _select_by_prediction(candidates: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked["calibrated_iou"] = prediction
    indices = ranked.groupby(["segment_id", "frame_idx"], sort=False).calibrated_iou.idxmax()
    return ranked.loc[indices].sort_values(["segment_id", "frame_idx"]).reset_index(drop=True)


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
    )


def cross_validated_rerank(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    frame_width: float = 1920,
    frame_height: float = 1080,
    folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    validation_features = candidate_features(validation, frame_width, frame_height)
    test_features = candidate_features(test, frame_width, frame_height)
    target = box_iou_rows(validation)
    frames = np.array(sorted(validation.frame_idx.unique()))
    out_of_fold = np.zeros(len(validation), dtype=float)
    fold_ids = np.full(len(validation), -1, dtype=int)
    splitter = KFold(n_splits=folds, shuffle=False)
    for fold, (train_frame_positions, holdout_frame_positions) in enumerate(splitter.split(frames)):
        train_frames = set(frames[train_frame_positions].tolist())
        holdout_frames = set(frames[holdout_frame_positions].tolist())
        train_mask = validation.frame_idx.isin(train_frames).to_numpy()
        holdout_mask = validation.frame_idx.isin(holdout_frames).to_numpy()
        estimator = _model().fit(validation_features.loc[train_mask], target[train_mask])
        out_of_fold[holdout_mask] = estimator.predict(validation_features.loc[holdout_mask])
        fold_ids[holdout_mask] = fold
    selected_validation = _select_by_prediction(validation, out_of_fold)
    operating_point = max(threshold_sweep(selected_validation), key=lambda row: (row["f1"], row["precision"]))
    estimator = _model().fit(validation_features, target)
    selected_test = _select_by_prediction(test, estimator.predict(test_features))
    threshold = float(operating_point["threshold"])
    validation_iou = box_iou_rows(selected_validation)
    test_iou = box_iou_rows(selected_test)
    summary = {
        "method": "HistGradientBoosting candidate IoU regression",
        "validation_protocol": f"{folds}-fold contiguous-frame out-of-fold predictions",
        "features_use_labels": False,
        "training_target": "candidate IoU from validation annotations",
        "test_used_for_training_or_selection": False,
        "threshold": threshold,
        "validation": {
            "mean_iou": float(validation_iou.mean()), "iou50_rate": float((validation_iou >= 0.5).mean()),
            "iou75_rate": float((validation_iou >= 0.75).mean()),
            "iou50": metrics_at_threshold(selected_validation, threshold, 0.5, validation_iou),
        },
        "test": {
            "mean_iou": float(test_iou.mean()), "iou50_rate": float((test_iou >= 0.5).mean()),
            "iou75_rate": float((test_iou >= 0.75).mean()),
            "iou50": metrics_at_threshold(selected_test, threshold, 0.5, test_iou),
        },
    }
    return selected_validation, selected_test, threshold, summary


def evaluate_learned_reranker(validation_path: Path, test_path: Path, output_dir: Path) -> tuple[Path, dict]:
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    selected_validation, selected_test, _, summary = cross_validated_rerank(validation, test)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_validation.to_csv(output_dir / "selected_val_oof.csv", index=False)
    selected_test.to_csv(output_dir / "selected_test.csv", index=False)
    path = output_dir / "summary.json"
    write_json(path, summary)
    return path, summary
