from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


@dataclass(frozen=True)
class TemporalParameters:
    """Parameters for single-instrument Viterbi candidate selection."""

    top_k: int = 5
    confidence_weight: float = 1.0
    overlap_weight: float = 2.0
    motion_weight: float = 0.25
    scale_weight: float = 0.1


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=2)
    left_area = np.prod(np.maximum(0.0, left[:, 2:] - left[:, :2]), axis=1)[:, None]
    right_area = np.prod(np.maximum(0.0, right[:, 2:] - right[:, :2]), axis=1)[None, :]
    return intersection / np.maximum(left_area + right_area - intersection, 1e-12)


def _transition_score(left: np.ndarray, right: np.ndarray, frame_gap: int, params: TemporalParameters) -> np.ndarray:
    gap = max(1, int(frame_gap))
    overlap = _pairwise_iou(left, right)
    left_centers = (left[:, :2] + left[:, 2:]) / 2
    right_centers = (right[:, :2] + right[:, 2:]) / 2
    left_size = np.maximum(left[:, 2:] - left[:, :2], 1e-6)
    right_size = np.maximum(right[:, 2:] - right[:, :2], 1e-6)
    normalizer = np.sqrt(np.prod(left_size, axis=1))[:, None]
    motion = np.linalg.norm(left_centers[:, None, :] - right_centers[None, :, :], axis=2)
    motion = motion / np.maximum(normalizer * gap, 1e-6)
    left_area = np.prod(left_size, axis=1)[:, None]
    right_area = np.prod(right_size, axis=1)[None, :]
    scale_change = np.abs(np.log(np.maximum(right_area, 1e-6) / np.maximum(left_area, 1e-6))) / gap
    temporal_decay = 1 / np.sqrt(gap)
    return temporal_decay * params.overlap_weight * overlap - params.motion_weight * motion - params.scale_weight * scale_change


def select_temporal_path(candidates: pd.DataFrame, params: TemporalParameters) -> pd.DataFrame:
    """Select one candidate per frame with sequence-level dynamic programming.

    The input may contain multiple segments. Required columns are ``segment_id``,
    ``frame_idx``, ``score`` and xyxy box coordinates. The method uses no labels.
    """
    required = {"segment_id", "frame_idx", "score", *BOX_COLUMNS}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    if candidates.empty:
        return candidates.assign(temporal_rank=pd.Series(dtype=int))

    selected_groups = []
    for _, segment in candidates.groupby("segment_id", sort=False):
        frames = []
        for frame_idx, frame in segment.groupby("frame_idx", sort=True):
            ranked = frame.sort_values("score", ascending=False).head(params.top_k).copy()
            if not ranked.empty:
                frames.append((int(frame_idx), ranked))
        if not frames:
            continue

        backpointers: list[np.ndarray] = []
        first_scores = np.log(np.clip(frames[0][1]["score"].to_numpy(float), 1e-8, 1.0))
        accumulated = params.confidence_weight * first_scores
        for frame_position in range(1, len(frames)):
            previous_index, previous = frames[frame_position - 1]
            current_index, current = frames[frame_position]
            transitions = _transition_score(
                previous[BOX_COLUMNS].to_numpy(float),
                current[BOX_COLUMNS].to_numpy(float),
                current_index - previous_index,
                params,
            )
            joint = accumulated[:, None] + transitions
            parents = np.argmax(joint, axis=0)
            unary = np.log(np.clip(current["score"].to_numpy(float), 1e-8, 1.0))
            accumulated = joint[parents, np.arange(len(current))] + params.confidence_weight * unary
            backpointers.append(parents)

        choices = [int(np.argmax(accumulated))]
        for parents in reversed(backpointers):
            choices.append(int(parents[choices[-1]]))
        choices.reverse()
        for (_, frame), choice in zip(frames, choices):
            row = frame.iloc[[choice]].copy()
            row["temporal_rank"] = 1
            selected_groups.append(row)
    if not selected_groups:
        return candidates.iloc[0:0].assign(temporal_rank=pd.Series(dtype=int))
    return pd.concat(selected_groups, ignore_index=True).sort_values(["segment_id", "frame_idx"]).reset_index(drop=True)


def box_iou_rows(data: pd.DataFrame, prefix: str = "gt_") -> np.ndarray:
    prediction = data[BOX_COLUMNS].to_numpy(float)
    target = data[[f"{prefix}{column}" for column in BOX_COLUMNS]].to_numpy(float)
    top_left = np.maximum(prediction[:, :2], target[:, :2])
    bottom_right = np.minimum(prediction[:, 2:], target[:, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    prediction_area = np.prod(np.maximum(0.0, prediction[:, 2:] - prediction[:, :2]), axis=1)
    target_area = np.prod(np.maximum(0.0, target[:, 2:] - target[:, :2]), axis=1)
    return intersection / np.maximum(prediction_area + target_area - intersection, 1e-12)


def threshold_sweep(selected: pd.DataFrame, iou_threshold: float = 0.5) -> list[dict]:
    iou = box_iou_rows(selected)
    result = []
    for threshold in np.linspace(0.01, 0.99, 99):
        result.append(metrics_at_threshold(selected, float(threshold), iou_threshold, iou))
    return result


def metrics_at_threshold(
    selected: pd.DataFrame,
    threshold: float,
    iou_threshold: float = 0.5,
    iou: np.ndarray | None = None,
) -> dict:
    if iou is None:
        iou = box_iou_rows(selected)
    predicted = selected["score"].to_numpy(float) >= threshold
    matched = predicted & (iou >= iou_threshold)
    tp = int(matched.sum())
    fp = int((predicted & ~matched).sum())
    fn = int((~matched).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def tune_temporal_parameters(candidates: pd.DataFrame) -> tuple[TemporalParameters, float, pd.DataFrame]:
    """Tune only on validation labels and return every tried configuration."""
    rows = []
    best_key: tuple[float, float, float] | None = None
    best_params = TemporalParameters()
    best_threshold = 0.5
    grid = product((3, 5, 10), (0.5, 1.0), (0.5, 1.0, 2.0, 4.0), (0.0, 0.15, 0.3), (0.0, 0.1))
    for top_k, confidence, overlap, motion, scale in grid:
        params = TemporalParameters(top_k, confidence, overlap, motion, scale)
        selected = select_temporal_path(candidates, params)
        iou = box_iou_rows(selected)
        operating_point = max(threshold_sweep(selected), key=lambda row: (row["f1"], row["precision"]))
        mean_iou = float(iou.mean())
        key = (float(operating_point["f1"]), mean_iou, float(operating_point["precision"]))
        rows.append({**params.__dict__, **operating_point, "mean_iou": mean_iou, "iou50_rate": float((iou >= 0.5).mean()), "iou75_rate": float((iou >= 0.75).mean())})
        if best_key is None or key > best_key:
            best_key, best_params = key, params
            best_threshold = float(operating_point["threshold"])
    return best_params, best_threshold, pd.DataFrame(rows).sort_values(["f1", "mean_iou"], ascending=False).reset_index(drop=True)
