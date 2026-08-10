from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import CocoInstrumentDataset, _annotation_path, _target_xyxy, make_collator
from .utils import write_json


def _iou_many(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=float)
    top_left = np.maximum(boxes[:, :2], target[:2])
    bottom_right = np.minimum(boxes[:, 2:], target[2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    box_area = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    target_area = np.prod(np.maximum(0.0, target[2:] - target[:2]))
    return intersection / np.maximum(box_area + target_area - intersection, 1e-12)


@torch.no_grad()
def _predict_split(config: dict, checkpoint: Path, split: str, processor, model, device: torch.device) -> pd.DataFrame:
    annotation_path = _annotation_path(config, split)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_info = {int(row["id"]): row for row in annotation["images"]}
    dataset = CocoInstrumentDataset(annotation_path, config["artifacts_dir"] / "dataset", processor)
    loader = DataLoader(
        dataset, batch_size=int(config.get("evaluation", {}).get("batch_size", 8)),
        shuffle=False, collate_fn=make_collator(processor),
    )
    rows = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.stack([label["orig_size"] for label in batch["labels"]]).to(device)
        predictions = processor.post_process_object_detection(outputs, threshold=0.001, target_sizes=target_sizes)
        for prediction, label in zip(predictions, batch["labels"]):
            image_id = int(label["image_id"].item())
            info = image_info[image_id]
            ground_truth = _target_xyxy(label).numpy()[0]
            boxes = prediction["boxes"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()
            order = np.argsort(scores)[::-1]
            boxes, scores = boxes[order], scores[order]
            ious = _iou_many(boxes, ground_truth)
            top_box = boxes[0] if len(boxes) else np.full(4, np.nan)
            width, height = float(info["width"]), float(info["height"])
            gt_width, gt_height = ground_truth[2] - ground_truth[0], ground_truth[3] - ground_truth[1]
            center_x, center_y = (ground_truth[0] + ground_truth[2]) / 2 / width, (ground_truth[1] + ground_truth[3]) / 2 / height
            rows.append({
                "split": split, "image_id": image_id, "segment_id": info["segment_id"],
                "frame_idx": int(info["frame_idx"]), "image_path": str(config["artifacts_dir"] / "dataset" / info["file_name"]),
                "top_score": float(scores[0]) if len(scores) else 0.0,
                "top_iou": float(ious[0]) if len(ious) else 0.0,
                "best_iou_any": float(ious.max()) if len(ious) else 0.0,
                "predictions_at_025": int(np.sum(scores >= 0.25)),
                "gt_area_fraction": float(gt_width * gt_height / (width * height)),
                "gt_center_x": center_x, "gt_center_y": center_y,
                "gt_x1": float(ground_truth[0]), "gt_y1": float(ground_truth[1]),
                "gt_x2": float(ground_truth[2]), "gt_y2": float(ground_truth[3]),
                "pred_x1": float(top_box[0]), "pred_y1": float(top_box[1]),
                "pred_x2": float(top_box[2]), "pred_y2": float(top_box[3]),
            })
    return pd.DataFrame(rows)


def _threshold_metrics(data: pd.DataFrame, threshold: float, iou_threshold: float = 0.5) -> dict:
    predicted = data.top_score >= threshold
    matched = predicted & (data.top_iou >= iou_threshold)
    tp, fp, fn = int(matched.sum()), int((predicted & ~matched).sum()), int((~matched).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _group_diagnostics(data: pd.DataFrame, threshold: float) -> list[dict]:
    result = []
    for dimension in ("area_bucket", "position_bucket", "time_quartile"):
        for value, group in data.groupby(dimension, observed=True):
            metric = _threshold_metrics(group, threshold)
            result.append({"dimension": dimension, "bucket": str(value), "n": len(group), "mean_top_iou": float(group.top_iou.mean()), **metric})
    return result


def _save_examples(data: pd.DataFrame, output_dir: Path, threshold: float, count: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = data.assign(
        error_priority=np.where(data.top_score >= threshold, 1 - data.top_iou, 1 + threshold - data.top_score)
    ).sort_values("error_priority", ascending=False).head(count)
    for _, row in candidates.iterrows():
        image = cv2.imread(row.image_path)
        if image is None:
            continue
        gt = tuple(map(int, (row.gt_x1, row.gt_y1, row.gt_x2, row.gt_y2)))
        pred = tuple(map(int, (row.pred_x1, row.pred_y1, row.pred_x2, row.pred_y2)))
        cv2.rectangle(image, gt[:2], gt[2:], (0, 255, 0), 3)
        cv2.rectangle(image, pred[:2], pred[2:], (0, 255, 255), 3)
        cv2.putText(image, f"GT green | pred yellow score={row.top_score:.2f} IoU={row.top_iou:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(output_dir / f"{row.segment_id}_{int(row.frame_idx):08d}.jpg"), image)


def analyze_detection_errors(config: dict, checkpoint: Path, validation_split: str = "val", test_split: str = "test") -> tuple[Path, dict]:
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False)
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint).to(device).eval()
    validation = _predict_split(config, checkpoint, validation_split, processor, model, device)
    test = _predict_split(config, checkpoint, test_split, processor, model, device)
    thresholds = [_threshold_metrics(validation, float(value)) for value in np.linspace(0.05, 0.95, 91)]
    selected = max(thresholds, key=lambda row: (row["f1"], row["precision"]))["threshold"]
    area_edges = validation.gt_area_fraction.quantile([0.33, 0.67]).to_numpy()
    for data in (validation, test):
        data["area_bucket"] = pd.cut(data.gt_area_fraction, [-np.inf, area_edges[0], area_edges[1], np.inf], labels=["small", "medium", "large"])
        radial = np.hypot(data.gt_center_x - 0.5, data.gt_center_y - 0.5)
        data["position_bucket"] = np.where(radial > 0.4, "edge", "central")
        data["time_quartile"] = pd.qcut(data.frame_idx.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        data["error_type"] = np.select(
            [data.top_score < selected, data.top_iou < 0.5, data.top_iou < 0.75],
            ["missed_low_confidence", "localization_iou_below_050", "localization_iou_050_to_075"],
            default="correct_iou_075",
        )
    output_dir = config["artifacts_dir"] / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image = pd.concat([validation, test], ignore_index=True)
    per_image_path = output_dir / "per_image.csv"
    per_image.to_csv(per_image_path, index=False)
    summary = {
        "checkpoint": str(checkpoint), "threshold_selected_on_validation": selected,
        "validation_top1_iou50": _threshold_metrics(validation, selected, 0.5),
        "validation_top1_iou75": _threshold_metrics(validation, selected, 0.75),
        "test_top1_iou50": _threshold_metrics(test, selected, 0.5),
        "test_top1_iou75": _threshold_metrics(test, selected, 0.75),
        "validation_mean_top_iou": float(validation.top_iou.mean()),
        "test_mean_top_iou": float(test.top_iou.mean()),
        "test_error_counts": test.error_type.value_counts().to_dict(),
        "test_group_diagnostics": _group_diagnostics(test, selected),
    }
    write_json(output_dir / "summary.json", summary)
    pd.DataFrame(thresholds).to_csv(output_dir / "validation_threshold_sweep.csv", index=False)
    _save_examples(test, output_dir / "examples", selected)
    return per_image_path, summary
