from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from .temporal import (
    TemporalParameters,
    box_iou_rows,
    metrics_at_threshold,
    select_temporal_path,
    threshold_sweep,
    tune_temporal_parameters,
)
from .utils import write_json
from .data import parse_video_identity


COCO_STAT_NAMES = (
    "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large",
)


def coco_stats_to_metrics(stats: list[float] | tuple[float, ...]) -> dict[str, float]:
    if len(stats) != len(COCO_STAT_NAMES):
        raise ValueError(f"Expected {len(COCO_STAT_NAMES)} COCO statistics, received {len(stats)}")
    return {name: float(value) for name, value in zip(COCO_STAT_NAMES, stats)}


def write_best_log_metrics(log_path: Path, output_path: Path) -> tuple[Path, dict]:
    """Convert the best validation epoch in an official D-FINE log to registry format."""
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No epochs in {log_path}")
    best = max(rows, key=lambda row: (row["test_coco_eval_bbox"][0], row["test_coco_eval_bbox"][2]))
    metrics = coco_stats_to_metrics(best["test_coco_eval_bbox"])
    metrics.update({"epoch": int(best["epoch"]), "selection_split": "val", "selection_metric": "map"})
    write_json(output_path, metrics)
    return output_path, metrics


class _InferenceImages(Dataset):
    def __init__(self, annotation_path: Path, dataset_root: Path, image_size: int) -> None:
        self.payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.images = self.payload["images"]
        self.dataset_root = dataset_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        info = self.images[index]
        image = Image.open(self.dataset_root / info["file_name"]).convert("RGB")
        tensor = pil_to_tensor(image).float() / 255
        tensor = functional.interpolate(
            tensor.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        return tensor, info


def _collate_images(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    return torch.stack([row[0] for row in batch]), [row[1] for row in batch]


def _load_dfine(dfine_root: Path, config_path: Path, checkpoint: Path, device: torch.device):
    root = str(dfine_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path), resume=str(checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(weights)
    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()
    return model, postprocessor


@torch.inference_mode()
def extract_dfine_candidates(
    config: dict,
    dfine_config: Path,
    checkpoint: Path,
    split: str,
    output: Path,
    image_size: int = 640,
    top_k: int = 20,
) -> tuple[Path, dict]:
    """Run D-FINE on a labeled split and retain candidates for temporal analysis."""
    dataset_root = config["artifacts_dir"] / "dataset"
    annotation_path = dataset_root / f"annotations_{split}.json"
    dataset = _InferenceImages(annotation_path, dataset_root, image_size)
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in dataset.payload["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config["model"].get("num_workers", 4)),
        collate_fn=_collate_images,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfine_root = config["project_root"] / "third_party" / "D-FINE"
    model, postprocessor = _load_dfine(dfine_root, dfine_config, checkpoint, device)
    rows, inference_seconds = [], 0.0
    for images, infos in loader:
        images = images.to(device, non_blocking=True)
        sizes = torch.tensor([[row["width"], row["height"]] for row in infos], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = model(images)
        labels, boxes, scores = postprocessor(outputs, sizes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        for info, image_labels, image_boxes, image_scores in zip(infos, labels, boxes, scores):
            image_id = int(info["id"])
            ground_truth = annotations[image_id][0]["bbox"]
            gt_x1, gt_y1, gt_width, gt_height = map(float, ground_truth)
            order = torch.argsort(image_scores, descending=True)[:top_k]
            for rank, candidate_index in enumerate(order.tolist(), start=1):
                box = image_boxes[candidate_index].detach().cpu().tolist()
                rows.append({
                    "split": split, "image_id": image_id, "segment_id": info["segment_id"],
                    "frame_idx": int(info["frame_idx"]), "candidate_rank": rank,
                    "class_id": int(image_labels[candidate_index].item()),
                    "score": float(image_scores[candidate_index].item()),
                    "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                    "gt_x1": gt_x1, "gt_y1": gt_y1,
                    "gt_x2": gt_x1 + gt_width, "gt_y2": gt_y1 + gt_height,
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    metadata = {
        "split": split, "images": len(dataset), "candidates": len(rows),
        "checkpoint": str(checkpoint), "image_size": image_size,
        "inference_seconds": inference_seconds,
        "model_fps_excluding_io": len(dataset) / inference_seconds if inference_seconds else 0.0,
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return output, metadata


def _selection_summary(selected: pd.DataFrame, threshold: float) -> dict:
    iou = box_iou_rows(selected)
    return {
        "frames": len(selected), "mean_iou": float(iou.mean()),
        "iou50_rate": float((iou >= 0.5).mean()), "iou75_rate": float((iou >= 0.75).mean()),
        "iou50": metrics_at_threshold(selected, threshold, 0.5, iou),
        "iou75": metrics_at_threshold(selected, threshold, 0.75, iou),
    }


def evaluate_temporal_candidates(validation_path: Path, test_path: Path, output_dir: Path) -> tuple[Path, dict]:
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    parameters, threshold, trials = tune_temporal_parameters(validation)
    baseline_parameters = TemporalParameters(top_k=1, overlap_weight=0.0, motion_weight=0.0, scale_weight=0.0)
    selected_validation = select_temporal_path(validation, parameters)
    selected_test = select_temporal_path(test, parameters)
    baseline_validation = select_temporal_path(validation, baseline_parameters)
    baseline_threshold = max(threshold_sweep(baseline_validation), key=lambda row: (row["f1"], row["precision"]))["threshold"]
    baseline_test = select_temporal_path(test, baseline_parameters)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_dir / "validation_grid.csv", index=False)
    selected_validation.to_csv(output_dir / "selected_val.csv", index=False)
    selected_test.to_csv(output_dir / "selected_test.csv", index=False)
    summary = {
        "selection_rule": "all hyperparameters and confidence thresholds selected on validation only",
        "parameters": parameters.__dict__, "threshold": threshold,
        "validation": _selection_summary(selected_validation, threshold),
        "test": _selection_summary(selected_test, threshold),
        "framewise_baseline_threshold": baseline_threshold,
        "framewise_validation": _selection_summary(baseline_validation, baseline_threshold),
        "framewise_test": _selection_summary(baseline_test, baseline_threshold),
    }
    path = output_dir / "summary.json"
    write_json(path, summary)
    return path, summary


@torch.inference_mode()
def infer_dfine_video(
    config: dict,
    dfine_config: Path,
    checkpoint: Path,
    video_path: Path,
    output_dir: Path,
    temporal_parameters: TemporalParameters,
    confidence_threshold: float,
    image_size: int = 640,
    top_k: int = 10,
) -> tuple[Path, dict]:
    """Run batched D-FINE inference and produce a single temporally coherent track."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    case_id, segment_id = parse_video_identity(video_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfine_root = config["project_root"] / "third_party" / "D-FINE"
    model, postprocessor = _load_dfine(dfine_root, dfine_config, checkpoint, device)
    batch_size = int(config.get("inference", {}).get("batch_size", 8))
    candidate_rows, frame_index, inference_seconds = [], 0, 0.0
    while True:
        frames, indices = [], []
        while len(frames) < batch_size:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
            indices.append(frame_index)
            frame_index += 1
        if not frames:
            break
        images = torch.stack(frames).to(device=device, dtype=torch.float32, non_blocking=True) / 255
        images = functional.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
        sizes = torch.tensor([[width, height]] * len(frames), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = model(images)
        labels, boxes, scores = postprocessor(outputs, sizes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        for current_index, image_labels, image_boxes, image_scores in zip(indices, labels, boxes, scores):
            order = torch.argsort(image_scores, descending=True)[:top_k]
            for rank, candidate_index in enumerate(order.tolist(), start=1):
                box = image_boxes[candidate_index].detach().cpu().tolist()
                candidate_rows.append({
                    "case_id": case_id, "segment_id": segment_id, "video_id": segment_id,
                    "frame_idx": current_index, "time_s": current_index / fps,
                    "candidate_rank": rank, "class_id": int(image_labels[candidate_index].item()),
                    "class_name": config["class_names"][int(image_labels[candidate_index].item())],
                    "score": float(image_scores[candidate_index].item()),
                    "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                    "frame_width": width, "frame_height": height,
                })
    cap.release()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(candidate_rows)
    candidates_path = output_dir / "candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    selected = select_temporal_path(candidates, temporal_parameters)
    selected_by_frame = selected.set_index("frame_idx")
    track_rows = []
    for current_index in range(frame_index):
        row = selected_by_frame.loc[current_index]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        detected = float(row.score) >= confidence_threshold
        track_rows.append({
            "case_id": case_id, "segment_id": segment_id, "video_id": segment_id,
            "frame_idx": current_index, "time_s": current_index / fps,
            "track_id": 1 if detected else np.nan,
            "class_id": int(row.class_id) if detected else np.nan,
            "class_name": row.class_name if detected else np.nan,
            "confidence": float(row.score) if detected else np.nan,
            "x1": float(row.x1) if detected else np.nan, "y1": float(row.y1) if detected else np.nan,
            "x2": float(row.x2) if detected else np.nan, "y2": float(row.y2) if detected else np.nan,
            "frame_width": width, "frame_height": height,
        })
    tracks_path = output_dir / "tracks.csv"
    pd.DataFrame(track_rows).to_csv(tracks_path, index=False)
    metadata = {
        "case_id": case_id, "segment_id": segment_id, "frames": frame_index, "fps": fps,
        "checkpoint": str(checkpoint), "image_size": image_size,
        "confidence_threshold_selected_on_validation": confidence_threshold,
        "temporal_parameters": temporal_parameters.__dict__,
        "inference_seconds": inference_seconds,
        "model_fps_excluding_io": frame_index / inference_seconds if inference_seconds else 0.0,
        "detection_rate": float(pd.DataFrame(track_rows).confidence.notna().mean()),
        "candidates": str(candidates_path), "tracks": str(tracks_path),
    }
    write_json(output_dir / "metadata.json", metadata)
    return tracks_path, metadata
