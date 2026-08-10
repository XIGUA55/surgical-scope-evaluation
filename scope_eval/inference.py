from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import parse_video_identity
from .tracking import IoUTracker


def infer_video(config: dict, checkpoint: Path, video_path: Path, render: bool = True, max_frames: int | None = None) -> tuple[Path, dict]:
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint).to(device).eval()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    case_id, segment_id = parse_video_identity(video_path)
    output_dir = config["artifacts_dir"] / "inference" / segment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    rendered_path = output_dir / "detections.mp4"
    if render:
        writer = cv2.VideoWriter(str(rendered_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracker = IoUTracker(float(config["tracking"]["iou_threshold"]), int(config["tracking"]["maximum_age_frames"]))
    threshold = float(config["model"]["confidence_threshold"])
    rows, frame_index, inference_seconds = [], 0, 0.0
    batch_size = max(1, int(config.get("inference", {}).get("batch_size", 1)))
    with torch.inference_mode():
        while True:
            batch_frames, batch_indices = [], []
            while len(batch_frames) < batch_size:
                if max_frames is not None and frame_index >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_index)
                frame_index += 1
            if not batch_frames:
                break
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch_frames]
            inputs = processor(images=rgb_frames, return_tensors="pt").to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            target_sizes = torch.tensor([(height, width)] * len(batch_frames), device=device)
            results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)
            for current_index, frame, result in zip(batch_indices, batch_frames, results):
                boxes = result["boxes"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                labels = result["labels"].detach().cpu().numpy().astype(int)
                maximum = int(config["tracking"].get("maximum_detections_per_frame", 20))
                order = np.argsort(scores)[::-1][:maximum]
                boxes, scores, labels = boxes[order], scores[order], labels[order]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
                track_ids = tracker.update(boxes)
                if len(boxes) == 0:
                    rows.append({
                        "case_id": case_id, "segment_id": segment_id, "video_id": segment_id,
                        "frame_idx": current_index, "time_s": current_index / fps, "track_id": np.nan,
                        "class_id": np.nan, "class_name": np.nan, "confidence": np.nan,
                        "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan,
                        "frame_width": width, "frame_height": height,
                    })
                for box, score, label, track_id in zip(boxes, scores, labels, track_ids):
                    class_name = model.config.id2label.get(label, str(label))
                    rows.append({
                        "case_id": case_id, "segment_id": segment_id, "video_id": segment_id,
                        "frame_idx": current_index, "time_s": current_index / fps, "track_id": track_id,
                        "class_id": label, "class_name": class_name, "confidence": float(score),
                        "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                        "frame_width": width, "frame_height": height,
                    })
                    if writer is not None:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"{class_name} #{track_id} {score:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if writer is not None:
                    writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
    tracks_path = output_dir / "tracks.csv"
    pd.DataFrame(rows).to_csv(tracks_path, index=False)
    metadata = {
        "case_id": case_id, "segment_id": segment_id, "frames": frame_index,
        "fps": fps, "inference_seconds": inference_seconds,
        "model_fps_excluding_io": frame_index / inference_seconds if inference_seconds else 0.0,
        "device": str(device), "rendered_video": str(rendered_path) if render else None,
    }
    from .utils import write_json
    write_json(output_dir / "metadata.json", metadata)
    return tracks_path, metadata
