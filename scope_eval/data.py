from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .utils import stable_environment, write_json


VIDEO_PATTERN = re.compile(r"^(?P<case>VID\d+)(?:_(?P<segment>[A-Za-z0-9]+))?(?:_\d+fps)?$")
LABEL_PATTERN = re.compile(r"^(?P<case>VID\d+)_(?P<segment>[A-Za-z0-9]+)seg$")


@dataclass
class VideoAudit:
    path: str
    case_id: str
    segment_id: str
    readable: bool
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    duration_s: float | None
    file_size: int
    truncated_atom: str | None = None
    declared_atom_end: int | None = None
    error: str | None = None


def parse_video_identity(path: Path, identity_override: tuple[str, str] | None = None) -> tuple[str, str]:
    if identity_override is not None:
        return identity_override
    match = VIDEO_PATTERN.match(path.stem)
    if not match:
        return path.stem, path.stem
    case_id = match.group("case")
    segment = match.group("segment")
    return case_id, f"{case_id}_{segment}" if segment else case_id


def parse_label_identity(path: Path, identity_override: tuple[str, str] | None = None) -> tuple[str, str]:
    if identity_override is not None:
        return identity_override
    match = LABEL_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized label directory: {path}")
    case_id, segment = match.group("case"), match.group("segment")
    return case_id, f"{case_id}_{segment}"


def inspect_mp4_atoms(path: Path) -> tuple[str | None, int | None]:
    """Return a top-level atom that extends beyond EOF, if present."""
    size = path.stat().st_size
    offset = 0
    with path.open("rb") as handle:
        while offset + 8 <= size:
            handle.seek(offset)
            header = handle.read(16)
            atom_size, atom_type = struct.unpack(">I4s", header[:8])
            header_size = 8
            if atom_size == 1:
                if len(header) < 16:
                    return atom_type.decode("latin1"), offset + 16
                atom_size = struct.unpack(">Q", header[8:16])[0]
                header_size = 16
            elif atom_size == 0:
                atom_size = size - offset
            if atom_size < header_size:
                return atom_type.decode("latin1"), offset + atom_size
            end = offset + atom_size
            if end > size:
                return atom_type.decode("latin1"), end
            offset = end
    return None, None


def probe_video(
    path: Path,
    ffprobe: str = "ffprobe",
    identity_override: tuple[str, str] | None = None,
) -> VideoAudit:
    case_id, segment_id = parse_video_identity(path, identity_override)
    truncated, declared_end = inspect_mp4_atoms(path)
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=stable_environment(), check=True)
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
        fps = numerator / denominator if denominator else 0.0
        cap = cv2.VideoCapture(str(path))
        first_ok, _ = cap.read()
        cap.release()
        if not first_ok:
            raise ValueError("OpenCV could not decode the first frame")
        return VideoAudit(
            str(path), case_id, segment_id, True, int(stream["width"]), int(stream["height"]),
            fps, int(stream.get("nb_frames") or 0), float(payload["format"]["duration"]),
            path.stat().st_size, truncated, declared_end,
        )
    except Exception as exc:
        return VideoAudit(
            str(path), case_id, segment_id, False, None, None, None, None, None,
            path.stat().st_size, truncated, declared_end, str(exc),
        )


def parse_label_file(path: Path, class_names: dict[int, str]) -> list[dict]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        fields = [item.strip() for item in raw.split(",")]
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_number}: expected 6 comma-separated fields")
        class_id, object_id = int(fields[0]), int(fields[1])
        if class_id not in class_names:
            raise ValueError(f"{path}:{line_number}: unknown class {class_id}")
        cx, cy, width, height = map(float, fields[2:])
        values = (cx, cy, width, height)
        if not all(np.isfinite(values)) or not all(0 < value <= 1 for value in values):
            raise ValueError(f"{path}:{line_number}: invalid normalized box {values}")
        # A normalized centre inside the image with a positive box always overlaps
        # the image. Border-crossing boxes are valid annotations and are clipped
        # to image bounds during COCO conversion below.
        rows.append({
            "class_id": class_id, "class_name": class_names[class_id], "object_id": object_id,
            "cx": cx, "cy": cy, "box_width": width, "box_height": height,
        })
    return rows


def discover_videos(config: dict) -> list[Path]:
    paths = sorted(config["video_dir"].glob("*.mp4"))
    if not paths:
        paths = sorted(config["data_dir"].glob("*.mp4"))
    return paths


def _identity_override(config: dict, kind: str, name: str) -> tuple[str, str] | None:
    raw = config.get("identity_overrides", {}).get(kind, {}).get(name)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2 or not all(isinstance(value, str) and value for value in raw):
        raise ValueError(f"identity_overrides.{kind}.{name} must be [case_id, segment_id]")
    return raw[0], raw[1]


def discover_label_directories(config: dict) -> list[Path]:
    directories = set(config["label_dir"].glob("*seg"))
    configured = config.get("identity_overrides", {}).get("labels", {})
    directories.update(config["label_dir"] / name for name in configured)
    return sorted(path for path in directories if path.is_dir())


def audit_dataset(config: dict) -> dict:
    ffprobe = shutil.which("ffprobe") or "/root/miniconda3/envs/surgical-repro/bin/ffprobe"
    videos = [
        probe_video(
            path,
            ffprobe=ffprobe,
            identity_override=_identity_override(config, "videos", path.name),
        )
        for path in discover_videos(config)
    ]
    video_by_segment = {video.segment_id: video for video in videos}
    labels = []
    errors = []
    for directory in discover_label_directories(config):
        try:
            case_id, segment_id = parse_label_identity(
                directory,
                _identity_override(config, "labels", directory.name),
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        frame_files = sorted(directory.glob("*.txt"))
        frame_numbers = []
        row_count = 0
        object_ids: Counter[int] = Counter()
        for label_path in frame_files:
            try:
                frame_idx = int(label_path.stem)
                parsed = parse_label_file(label_path, config["class_names"])
                frame_numbers.append(frame_idx)
                row_count += len(parsed)
                object_ids.update(row["object_id"] for row in parsed)
            except Exception as exc:
                errors.append(str(exc))
        video = video_by_segment.get(segment_id)
        status = "ready" if video and video.readable else "excluded_missing_or_corrupt_video"
        max_frame = max(frame_numbers) if frame_numbers else None
        if video and video.readable and max_frame is not None and max_frame >= (video.frame_count or 0):
            status = "excluded_label_frame_out_of_range"
        labels.append({
            "path": str(directory), "case_id": case_id, "segment_id": segment_id,
            "label_files": len(frame_files), "annotation_rows": row_count,
            "minimum_frame": min(frame_numbers) if frame_numbers else None,
            "maximum_frame": max_frame, "object_ids": dict(object_ids), "status": status,
        })
    result = {
        "videos": [asdict(video) for video in videos],
        "label_sets": labels,
        "errors": errors,
        "summary": {
            "video_count": len(videos),
            "readable_video_count": sum(video.readable for video in videos),
            "label_file_count": sum(row["label_files"] for row in labels),
            "ready_label_file_count": sum(row["label_files"] for row in labels if row["status"] == "ready"),
            "error_count": len(errors),
        },
    }
    output = config["artifacts_dir"] / "audit" / "dataset_audit.json"
    write_json(output, result)
    return result


def _video_map(config: dict, audit: dict) -> dict[str, Path]:
    return {
        row["segment_id"]: Path(row["path"])
        for row in audit["videos"] if row["readable"]
    }


def prepare_dataset(config: dict, jpeg_quality: int = 95) -> tuple[pd.DataFrame, Path]:
    audit = audit_dataset(config)
    if audit["errors"]:
        raise ValueError(f"Dataset audit has {len(audit['errors'])} label errors; see audit report")
    video_map = _video_map(config, audit)
    ready = [row for row in audit["label_sets"] if row["status"] == "ready"]
    output_root = config["artifacts_dir"] / "dataset"
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    images, annotations, manifest = [], [], []
    image_id = 1
    annotation_id = 1
    for label_set in ready:
        case_id, segment_id = label_set["case_id"], label_set["segment_id"]
        label_dir, video_path = Path(label_set["path"]), video_map[segment_id]
        requested = {int(path.stem): path for path in label_dir.glob("*.txt")}
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_dir = image_root / segment_id
        target_dir.mkdir(parents=True, exist_ok=True)
        found = set()
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            label_path = requested.get(frame_idx)
            if label_path is not None:
                image_path = target_dir / f"{frame_idx:08d}.jpg"
                if not image_path.exists():
                    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                relative_image = image_path.relative_to(output_root).as_posix()
                images.append({
                    "id": image_id, "file_name": relative_image, "width": width, "height": height,
                    "case_id": case_id, "segment_id": segment_id, "frame_idx": frame_idx,
                    "time_s": frame_idx / fps,
                })
                parsed_rows = parse_label_file(label_path, config["class_names"])
                for row in parsed_rows:
                    x1 = np.clip(row["cx"] - row["box_width"] / 2, 0, 1) * width
                    y1 = np.clip(row["cy"] - row["box_height"] / 2, 0, 1) * height
                    x2 = np.clip(row["cx"] + row["box_width"] / 2, 0, 1) * width
                    y2 = np.clip(row["cy"] + row["box_height"] / 2, 0, 1) * height
                    x, y, box_width, box_height = x1, y1, x2 - x1, y2 - y1
                    annotations.append({
                        "id": annotation_id, "image_id": image_id, "category_id": row["class_id"],
                        "bbox": [x, y, box_width, box_height], "area": box_width * box_height,
                        "iscrowd": 0, "object_id": row["object_id"],
                    })
                    manifest.append({
                        "case_id": case_id, "segment_id": segment_id, "frame_idx": frame_idx,
                        "time_s": frame_idx / fps, "image_id": image_id,
                        "image_path": str(image_path), "source_label_path": str(label_path),
                        **row, "split": "unassigned",
                    })
                    annotation_id += 1
                image_id += 1
                found.add(frame_idx)
            frame_idx += 1
        cap.release()
        missing = sorted(set(requested) - found)
        if missing:
            raise ValueError(f"{segment_id}: {len(missing)} labelled frames could not be decoded; first={missing[:5]}")
    coco = {
        "info": {"description": "Pancreatic laparoscopic instrument dataset", "version": "1.0"},
        "images": images, "annotations": annotations,
        "categories": [{"id": key, "name": value} for key, value in config["class_names"].items()],
    }
    coco_path = output_root / "annotations_all.json"
    write_json(coco_path, coco)
    for segment_id in sorted({row["segment_id"] for row in images}):
        segment_images = [row for row in images if row["segment_id"] == segment_id]
        segment_image_ids = {row["id"] for row in segment_images}
        segment_coco = {
            **{key: value for key, value in coco.items() if key not in {"images", "annotations"}},
            "images": segment_images,
            "annotations": [row for row in annotations if row["image_id"] in segment_image_ids],
        }
        write_json(output_root / f"annotations_segment_{segment_id}.json", segment_coco)
    manifest_frame = pd.DataFrame(manifest)
    manifest_path = output_root / "manifest.csv"
    manifest_frame.to_csv(manifest_path, index=False)
    return manifest_frame, coco_path


def assign_splits(config: dict, smoke: bool = False, segment_holdout: bool = False) -> pd.DataFrame:
    manifest_path = config["artifacts_dir"] / "dataset" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run prepare before split")
    manifest = pd.read_csv(manifest_path)
    cases = sorted(manifest.case_id.unique())
    minimum = int(config["split"]["minimum_cases"])
    rng = np.random.default_rng(int(config["seed"]))
    if len(cases) < minimum and not smoke and not segment_holdout:
        raise ValueError(f"Formal split requires at least {minimum} independent cases; found {len(cases)}")
    if smoke and segment_holdout:
        raise ValueError("smoke and segment_holdout are mutually exclusive")
    if segment_holdout:
        scheme = config["split"]["segment_holdout"]
        segment_to_split = {
            segment: split
            for split in ("train", "val", "test")
            for segment in scheme[split]
        }
        duplicates = sum(len(scheme[split]) for split in ("train", "val", "test")) - len(segment_to_split)
        if duplicates:
            raise ValueError("A segment appears in more than one holdout split")
        missing_segments = set(manifest.segment_id.unique()) - set(segment_to_split)
        unknown_segments = set(segment_to_split) - set(manifest.segment_id.unique())
        if missing_segments or unknown_segments:
            raise ValueError(f"Segment holdout mismatch: unassigned={sorted(missing_segments)}, unknown={sorted(unknown_segments)}")
        manifest["split"] = manifest.segment_id.map(segment_to_split)
    elif smoke:
        unique_images = manifest[["image_id", "segment_id", "frame_idx"]].drop_duplicates()
        picked = []
        for _, group in unique_images.groupby("segment_id"):
            indices = np.linspace(0, len(group) - 1, min(24, len(group)), dtype=int)
            picked.extend(group.iloc[indices].image_id.tolist())
        rng.shuffle(picked)
        validation_count = max(1, len(picked) // 5)
        split_map = {image_id: ("smoke_val" if i < validation_count else "smoke_train") for i, image_id in enumerate(picked)}
        manifest["split"] = manifest.image_id.map(split_map).fillna("unused")
    else:
        shuffled = list(cases)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * float(config["split"]["test"])))
        n_val = max(1, round(n * float(config["split"]["val"])))
        if n_test + n_val >= n:
            n_val = 1
            n_test = 1
        test_cases, val_cases = set(shuffled[:n_test]), set(shuffled[n_test:n_test + n_val])
        manifest["split"] = manifest.case_id.map(lambda case: "test" if case in test_cases else "val" if case in val_cases else "train")
    manifest.to_csv(manifest_path, index=False)
    _write_split_coco(config, manifest)
    return manifest


def _write_split_coco(config: dict, manifest: pd.DataFrame) -> None:
    root = config["artifacts_dir"] / "dataset"
    all_coco = json.loads((root / "annotations_all.json").read_text(encoding="utf-8"))
    for split in sorted(set(manifest.split) - {"unused", "unassigned"}):
        image_ids = set(map(int, manifest.loc[manifest.split == split, "image_id"].unique()))
        payload = {
            **{key: value for key, value in all_coco.items() if key not in {"images", "annotations"}},
            "images": [row for row in all_coco["images"] if row["id"] in image_ids],
            "annotations": [row for row in all_coco["annotations"] if row["image_id"] in image_ids],
        }
        write_json(root / f"annotations_{split}.json", payload)
