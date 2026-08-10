from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _robust_score(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = max(float(values.std(ddof=0)), 1e-12)
    return (values - median) / scale


def _image_statistics(path: str) -> tuple[float, float, float]:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read extracted frame: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_annotation_quality(manifest: pd.DataFrame) -> pd.DataFrame:
    scored = manifest.copy().sort_values(["segment_id", "frame_idx"]).reset_index(drop=True)
    scored["box_area_norm"] = scored.box_width * scored.box_height
    scored["aspect_ratio"] = scored.box_width / scored.box_height
    scored["left"] = scored.cx - scored.box_width / 2
    scored["top"] = scored.cy - scored.box_height / 2
    scored["right"] = scored.cx + scored.box_width / 2
    scored["bottom"] = scored.cy + scored.box_height / 2
    scored["boundary_margin"] = scored[["left", "top"]].min(axis=1).combine(
        (1 - scored[["right", "bottom"]]).min(axis=1), min
    )

    previous_cx = scored.groupby("segment_id").cx.shift()
    previous_cy = scored.groupby("segment_id").cy.shift()
    delta_t = scored.groupby("segment_id").time_s.diff()
    displacement = np.hypot(scored.cx - previous_cx, scored.cy - previous_cy)
    scored["position_speed"] = displacement / delta_t.clip(lower=0.2)
    scored.loc[delta_t.isna() | (delta_t > 2.0), "position_speed"] = np.nan

    statistics = [_image_statistics(path) for path in scored.image_path]
    scored[["brightness", "contrast", "blur_variance"]] = pd.DataFrame(statistics, index=scored.index)

    scored["area_z"] = scored.groupby("segment_id").box_area_norm.transform(_robust_score)
    scored["aspect_z"] = scored.groupby("segment_id").aspect_ratio.transform(_robust_score)
    scored["speed_z"] = scored.groupby("segment_id").position_speed.transform(_robust_score)
    scored["brightness_z"] = scored.groupby("segment_id").brightness.transform(_robust_score)
    scored["blur_z"] = scored.groupby("segment_id").blur_variance.transform(_robust_score)
    return scored


def _draw_annotation(image: np.ndarray, row: pd.Series, reason: str) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = int(np.clip(row.left, 0, 1) * width)
    y1 = int(np.clip(row.top, 0, 1) * height)
    x2 = int(np.clip(row.right, 0, 1) * width)
    y2 = int(np.clip(row.bottom, 0, 1) * height)
    result = image.copy()
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), max(2, width // 640))
    label = f"{reason} frame={int(row.frame_idx)}"
    cv2.rectangle(result, (8, 8), (min(width - 8, 16 + 12 * len(label)), 48), (0, 0, 0), -1)
    cv2.putText(result, label, (16, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return result


def _write_contact_sheet(pairs: list[tuple[np.ndarray, np.ndarray, str]], output: Path) -> None:
    if not pairs:
        return
    rows = []
    for original, boxed, title in pairs:
        target_width = 480
        scale = target_width / original.shape[1]
        target_height = int(original.shape[0] * scale)
        left = cv2.resize(original, (target_width, target_height))
        right = cv2.resize(boxed, (target_width, target_height))
        pair = np.hstack([left, right])
        cv2.putText(pair, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        rows.append(pair)
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def sample_annotation_anomalies(config: dict, samples_per_reason: int = 4) -> tuple[Path, pd.DataFrame]:
    manifest_path = config["artifacts_dir"] / "dataset" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    scored = score_annotation_quality(manifest)
    output_root = config["artifacts_dir"] / "audit" / "anomaly_samples"
    output_root.mkdir(parents=True, exist_ok=True)

    selectors = {
        "small_box": ("area_z", True),
        "large_box": ("area_z", False),
        "extreme_aspect": ("aspect_z_abs", False),
        "boundary_box": ("boundary_margin", True),
        "position_jump": ("speed_z", False),
        "dark_frame": ("brightness_z", True),
        "bright_frame": ("brightness_z", False),
        "blurred_frame": ("blur_z", True),
    }
    scored["aspect_z_abs"] = scored.aspect_z.abs()
    selected_rows: list[dict] = []
    for segment_id, segment in scored.groupby("segment_id", sort=True):
        for reason, (column, ascending) in selectors.items():
            candidates = segment.dropna(subset=[column]).sort_values(column, ascending=ascending).head(samples_per_reason)
            reason_dir = output_root / segment_id / reason
            reason_dir.mkdir(parents=True, exist_ok=True)
            contact_pairs = []
            for rank, (_, row) in enumerate(candidates.iterrows(), 1):
                source = Path(row.image_path)
                stem = f"rank{rank:02d}_frame{int(row.frame_idx):08d}"
                original_path = reason_dir / f"{stem}_original.jpg"
                boxed_path = reason_dir / f"{stem}_boxed.jpg"
                shutil.copy2(source, original_path)
                original = cv2.imread(str(source))
                boxed = _draw_annotation(original, row, reason)
                cv2.imwrite(str(boxed_path), boxed, [cv2.IMWRITE_JPEG_QUALITY, 95])
                contact_pairs.append((original, boxed, f"{segment_id} {stem}"))
                selected_rows.append({
                    **row.to_dict(), "reason": reason, "rank": rank,
                    "reason_score": float(row[column]),
                    "original_path": str(original_path), "boxed_path": str(boxed_path),
                })
            _write_contact_sheet(contact_pairs, reason_dir / "contact_sheet.jpg")

    selected = pd.DataFrame(selected_rows)
    index_path = output_root / "index.csv"
    selected.to_csv(index_path, index=False)
    scored.to_csv(output_root / "all_frame_scores.csv", index=False)
    return index_path, selected
