"""Convert per-frame instrument detections into interpretable camera metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED = {
    "video_id", "frame_idx", "time_s", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "frame_width", "frame_height",
}


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def compute_metrics(group: pd.DataFrame, config: dict) -> dict[str, float | str]:
    """Calculate metrics for one video/segment using its best target per frame."""
    target = config["target_class"]
    threshold = float(config["confidence_threshold"])
    all_frames = group[["frame_idx", "time_s"]].drop_duplicates().sort_values("time_s")
    selected = group[(group.class_name == target) & (group.confidence >= threshold)]
    selected = selected.sort_values("confidence").drop_duplicates("frame_idx", keep="last")
    frames = all_frames.merge(selected, on=["frame_idx", "time_s"], how="left")

    detected = frames["x1"].notna().to_numpy()
    x = ((frames.x1 + frames.x2) / 2 / frames.frame_width).to_numpy(float)
    y = ((frames.y1 + frames.y2) / 2 / frames.frame_height).to_numpy(float)
    t = frames.time_s.to_numpy(float)
    radial = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

    # Derivatives are calculated only across adjacent valid detections, avoiding
    # artificial jumps across long detector dropouts.
    valid_idx = np.flatnonzero(detected)
    speed = np.full(len(frames), np.nan)
    acceleration = np.full(len(frames), np.nan)
    jerk = np.full(len(frames), np.nan)
    if valid_idx.size >= 2:
        contiguous = valid_idx[1:] == valid_idx[:-1] + 1
        dt = np.diff(t[valid_idx])
        distance = np.hypot(np.diff(x[valid_idx]), np.diff(y[valid_idx]))
        ok = contiguous & (dt > 0)
        speed[valid_idx[1:][ok]] = distance[ok] / dt[ok]
    finite_speed = np.flatnonzero(np.isfinite(speed))
    if finite_speed.size >= 2:
        pairs = finite_speed[1:] == finite_speed[:-1] + 1
        dt = np.diff(t[finite_speed])
        ok = pairs & (dt > 0)
        acceleration[finite_speed[1:][ok]] = np.diff(speed[finite_speed])[ok] / dt[ok]
    finite_acc = np.flatnonzero(np.isfinite(acceleration))
    if finite_acc.size >= 2:
        pairs = finite_acc[1:] == finite_acc[:-1] + 1
        dt = np.diff(t[finite_acc])
        ok = pairs & (dt > 0)
        jerk[finite_acc[1:][ok]] = np.diff(acceleration[finite_acc])[ok] / dt[ok]

    radius = float(config["center_radius"])
    return {
        "video_id": str(group.video_id.iloc[0]),
        "frame_count": int(len(frames)),
        "duration_s": float(t.max() - t.min()) if len(t) > 1 else 0.0,
        "detection_rate": float(detected.mean()),
        "centered_rate": float(np.mean(radial[detected] <= radius)) if detected.any() else 0.0,
        "mean_center_distance": _safe_mean(radial),
        "p95_center_distance": float(np.nanpercentile(radial, 95)) if detected.any() else float("nan"),
        "mean_speed": _safe_mean(speed),
        "p95_speed": float(np.nanpercentile(speed, 95)) if np.isfinite(speed).any() else float("nan"),
        "mean_abs_acceleration": _safe_mean(np.abs(acceleration)),
        "mean_abs_jerk": _safe_mean(np.abs(jerk)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.detections)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = pd.DataFrame(compute_metrics(g, config) for _, g in data.groupby("video_id"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
