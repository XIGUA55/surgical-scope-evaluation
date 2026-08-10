from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


REQUIRED_COLUMNS = {
    "case_id", "segment_id", "frame_idx", "time_s", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "frame_width", "frame_height",
}


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _p95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanpercentile(values, 95)) if np.isfinite(values).any() else float("nan")


def _best_per_frame(group: pd.DataFrame, target_class: str, threshold: float) -> pd.DataFrame:
    frames = group[["frame_idx", "time_s", "frame_width", "frame_height"]].drop_duplicates("frame_idx").sort_values("frame_idx")
    selected = group[(group.class_name == target_class) & (group.confidence >= threshold)]
    # Quality metrics must describe one physical instrument, not jump to the
    # highest-scoring detection at each frame. Prefer the longest persistent
    # track and use confidence only as a tie-breaker.
    if len(selected):
        track_summary = selected.groupby("track_id", dropna=True).agg(
            detected_frames=("frame_idx", "nunique"), mean_confidence=("confidence", "mean")
        ).sort_values(["detected_frames", "mean_confidence"], ascending=False)
        if len(track_summary):
            selected = selected[selected.track_id == track_summary.index[0]]
    selected = selected.sort_values("confidence").drop_duplicates("frame_idx", keep="last")
    columns = ["frame_idx", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    return frames.merge(selected[columns], on="frame_idx", how="left")


def _interpolate_short(series: pd.Series, maximum_gap_frames: int) -> pd.Series:
    return series.interpolate(limit=maximum_gap_frames, limit_area="inside")


def _smooth(values: np.ndarray, window_frames: int) -> np.ndarray:
    result = values.copy()
    valid = np.isfinite(values)
    if valid.sum() < 5:
        return result
    window = min(window_frames if window_frames % 2 else window_frames + 1, int(valid.sum()) // 2 * 2 - 1)
    if window < 5:
        return result
    result[valid] = savgol_filter(values[valid], window_length=window, polyorder=2, mode="interp")
    return result


def _derivative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Differentiate each finite contiguous run without bridging missing data."""
    values, time_s = np.asarray(values, dtype=float), np.asarray(time_s, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values) & np.isfinite(time_s))
    if not len(valid_indices):
        return result
    for run in np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1):
        if len(run) >= 2 and np.all(np.diff(time_s[run]) > 0):
            result[run] = np.gradient(values[run], time_s[run])
    return result


def _recenter_latencies(time_s: np.ndarray, radial: np.ndarray, radius: float) -> list[float]:
    latencies, start = [], None
    for index, distance in enumerate(radial):
        if not np.isfinite(distance):
            continue
        if distance > radius and start is None:
            start = index
        elif distance <= radius and start is not None:
            latencies.append(float(time_s[index] - time_s[start]))
            start = None
    return latencies


def compute_segment_metrics(group: pd.DataFrame, settings: dict) -> dict:
    target_class = str(settings["target_class"])
    frames = _best_per_frame(group, target_class, float(settings["confidence_threshold"]))
    detected = frames.x1.notna().to_numpy()
    qualified_frames = set(group.loc[
        (group.class_name == target_class) & (group.confidence >= float(settings["confidence_threshold"])), "frame_idx"
    ].tolist())
    any_detected = frames.frame_idx.isin(qualified_frames).to_numpy()
    time_s = frames.time_s.to_numpy(float)
    dt_values = np.diff(time_s)
    dt = float(np.median(dt_values[dt_values > 0])) if np.any(dt_values > 0) else 1.0
    max_gap = max(1, int(round(float(settings["maximum_interpolation_gap_seconds"]) / dt)))
    smooth_window = max(5, int(round(float(settings["smoothing_window_seconds"]) / dt)))
    x = (frames.x1 + frames.x2) / 2 / frames.frame_width
    y = (frames.y1 + frames.y2) / 2 / frames.frame_height
    x_interpolated = _interpolate_short(x, max_gap).to_numpy(float)
    y_interpolated = _interpolate_short(y, max_gap).to_numpy(float)
    x_smooth, y_smooth = _smooth(x_interpolated, smooth_window), _smooth(y_interpolated, smooth_window)
    radial = np.hypot(x_smooth - 0.5, y_smooth - 0.5)
    velocity_x, velocity_y = _derivative(x_smooth, time_s), _derivative(y_smooth, time_s)
    speed = np.hypot(velocity_x, velocity_y)
    acceleration = np.hypot(_derivative(velocity_x, time_s), _derivative(velocity_y, time_s))
    jerk = _derivative(acceleration, time_s)
    jitter = np.hypot(x_interpolated - x_smooth, y_interpolated - y_smooth)
    radius = float(settings["center_radius"])
    latencies = _recenter_latencies(time_s, radial, radius)
    def missing_runs(mask: np.ndarray) -> list[int]:
        runs, current = [], 0
        for value in ~mask:
            if value:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        return runs

    any_missing_runs = missing_runs(any_detected)
    trajectory_missing_runs = missing_runs(detected)
    return {
        "case_id": str(group.case_id.iloc[0]), "segment_id": str(group.segment_id.iloc[0]),
        "frame_count": int(len(frames)), "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0,
        "detection_rate": float(any_detected.mean()),
        "dominant_track_coverage_rate": float(detected.mean()),
        "trajectory_metrics_reliable": bool(detected.mean() >= 0.8),
        "centered_rate_detected": float(np.mean(radial[detected & np.isfinite(radial)] <= radius)) if np.any(detected & np.isfinite(radial)) else 0.0,
        "mean_center_distance": _finite_mean(radial), "p95_center_distance": _p95(radial),
        "out_of_view_rate": float(np.mean(((x_smooth < 0) | (x_smooth > 1) | (y_smooth < 0) | (y_smooth > 1))[np.isfinite(x_smooth) & np.isfinite(y_smooth)])) if np.any(np.isfinite(x_smooth) & np.isfinite(y_smooth)) else 0.0,
        "mean_speed": _finite_mean(speed), "p95_speed": _p95(speed),
        "mean_acceleration": _finite_mean(acceleration), "mean_abs_jerk": _finite_mean(np.abs(jerk)),
        "jitter_rms": float(np.sqrt(np.nanmean(jitter ** 2))) if np.isfinite(jitter).any() else float("nan"),
        "recenter_event_count": len(latencies), "mean_recenter_latency_s": _finite_mean(np.asarray(latencies)),
        "detection_gap_count": len(any_missing_runs), "longest_detection_gap_s": max(any_missing_runs, default=0) * dt,
        "dominant_track_gap_count": len(trajectory_missing_runs),
        "longest_dominant_track_gap_s": max(trajectory_missing_runs, default=0) * dt,
    }


def compute_metrics_file(config: dict, tracks_path: Path, output_path: Path | None = None) -> Path:
    return compute_metrics_files(config, [tracks_path], output_path)


def compute_metrics_files(config: dict, tracks_paths: list[Path], output_path: Path | None = None) -> Path:
    if not tracks_paths:
        raise ValueError("At least one tracks CSV is required")
    data = pd.concat([pd.read_csv(path) for path in tracks_paths], ignore_index=True)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing track columns: {sorted(missing)}")
    rows = [compute_segment_metrics(group, config["metrics"]) for _, group in data.groupby(["case_id", "segment_id"], sort=True)]
    output_path = output_path or config["artifacts_dir"] / "metrics" / "camera_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
