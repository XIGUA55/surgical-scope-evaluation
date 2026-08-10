from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_RATINGS = {"case_id", "segment_id", "rater_id", "salas_total", "osa_cns_total"}


def _bootstrap_spearman(valid: pd.DataFrame, feature: str, score: str, iterations: int, seed: int) -> tuple[float, float]:
    cases = valid.case_id.drop_duplicates().to_numpy()
    if len(cases) < 3:
        return float("nan"), float("nan")
    rng, estimates = np.random.default_rng(seed), []
    groups = {case: valid[valid.case_id == case] for case in cases}
    for _ in range(iterations):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        replicate = pd.concat([groups[case] for case in sampled], ignore_index=True)
        if replicate[feature].nunique() > 1 and replicate[score].nunique() > 1:
            estimates.append(float(spearmanr(replicate[feature], replicate[score]).statistic))
    if not estimates:
        return float("nan"), float("nan")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def icc_2_1(matrix: np.ndarray) -> float:
    """Two-way random effects, absolute agreement, single-measure ICC."""
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means, column_means = matrix.mean(axis=1), matrix.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = np.sum((matrix - row_means[:, None] - column_means[None, :] + grand) ** 2)
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def _score_icc(ratings: pd.DataFrame, score: str) -> float:
    table = ratings.pivot_table(index=["case_id", "segment_id"], columns="rater_id", values=score, aggfunc="mean").dropna()
    return icc_2_1(table.to_numpy())


def analyze_ratings(config: dict, ratings_path: Path, metrics_path: Path) -> tuple[Path, dict]:
    ratings, metrics = pd.read_csv(ratings_path), pd.read_csv(metrics_path)
    missing = REQUIRED_RATINGS - set(ratings.columns)
    if missing:
        raise ValueError(f"Missing rating columns: {sorted(missing)}")
    means = ratings.groupby(["case_id", "segment_id"], as_index=False)[["salas_total", "osa_cns_total"]].mean()
    merged = metrics.merge(means, on=["case_id", "segment_id"], how="inner")
    feature_columns = [
        column for column in metrics.columns
        if column not in {"case_id", "segment_id", "frame_count", "duration_s"} and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    correlations = []
    for feature in feature_columns:
        for score in ("salas_total", "osa_cns_total"):
            valid = merged[["case_id", feature, score]].dropna()
            statistic, pvalue = spearmanr(valid[feature], valid[score]) if len(valid) >= 3 else (np.nan, np.nan)
            lower, upper = _bootstrap_spearman(
                valid, feature, score, int(config.get("ratings", {}).get("bootstrap_iterations", 2000)), int(config["seed"])
            )
            correlations.append({
                "feature": feature, "score": score, "spearman_rho": statistic,
                "ci95_case_bootstrap_low": lower, "ci95_case_bootstrap_high": upper,
                "p_value": pvalue, "n": len(valid), "independent_cases": valid.case_id.nunique(),
            })
    output_dir = config["artifacts_dir"] / "ratings"
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation_path = output_dir / "correlations.csv"
    pd.DataFrame(correlations).to_csv(correlation_path, index=False)
    summary = {
        "segments_with_scores": int(len(merged)), "independent_cases": int(merged.case_id.nunique()) if len(merged) else 0,
        "salas_icc_2_1": _score_icc(ratings, "salas_total"),
        "osa_cns_icc_2_1": _score_icc(ratings, "osa_cns_total"),
        "predictive_validation": None,
    }
    if merged.case_id.nunique() >= 3 and len(merged) >= 6 and feature_columns:
        complete = merged.dropna(subset=feature_columns + ["salas_total"])
        if complete.case_id.nunique() >= 3:
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            predictions = cross_val_predict(estimator, complete[feature_columns], complete.salas_total, groups=complete.case_id, cv=LeaveOneGroupOut())
            summary["predictive_validation"] = {
                "target": "salas_total", "mae": float(mean_absolute_error(complete.salas_total, predictions)),
                "r2": float(r2_score(complete.salas_total, predictions)), "method": "leave-one-case-out",
            }
    pd.DataFrame([summary]).to_json(output_dir / "summary.json", orient="records", indent=2, force_ascii=False)
    return correlation_path, summary


def write_rating_template(config: dict) -> Path:
    output = config["project_root"] / "ratings" / "expert_scores_template.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        pd.DataFrame(columns=sorted(REQUIRED_RATINGS)).to_csv(output, index=False)
    return output
