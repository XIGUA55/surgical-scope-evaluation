from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .utils import read_json, write_json


def register_experiment(
    config: dict,
    name: str,
    architecture: str,
    checkpoint: Path,
    experiment_config: Path | None = None,
    metrics_dir: Path | None = None,
    notes: str = "",
) -> tuple[Path, dict]:
    root = config["artifacts_dir"] / "experiments"
    registry_path = root / "registry.json"
    registry = read_json(registry_path) if registry_path.exists() else {"experiments": []}
    metrics_dir = metrics_dir or config["artifacts_dir"] / "evaluation"
    metrics = {}
    for split in ("train", "val", "test"):
        path = metrics_dir / f"{split}_metrics.json"
        if path.exists():
            metrics[split] = read_json(path)
    record = {
        "name": name, "architecture": architecture, "checkpoint": str(checkpoint.resolve()),
        "config": str(experiment_config.resolve()) if experiment_config else str(config["config_path"]),
        "metrics": metrics, "notes": notes,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["experiments"] = [row for row in registry["experiments"] if row["name"] != name] + [record]
    eligible = [row for row in registry["experiments"] if row.get("metrics", {}).get("val")]
    if eligible:
        best = max(eligible, key=lambda row: (
            row["metrics"]["val"].get("map", -1), row["metrics"]["val"].get("map_75", -1)
        ))
        registry["best_by_validation_map"] = {
            "name": best["name"], "architecture": best["architecture"],
            "checkpoint": best["checkpoint"], "config": best["config"],
            "val_map": best["metrics"]["val"].get("map"),
            "val_map_50": best["metrics"]["val"].get("map_50"),
            "val_map_75": best["metrics"]["val"].get("map_75"),
        }
        write_json(root / "best_model.json", registry["best_by_validation_map"])
    write_json(registry_path, registry)
    return registry_path, record
