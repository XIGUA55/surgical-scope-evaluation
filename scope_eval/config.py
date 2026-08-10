from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "project.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = config_path.parent
    root_value = Path(config.get("project_root", ".."))
    root = (base / root_value).resolve() if not root_value.is_absolute() else root_value
    config["project_root"] = root
    for key in ("data_dir", "artifacts_dir", "video_dir", "label_dir"):
        value = Path(config[key])
        config[key] = value if value.is_absolute() else root / value
    config["config_path"] = config_path
    config["class_names"] = {int(k): str(v) for k, v in config["class_names"].items()}
    return config

