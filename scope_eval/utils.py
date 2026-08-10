from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def stable_environment() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("OMP_NUM_THREADS", "").isdigit() or int(env.get("OMP_NUM_THREADS", "0")) < 1:
        env["OMP_NUM_THREADS"] = "1"
    return env


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

