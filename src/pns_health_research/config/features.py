from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pns_health_research.core.paths import CONFIGS_DIR, PROJECT_ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(_resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_column_map(path: str | Path | None = None) -> dict[str, str]:
    cfg_path = path or CONFIGS_DIR / "data" / "pns2019_columns.yaml"
    data = load_yaml(cfg_path)
    return dict(data.get("columns", {}))


def load_classification_features(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIGS_DIR / "features" / "classification_default.yaml"
    return load_yaml(cfg_path)
