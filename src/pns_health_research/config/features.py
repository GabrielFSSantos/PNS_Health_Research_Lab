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


def load_column_map(path: str | Path | dict[str, str] | None = None) -> dict[str, str]:
    if isinstance(path, dict):
        return dict(path)
    if path is None:
        from pns_health_research.data.catalog import resolve_edition

        return dict(resolve_edition(2019, "pns_2019").columns)
    data = load_yaml(path)
    if "columns" in data:
        return dict(data["columns"])
    return dict(data)


def load_classification_features(path: str | Path | None = None) -> dict[str, Any]:
    if path is not None:
        return load_yaml(path)
    pp_path = CONFIGS_DIR / "_internal" / "pipelines" / "preprocessing" / "pns.yaml"
    if pp_path.exists():
        data = load_yaml(pp_path)
        feats = data.get("classification_features")
        if feats:
            return dict(feats)
    return {}
