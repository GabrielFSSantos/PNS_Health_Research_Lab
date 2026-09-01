from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class ModelStore:
    """Persist fitted estimators per run."""

    def __init__(self, run_dir: Path) -> None:
        self.models_dir = run_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, model: Any) -> Path:
        path = self.models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        return path

    def load(self, name: str) -> Any:
        return joblib.load(self.models_dir / f"{name}.joblib")
