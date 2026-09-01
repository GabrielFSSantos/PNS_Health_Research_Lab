from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricStore:
    """Filesystem-backed metric persistence for a run."""

    def __init__(self, run_dir: Path) -> None:
        self.metrics_dir = run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, data: dict[str, Any]) -> Path:
        path = self.metrics_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def read(self, name: str) -> dict[str, Any]:
        path = self.metrics_dir / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))
