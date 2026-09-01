from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pns_health_research.config.models import ExperimentConfig


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        for sub in ("metrics", "tables", "predictions", "models", "logs"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

    def write_json(self, rel_path: str, data: dict[str, Any]) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def write_yaml(self, rel_path: str, data: dict[str, Any]) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    def write_parquet(self, rel_path: str, df: pd.DataFrame) -> Path:
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path


def make_run_id(experiment_name: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = hashlib.sha256(ts.encode()).hexdigest()[:7]
    safe_name = experiment_name.replace(" ", "-").lower()
    return f"{ts}_{safe_name}_{short}"


def config_hash(config: ExperimentConfig) -> str:
    payload = config.model_dump_json()
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_manifest(
    run_id: str,
    config: ExperimentConfig,
    status: str,
    started_at: str,
    finished_at: str | None,
    data_fingerprint: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_name": config.experiment.name,
        "profile": config.experiment.profile,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "random_seed": config.experiment.seed,
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "data_fingerprint": data_fingerprint,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
