from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def ensure_dirs() -> None:
    for d in (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
