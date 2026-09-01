from __future__ import annotations

import shutil
from pathlib import Path

from pns_health_research.core.paths import PROJECT_ROOT, RUNS_DIR


def export_baseline_snapshot(
    run_id: str,
    *,
    label: str = "legacy_reproduction",
) -> Path:
    """Copia JSONs pequenos de uma run para artifacts/baselines/ (versionável)."""
    src = RUNS_DIR / run_id
    if not src.exists():
        raise FileNotFoundError(run_id)
    dest = PROJECT_ROOT / "artifacts" / "baselines" / label
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "manifest.json",
        "reproduction_report.json",
        "data_fingerprint.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, dest / name)
    for metric in (
        "preprocessing.json",
        "clustering.json",
        "classification.json",
        "statistics.json",
    ):
        p = src / "metrics" / metric
        if p.exists():
            shutil.copy2(p, dest / metric)
    readme = dest / "README.md"
    readme.write_text(
        f"# Baseline: {label}\n\n"
        f"Snapshot exportado da run local `{run_id}`.\n"
        f"Parquets não versionados — apenas métricas e relatórios.\n",
        encoding="utf-8",
    )
    return dest
