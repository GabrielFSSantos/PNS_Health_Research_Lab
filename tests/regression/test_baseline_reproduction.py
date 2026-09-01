from __future__ import annotations

import pytest

from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT, RAW_DATA_DIR
from pns_health_research.experiments.runner import ExperimentRunner


@pytest.mark.regression
def test_legacy_reproduction_with_real_pns():
    raw = RAW_DATA_DIR / "pns2019.parquet"
    if not raw.exists():
        pytest.skip("PNS real não disponível em data/raw/pns2019.parquet")

    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    run_dir = ExperimentRunner(cfg).run()
    report_path = run_dir / "reproduction_report.json"
    assert report_path.exists()
    import json

    report = json.loads(report_path.read_text())
    assert "overall_status" in report
