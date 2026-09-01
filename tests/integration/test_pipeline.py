from pathlib import Path

import pytest
from tests.fixtures.synthetic_pns import make_synthetic_pns

from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.experiments.runner import ExperimentRunner
from pns_health_research.preprocessing.pipelines import run_preprocessing


@pytest.fixture
def synthetic_raw_path(tmp_path: Path) -> Path:
    df = make_synthetic_pns(300)
    path = tmp_path / "pns2019.parquet"
    df.to_parquet(path)
    return path


def test_preprocessing_pipeline(synthetic_raw_path: Path):
    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    result = run_preprocessing(cfg, raw_path=synthetic_raw_path)
    assert result.metrics["rows_out"] > 0
    assert result.binary_df.shape[1] > 10


def test_integration_experiment_smoke(synthetic_raw_path: Path, tmp_path: Path, monkeypatch):
    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    cfg.data.source = str(synthetic_raw_path)
    cfg.classification.models = ["mlp"]
    cfg.classification.n_splits = 2
    cfg.stages = ["preprocessing", "clustering", "classification"]
    cfg.baseline.enabled = False
    monkeypatch.setattr(
        "pns_health_research.experiments.runner.RUNS_DIR",
        tmp_path / "runs",
    )
    run_dir = ExperimentRunner(cfg).run()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "metrics/preprocessing.json").exists()
