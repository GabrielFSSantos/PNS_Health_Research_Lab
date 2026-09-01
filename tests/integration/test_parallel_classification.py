from pathlib import Path

import pytest
from tests.fixtures.synthetic_pns import make_synthetic_pns

from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.experiments.runner import ExperimentRunner


def test_parallel_classification_smoke(synthetic_raw_path: Path, tmp_path: Path, monkeypatch):
    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    cfg.data.source = str(synthetic_raw_path)
    cfg.classification.models = ["mlp", "random_forest"]
    cfg.classification.parallel = True
    cfg.classification.max_workers = 2
    cfg.classification.n_splits = 2
    cfg.stages = ["preprocessing", "clustering", "classification"]
    cfg.baseline.enabled = False
    monkeypatch.setattr(
        "pns_health_research.experiments.runner.RUNS_DIR",
        tmp_path / "runs",
    )
    run_dir = ExperimentRunner(cfg).run()
    assert (run_dir / "metrics/classification.json").exists()
    assert (run_dir / "tables/clustered_categorical.parquet").exists()


@pytest.fixture
def synthetic_raw_path(tmp_path: Path) -> Path:
    df = make_synthetic_pns(300)
    path = tmp_path / "pns2019.parquet"
    df.to_parquet(path)
    return path


def test_resume_from_run(synthetic_raw_path: Path, tmp_path: Path, monkeypatch):
    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    cfg.data.source = str(synthetic_raw_path)
    cfg.classification.models = ["mlp"]
    cfg.classification.n_splits = 2
    cfg.stages = ["preprocessing", "clustering", "classification"]
    cfg.baseline.enabled = False
    runs = tmp_path / "runs"
    monkeypatch.setattr("pns_health_research.experiments.runner.RUNS_DIR", runs)
    monkeypatch.setattr("pns_health_research.persistence.run_loader.RUNS_DIR", runs)

    source_dir = ExperimentRunner(cfg).run()
    source_id = source_dir.name

    cfg2 = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    cfg2.classification.input_run = source_id
    cfg2.classification.models = ["mlp"]
    cfg2.classification.n_splits = 2
    cfg2.stages = ["classification"]
    cfg2.baseline.enabled = False
    resume_dir = ExperimentRunner(cfg2).run()
    assert (resume_dir / "metrics/classification.json").exists()
    assert (resume_dir / "reproduction_report.json").exists()
