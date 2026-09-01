
import pandas as pd

from pns_health_research.classification.tuning import effective_n_jobs, load_model_n_jobs
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.persistence.run_loader import load_clustered_categorical


def test_load_model_n_jobs_from_yaml():
    assert load_model_n_jobs("random_forest") == 2
    assert load_model_n_jobs("mlp") == 1


def test_effective_n_jobs_parallel_off():
    n = effective_n_jobs("random_forest", parallel=False, max_workers=3)
    assert n >= 1


def test_effective_n_jobs_parallel_on():
    n = effective_n_jobs("unknown_model", parallel=True, max_workers=4)
    assert n >= 1


def test_load_clustered_categorical_roundtrip(tmp_path, monkeypatch):
    run_id = "test_run_abc"
    run_dir = tmp_path / "runs" / run_id / "tables"
    run_dir.mkdir(parents=True)
    df = pd.DataFrame({"Cluster": [0, 1], "Serviço de Saúde": [0, 1]})
    df.to_parquet(run_dir / "clustered_categorical.parquet", index=False)
    monkeypatch.setattr("pns_health_research.persistence.run_loader.RUNS_DIR", tmp_path / "runs")
    loaded = load_clustered_categorical(run_id)
    pd.testing.assert_frame_equal(loaded, df)


def test_legacy_config_parallel_and_no_svc():
    from pns_health_research.config.loader import load_config

    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy_reproduction.yaml")
    assert cfg.classification.parallel is True
    assert cfg.classification.max_workers == 3
    assert "svc" not in cfg.classification.models
    assert cfg.classification.input_run is None
    assert "preprocessing" in cfg.stages
