"""Estágios de pré-processamento respeitam enabled."""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.synthetic_pns import make_synthetic_pns

from pns_health_research.config.loader import load_config
from pns_health_research.config.models import PreprocessingConfig
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.preprocessing.derived import apply_imc_binary
from pns_health_research.preprocessing.features import create_binary_features
from pns_health_research.preprocessing.pipelines import run_preprocessing


def test_adult_filter_skip_keeps_all_rows(tmp_path: Path):
    raw = make_synthetic_pns(80)
    raw.loc[:10, "C008"] = 10
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path)

    cfg = load_config(PROJECT_ROOT / "configs/experiment.yaml")
    cfg.data.source = str(path)
    cfg.preprocessing.stages.adult_filter.enabled = False
    result = run_preprocessing(cfg, raw_path=str(path))
    assert result.filter_stats["removed"] == 0
    assert result.metrics["rows_out"] == len(raw)


def test_derived_imc_disabled_keeps_peso_altura():
    import pandas as pd

    df = pd.DataFrame({"Peso": [70.0], "Altura": [170.0], "Idade": [30]})
    pp = PreprocessingConfig()
    pp.derived["imc"].enabled = False
    out = create_binary_features(df, config=pp)
    assert "Baixo Peso" not in out.columns
    assert "Peso" in out.columns
    assert "Idade 18 a 29" in out.columns or "Idade 30 a 39" in out.columns


def test_derived_imc_enabled_matches_helper():
    import pandas as pd

    df = pd.DataFrame({"Peso": [70.0], "Altura": [170.0]})
    expected = apply_imc_binary(df)
    pp = PreprocessingConfig()
    out = create_binary_features(df, config=pp)
    for col in ("Baixo Peso", "Peso Normal", "Sobrepeso", "Obesidade"):
        assert col in out.columns
        assert col in expected.columns
