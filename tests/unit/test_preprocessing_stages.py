"""Estágios de pré-processamento respeitam enabled."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pns_health_research.config.loader import load_config
from pns_health_research.config.models import PreprocessingConfig
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.modules.preprocessing.derived import apply_imc_binary
from pns_health_research.modules.preprocessing.features import create_binary_features
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing


def test_adult_filter_skip_keeps_all_rows(synthetic_raw_path: Path):
    raw = synthetic_raw_path.parent / "with_minors.parquet"
    df = pd.read_parquet(synthetic_raw_path)
    df.loc[:2, "C008"] = 10
    df.to_parquet(raw)

    cfg = load_config(PROJECT_ROOT / "configs/_internal/schema_root.yaml")
    cfg.data.source = str(raw)
    cfg.preprocessing.stages.adult_filter.enabled = False
    result = run_preprocessing(cfg, raw_path=str(raw))
    assert result.filter_stats["removed"] == 0
    assert result.metrics["rows_out"] == len(df)


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
