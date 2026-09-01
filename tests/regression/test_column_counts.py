
import pytest

from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.preprocessing.pipelines import run_preprocessing


@pytest.mark.regression
def test_tcc_column_counts_with_real_pns():
    raw = PROJECT_ROOT / "data/raw/pns2019.parquet"
    if not raw.exists():
        pytest.skip("PNS real ausente")
    cfg = load_config(PROJECT_ROOT / "configs/experiments/legacy.yaml")
    result = run_preprocessing(cfg)
    assert result.metrics["rows_out"] == 87678
    assert result.metrics["binary_columns"] == 161
    assert result.metrics["categorical_columns"] == 61
