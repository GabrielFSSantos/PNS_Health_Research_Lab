from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT


def test_research_v2_no_global_leakage_flags():
    cfg = load_config(PROJECT_ROOT / "configs/experiments/research_v2.yaml")
    assert cfg.experiment.profile == "research_v2"
    assert cfg.preprocessing.target_encoding_before_split is False
    assert cfg.preprocessing.nearmiss_before_split is False
    assert cfg.preprocessing.impute_age_by_target is False
    assert cfg.classification.legacy_cv is False
