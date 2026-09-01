from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT


def test_experiment_yaml_no_legacy_leakage_flags():
    cfg = load_config(PROJECT_ROOT / "configs/experiment.yaml")
    assert cfg.experiment.profile == "custom"
    assert cfg.preprocessing.target_encoding_before_split is False
    assert cfg.preprocessing.nearmiss_before_split is False
    assert cfg.preprocessing.impute_age_by_target is False
    assert cfg.classification.legacy_cv is False
