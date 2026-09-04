from __future__ import annotations

from pns_health_research.config.features import load_classification_features
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.data.schema import LEGACY_WEIGHT_OUTPUT


def resolve_classification_columns(
    df_columns: list[str],
    config: ExperimentConfig,
) -> tuple[list[str], list[str], str]:
    """Retorna (feature_columns, target_encode_columns, weight_column)."""
    cf = config.preprocessing.classification_features
    yaml_cfg = (
        load_classification_features(config.classification.features_config)
        if config.classification.features_config
        else {}
    )
    exclude = set(
        config.classification.exclude_columns
        or cf.exclude_columns
        or yaml_cfg.get("exclude_columns", [])
    )
    target_encode = (
        config.classification.target_encode_columns
        or cf.target_encode_columns
        or yaml_cfg.get("target_encode_columns", [])
    )
    weight_col = cf.weight_column or yaml_cfg.get("weight_column", LEGACY_WEIGHT_OUTPUT)

    if config.classification.feature_columns:
        features = [c for c in config.classification.feature_columns if c in df_columns]
    else:
        features = [c for c in df_columns if c not in exclude]

    return features, list(target_encode), weight_col
