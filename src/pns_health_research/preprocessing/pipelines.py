from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pns_health_research.config.features import load_column_map
from pns_health_research.config.models import ExperimentConfig, PreprocessingConfig
from pns_health_research.data.loader import load_raw_pns
from pns_health_research.data.validation import add_record_id
from pns_health_research.preprocessing.features import (
    attach_weights,
    create_binary_features,
    fill_missing_binary,
    reinforce_gestational_and_age,
    transform_binary_columns,
    transform_categorical_numeric,
)
from pns_health_research.preprocessing.features_tcc import (
    INTERNAL_WEIGHT_COLS,
    drop_design_columns,
)
from pns_health_research.preprocessing.filtering import (
    filter_adults_legacy,
    select_and_rename_legacy,
)


@dataclass
class PreprocessingResult:
    binary_df: pd.DataFrame
    categorical_df: pd.DataFrame
    metrics: dict
    filter_stats: dict = field(default_factory=dict)


def run_preprocessing(
    config: ExperimentConfig,
    raw_path: str | None = None,
) -> PreprocessingResult:
    pp: PreprocessingConfig = config.preprocessing
    raw = load_raw_pns(raw_path or config.data.source)
    raw = add_record_id(raw, config.data.record_id_columns)
    column_map = load_column_map(pp.columns_config) if pp.columns_config else None
    renamed = select_and_rename_legacy(raw, column_map=column_map)
    filtered, filter_stats = filter_adults_legacy(renamed)

    bin_work = transform_binary_columns(filtered.copy())
    cat_work = filtered.copy()

    bin_work = reinforce_gestational_and_age(bin_work, impute_age_by_target=pp.impute_age_by_target)
    cat_work = reinforce_gestational_and_age(cat_work, impute_age_by_target=pp.impute_age_by_target)

    # Pesos legados usam UF antes do one-hot regional.
    bin_work = attach_weights(bin_work, legacy_formula=pp.legacy_weight_formula)
    cat_work = attach_weights(cat_work, legacy_formula=pp.legacy_weight_formula)

    bin_work = drop_design_columns(bin_work)
    cat_work = drop_design_columns(cat_work)

    bin_work = create_binary_features(bin_work)
    cat_work = transform_categorical_numeric(cat_work)

    bin_work = fill_missing_binary(bin_work, fill_value=pp.binary_missing_value)
    cat_work = fill_missing_binary(cat_work, fill_value=pp.categorical_missing_value)

    bin_numeric = bin_work.select_dtypes(include="number")
    drop_internal = [c for c in INTERNAL_WEIGHT_COLS if c in bin_numeric.columns]
    if drop_internal:
        bin_numeric = bin_numeric.drop(columns=drop_internal)
    cat_numeric = cat_work.select_dtypes(include="number")
    drop_cat_internal = [c for c in INTERNAL_WEIGHT_COLS if c in cat_numeric.columns]
    if drop_cat_internal:
        cat_numeric = cat_numeric.drop(columns=drop_cat_internal)

    target_dist = {}
    if "Serviço de Saúde" in bin_numeric.columns:
        vc = bin_numeric["Serviço de Saúde"].value_counts()
        target_dist = {str(k): int(v) for k, v in vc.items()}

    metrics = {
        "rows_in": filter_stats["rows_in"],
        "rows_out": filter_stats["rows_out"],
        "removed": filter_stats["removed"],
        "binary_columns": len(bin_numeric.columns),
        "categorical_columns": len(cat_numeric.columns),
        "target_distribution": target_dist,
    }

    return PreprocessingResult(
        binary_df=bin_numeric,
        categorical_df=cat_numeric,
        metrics=metrics,
        filter_stats=filter_stats,
    )
