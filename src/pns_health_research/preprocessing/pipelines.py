from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pns_health_research.config.models import ExperimentConfig, PreprocessingConfig
from pns_health_research.data.catalog import resolve_edition
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
from pns_health_research.preprocessing.survey_snapshot import build_survey_snapshot


@dataclass
class PreprocessingResult:
    binary_df: pd.DataFrame
    categorical_df: pd.DataFrame
    metrics: dict
    filter_stats: dict = field(default_factory=dict)
    processing_stats: dict = field(default_factory=dict)
    survey_snapshot: dict = field(default_factory=dict)


def _empty_filter_stats(n: int, n_cols: int) -> dict[str, int]:
    return {
        "rows_in": n,
        "rows_out": n,
        "removed": 0,
        "removed_by_age_or_missing": 0,
        "removed_under_18": 0,
        "removed_missing_weight": 0,
        "removed_missing_height": 0,
        "removed_missing_health_service": 0,
        "columns_selected": n_cols,
    }


def _sync_from_catalog(config: ExperimentConfig) -> PreprocessingConfig:
    """Preenche columns_map e record_id a partir do catálogo quando vazio."""
    pp = config.preprocessing
    try:
        edition = resolve_edition(config.data.year, config.data.dataset)
        if not pp.columns_map:
            pp = pp.model_copy(update={"columns_map": edition.columns})
        if not config.data.record_id_columns:
            config.data.record_id_columns = edition.record_id_columns
    except Exception:
        pass
    return pp


def run_preprocessing(
    config: ExperimentConfig,
    raw_path: str | None = None,
) -> PreprocessingResult:
    pp = _sync_from_catalog(config)
    stages = pp.stages
    path = raw_path or config.data.source
    if path is None:
        raw = load_raw_pns(dataset=config.data.dataset, year=config.data.year)
    else:
        raw = load_raw_pns(path, dataset=config.data.dataset, year=config.data.year)
    raw = add_record_id(raw, config.data.record_id_columns)

    if stages.select_columns.enabled:
        column_map = pp.columns_map or None
        renamed = select_and_rename_legacy(raw, column_map=column_map)
    else:
        renamed = raw.copy()

    if stages.adult_filter.enabled:
        filtered, filter_stats = filter_adults_legacy(renamed, min_age=stages.adult_filter.min_age)
    else:
        filtered = renamed
        filter_stats = _empty_filter_stats(len(renamed), len(renamed.columns))

    survey_snapshot = build_survey_snapshot(filtered)

    bin_work = transform_binary_columns(filtered.copy())
    cat_work = filtered.copy()

    bin_work = reinforce_gestational_and_age(
        bin_work, impute_age_by_target=pp.impute_age_by_target
    )
    cat_work = reinforce_gestational_and_age(
        cat_work, impute_age_by_target=pp.impute_age_by_target
    )

    bin_work = attach_weights(bin_work, legacy_formula=pp.legacy_weight_formula)
    cat_work = attach_weights(cat_work, legacy_formula=pp.legacy_weight_formula)

    bin_work = drop_design_columns(bin_work)
    cat_work = drop_design_columns(cat_work)

    bin_work = create_binary_features(bin_work, config=pp)
    cat_work = transform_categorical_numeric(cat_work, config=pp)

    skipped_impute = {"columns_with_imputation": {}, "total_cells_imputed": 0}
    if stages.impute_missing.enabled:
        bin_work, bin_imputation = fill_missing_binary(
            bin_work, fill_value=pp.binary_missing_value
        )
        cat_work, cat_imputation = fill_missing_binary(
            cat_work, fill_value=pp.categorical_missing_value
        )
    else:
        bin_imputation = {**skipped_impute, "fill_value_binary": pp.binary_missing_value}
        cat_imputation = {**skipped_impute, "fill_value_binary": pp.categorical_missing_value}

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

    processing_stats = {
        "impute_age_by_target": pp.impute_age_by_target,
        "legacy_weight_formula": pp.legacy_weight_formula,
        "binary_missing_value": pp.binary_missing_value,
        "categorical_missing_value": pp.categorical_missing_value,
        "stages_enabled": {
            "select_columns": stages.select_columns.enabled,
            "adult_filter": stages.adult_filter.enabled,
            "derive_features": stages.derive_features.enabled,
            "one_hot_and_numeric": stages.one_hot_and_numeric.enabled,
            "impute_missing": stages.impute_missing.enabled,
            "impute_age_by_target": stages.impute_age_by_target.enabled,
        },
        "derived_enabled": {
            key: pp.derived_enabled(key) for key in ("imc", "faixas_idade", "grupos_comorbidade")
        },
        "imputation": {
            "binary": bin_imputation,
            "categorical": cat_imputation,
            "columns_with_imputation": {
                **bin_imputation.get("columns_with_imputation", {}),
                **cat_imputation.get("columns_with_imputation", {}),
            },
            "total_cells_imputed": bin_imputation.get("total_cells_imputed", 0)
            + cat_imputation.get("total_cells_imputed", 0),
        },
    }

    metrics = {
        "rows_in": filter_stats["rows_in"],
        "rows_out": filter_stats["rows_out"],
        "removed": filter_stats["removed"],
        "binary_columns": len(bin_numeric.columns),
        "categorical_columns": len(cat_numeric.columns),
        "target_distribution": target_dist,
        "data_year": config.data.year,
        "data_dataset": config.data.dataset,
    }

    return PreprocessingResult(
        binary_df=bin_numeric,
        categorical_df=cat_numeric,
        metrics=metrics,
        filter_stats=filter_stats,
        processing_stats=processing_stats,
        survey_snapshot=survey_snapshot,
    )
