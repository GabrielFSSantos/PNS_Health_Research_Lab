"""Registry e implementação dos estágios do pipeline de pré-processamento."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pns_health_research.core.run_progress import RunProgress

from pns_health_research.data.variable_registry import load_variable_registry, validate_raw_columns
from pns_health_research.modules.preprocessing.column_catalog import finalize_column_catalog, record_column
from pns_health_research.modules.preprocessing.contract import PreprocessingContext
from pns_health_research.modules.preprocessing.features import (
    attach_weights,
    create_binary_features,
    fill_missing_binary,
    reinforce_gestational_and_age,
    transform_binary_columns,
    transform_categorical_numeric,
)
from pns_health_research.modules.preprocessing.features_tcc import (
    INTERNAL_WEIGHT_COLS,
    drop_design_columns,
)
from pns_health_research.modules.preprocessing.filtering import (
    filter_adults_legacy,
    select_and_rename_legacy,
)
from pns_health_research.modules.preprocessing.missing_audit import build_missing_audit
from pns_health_research.modules.preprocessing.survey_snapshot import build_survey_snapshot
from pns_health_research.modules.preprocessing.targets import (
    build_target_columns,
    build_target_definitions,
    comorbidity_columns_for_audit,
    target_prevalence,
)

StageFn = Callable[[PreprocessingContext, dict[str, Any]], None]

STAGE_REGISTRY: dict[str, "StageSpec"] = {}


@dataclass(frozen=True)
class StageSpec:
    id: str
    run: StageFn
    requires: tuple[str, ...] = ()


def register_stage(
    stage_id: str,
    *,
    requires: tuple[str, ...] = (),
) -> Callable[[StageFn], StageFn]:
    """Registra um estágio plugável no pipeline de pré-processamento."""

    def decorator(fn: StageFn) -> StageFn:
        STAGE_REGISTRY[stage_id] = StageSpec(stage_id, fn, requires)
        return fn

    return decorator


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


@register_stage("select_columns")
def _run_select_columns(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    _ = params
    pp = ctx.pp
    if pp.columns_map:
        ctx.renamed = select_and_rename_legacy(ctx.raw, column_map=pp.columns_map)
    else:
        ctx.renamed = select_and_rename_legacy(ctx.raw, column_map=None)
    ctx.raw_for_audit = ctx.renamed.copy()
    ctx.filtered = ctx.renamed

    missing_registry = validate_raw_columns(list(ctx.raw.columns))
    if missing_registry:
        ctx.processing_stats["variable_registry_missing"] = missing_registry


@register_stage("adult_filter")
def _run_adult_filter(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    min_age = int(params.get("min_age", ctx.pp.stages.adult_filter.min_age))
    require_imc = bool(
        params.get("require_imc_inputs", ctx.pp.stages.adult_filter.require_imc_inputs)
    )
    assert ctx.filtered is not None
    filtered, filter_stats = filter_adults_legacy(
        ctx.filtered,
        min_age=min_age,
        require_imc_inputs=require_imc,
    )
    ctx.filtered = filtered
    ctx.filter_stats = filter_stats
    ctx.survey_snapshot = build_survey_snapshot(filtered)


def _bootstrap_context(ctx: PreprocessingContext) -> None:
    if ctx.renamed is None:
        ctx.renamed = ctx.raw.copy()
    if ctx.filtered is None:
        ctx.filtered = ctx.renamed.copy()
    if ctx.raw_for_audit is None:
        ctx.raw_for_audit = ctx.renamed.copy()
    if not ctx.filter_stats:
        ctx.filter_stats = _empty_filter_stats(len(ctx.filtered), len(ctx.filtered.columns))


@register_stage("encode_binary")
def _run_encode_binary(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    _ = params
    _bootstrap_context(ctx)
    assert ctx.filtered is not None
    ctx.binary_work = transform_binary_columns(ctx.filtered.copy(), config=ctx.pp)
    ctx.categorical_work = transform_binary_columns(ctx.filtered.copy(), config=ctx.pp)


@register_stage("gestational_reinforce", requires=("encode_binary",))
def _run_gestational_reinforce(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    impute_age = bool(
        params.get("impute_age_by_target", ctx.pp.stages.impute_age_by_target.enabled)
    )
    assert ctx.binary_work is not None and ctx.categorical_work is not None
    ctx.binary_work = reinforce_gestational_and_age(
        ctx.binary_work, impute_age_by_target=impute_age
    )
    ctx.categorical_work = reinforce_gestational_and_age(
        ctx.categorical_work, impute_age_by_target=impute_age
    )


@register_stage("attach_weights", requires=("gestational_reinforce",))
def _run_attach_weights(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    _ = params
    assert ctx.binary_work is not None and ctx.categorical_work is not None
    ctx.binary_work = attach_weights(
        ctx.binary_work, legacy_formula=ctx.pp.legacy_weight_formula
    )
    ctx.categorical_work = attach_weights(
        ctx.categorical_work, legacy_formula=ctx.pp.legacy_weight_formula
    )
    for frame, work in (("binary", ctx.binary_work), ("categorical", ctx.categorical_work)):
        if "Peso Amostral" in work.columns:
            record_column(
                ctx,
                "Peso Amostral",
                frame,  # type: ignore[arg-type]
                semantic_type="weight",
                source_column="V00291",
            )


@register_stage("drop_design_columns", requires=("attach_weights",))
def _run_drop_design_columns(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    _ = params
    assert ctx.binary_work is not None and ctx.categorical_work is not None
    ctx.binary_work = drop_design_columns(ctx.binary_work)
    ctx.categorical_work = drop_design_columns(ctx.categorical_work)


@register_stage("derive_binary", requires=("drop_design_columns",))
def _run_derive_binary(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    one_hot = bool(params.get("one_hot", ctx.pp.stages.one_hot_and_numeric.enabled))
    assert ctx.binary_work is not None
    before = set(ctx.binary_work.columns)
    ctx.binary_work = create_binary_features(ctx.binary_work, config=ctx.pp, one_hot=one_hot)
    for col in set(ctx.binary_work.columns) - before:
        record_column(ctx, col, "binary", semantic_type="derived")


@register_stage("encode_categorical", requires=("drop_design_columns",))
def _run_encode_categorical(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    one_hot = bool(params.get("one_hot", ctx.pp.stages.one_hot_and_numeric.enabled))
    assert ctx.categorical_work is not None
    before = set(ctx.categorical_work.columns)
    ctx.categorical_work = transform_categorical_numeric(
        ctx.categorical_work, config=ctx.pp, one_hot=one_hot
    )
    registry = load_variable_registry()
    for col in set(ctx.categorical_work.columns) - before:
        reg = registry.get_by_label(col)
        semantic = reg.semantic_type if reg else "categorical"
        source = reg.raw_code if reg else None
        record_column(
            ctx,
            col,
            "categorical",
            semantic_type=semantic,
            source_column=source,
        )


@register_stage("build_targets", requires=("encode_binary",))
def _run_build_targets(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    _ = params
    if not ctx.pp.uses_dual_targets():
        return
    assert ctx.binary_work is not None and ctx.categorical_work is not None
    ctx.binary_work = build_target_columns(ctx.binary_work, ctx.pp.target)
    ctx.categorical_work = build_target_columns(ctx.categorical_work, ctx.pp.target)
    for target_name in ctx.pp.target.columns:
        record_column(ctx, target_name, "binary", semantic_type="target")
        record_column(ctx, target_name, "categorical", semantic_type="target")


@register_stage("missing_audit", requires=("encode_binary",))
def _run_missing_audit(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    _ = params
    if not (ctx.pp.emit_missing_audit or ctx.pp.uses_dual_targets()):
        return
    assert (
        ctx.raw_for_audit is not None
        and ctx.filtered is not None
        and ctx.binary_work is not None
    )
    comorb_cols = comorbidity_columns_for_audit()
    ctx.missing_audit = build_missing_audit(
        ctx.raw_for_audit.loc[ctx.filtered.index],
        ctx.binary_work,
        comorb_cols,
    )


@register_stage("impute_missing", requires=("encode_binary",))
def _run_impute_missing(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    binary_fill = int(params.get("binary", ctx.pp.binary_missing_value))
    categorical_fill = int(params.get("categorical", ctx.pp.categorical_missing_value))
    assert ctx.binary_work is not None and ctx.categorical_work is not None
    ctx.binary_work, ctx.bin_imputation = fill_missing_binary(
        ctx.binary_work, fill_value=binary_fill
    )
    ctx.categorical_work, ctx.cat_imputation = fill_missing_binary(
        ctx.categorical_work, fill_value=categorical_fill
    )


@register_stage("finalize_outputs", requires=("encode_binary",))
def _run_finalize_outputs(ctx: PreprocessingContext, params: dict[str, Any]) -> None:
    _ = params
    assert ctx.binary_work is not None and ctx.categorical_work is not None
    bin_numeric = ctx.binary_work.select_dtypes(include="number")
    drop_internal = [c for c in INTERNAL_WEIGHT_COLS if c in bin_numeric.columns]
    if drop_internal:
        bin_numeric = bin_numeric.drop(columns=drop_internal)
    cat_numeric = ctx.categorical_work.select_dtypes(include="number")
    drop_cat_internal = [c for c in INTERNAL_WEIGHT_COLS if c in cat_numeric.columns]
    if drop_cat_internal:
        cat_numeric = cat_numeric.drop(columns=drop_cat_internal)
    ctx.binary_numeric = bin_numeric
    ctx.categorical_numeric = cat_numeric
    finalize_column_catalog(ctx)


def run_registered_stages(
    ctx: PreprocessingContext,
    progress: RunProgress | None = None,
) -> None:
    """Executa pipeline resolvido na config, registrando métricas por estágio."""
    from pns_health_research.core.run_progress import RunProgress

    pipeline = ctx.pp.resolved_pipeline()
    enabled_ids = {step.id for step in pipeline if step.enabled}
    stage_stats: dict[str, Any] = {}
    own_progress = progress is None
    if progress is None:
        progress = RunProgress("pré-processamento", title="Pré-processamento")
        progress.banner(
            perfil=ctx.experiment_profile,
            linhas_entrada=len(ctx.raw),
        )

    for idx, step in enumerate(pipeline, start=1):
        spec = STAGE_REGISTRY.get(step.id)
        if spec is None:
            raise ValueError(f"Estágio desconhecido no pipeline: {step.id!r}")

        record = {
            "id": step.id,
            "enabled": step.enabled,
            "params": dict(step.params),
        }
        ctx.pipeline_executed.append(record)

        if not step.enabled:
            stage_stats[step.id] = {"enabled": False, "skipped": True}
            progress.step_skip(idx, len(pipeline), step.id)
            continue

        for req in spec.requires:
            if req not in enabled_ids:
                raise ValueError(
                    f"Estágio {step.id!r} requer {req!r} habilitado no pipeline"
                )

        t0 = time.perf_counter()
        progress.step_begin(idx, len(pipeline), step.id, indent=1)
        spec.run(ctx, step.params)
        elapsed = time.perf_counter() - t0
        progress.step_end(indent=1)
        rows = len(ctx.filtered) if ctx.filtered is not None else len(ctx.raw)
        stage_stats[step.id] = {
            "enabled": True,
            "elapsed_seconds": round(elapsed, 4),
            "rows": rows,
        }

    ctx.processing_stats["stages"] = stage_stats
    ctx.processing_stats["pipeline_executed"] = ctx.pipeline_executed
    if own_progress:
        progress.finish(
            f"{len(ctx.filtered) if ctx.filtered is not None else len(ctx.raw)} linhas"
        )


def build_result_metrics(ctx: PreprocessingContext) -> dict[str, Any]:
    """Monta métricas finais a partir do contexto."""
    assert ctx.binary_numeric is not None and ctx.categorical_numeric is not None
    pp = ctx.pp
    filter_stats = ctx.filter_stats or _empty_filter_stats(
        len(ctx.raw), len(ctx.raw.columns)
    )

    target_dist: dict[str, int] = {}
    if "Serviço de Saúde" in ctx.binary_numeric.columns:
        vc = ctx.binary_numeric["Serviço de Saúde"].value_counts()
        target_dist = {str(k): int(v) for k, v in vc.items()}

    target_prev = (
        target_prevalence(ctx.categorical_numeric, pp.target)
        if pp.uses_dual_targets()
        else {}
    )

    skipped_impute = {"columns_with_imputation": {}, "total_cells_imputed": 0}
    bin_imputation = ctx.bin_imputation or {
        **skipped_impute,
        "fill_value_binary": pp.binary_missing_value,
    }
    cat_imputation = ctx.cat_imputation or {
        **skipped_impute,
        "fill_value_binary": pp.categorical_missing_value,
    }

    ctx.processing_stats.update(
        {
            "impute_age_by_target": pp.impute_age_by_target,
            "legacy_weight_formula": pp.legacy_weight_formula,
            "binary_encoding_policy": pp.binary_encoding_policy,
            "binary_missing_value": pp.binary_missing_value,
            "categorical_missing_value": pp.categorical_missing_value,
            "require_imc_inputs": pp.stages.adult_filter.require_imc_inputs,
            "schema_version": pp.schema_version,
            "stages_enabled": {
                step.id: step.enabled for step in pp.resolved_pipeline()
            },
            "derived_enabled": {
                key: pp.derived_enabled(key)
                for key in ("imc", "faixas_idade", "grupos_comorbidade")
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
            "target_definitions": (
                build_target_definitions(pp.target) if pp.uses_dual_targets() else {}
            ),
            "missing_audit": ctx.missing_audit,
        }
    )

    return {
        "rows_in": filter_stats["rows_in"],
        "rows_out": filter_stats["rows_out"],
        "removed": filter_stats["removed"],
        "binary_columns": len(ctx.binary_numeric.columns),
        "categorical_columns": len(ctx.categorical_numeric.columns),
        "target_distribution": target_dist,
        "target_prevalence": target_prev,
        "profile": ctx.experiment_profile,
    }
