"""Orquestrador do pipeline de pré-processamento (schema 3.0)."""

from __future__ import annotations

from pns_health_research.config.models import ExperimentConfig, PreprocessingConfig
from pns_health_research.data.catalog import resolve_edition
from pns_health_research.data.loader import load_raw_pns
from pns_health_research.data.validation import add_record_id
from pns_health_research.modules.preprocessing.contract import PreprocessingContext, PreprocessingResult
from pns_health_research.modules.preprocessing.pipeline_adapter import validate_pipeline
from pns_health_research.modules.preprocessing.stages.registry import (
    build_result_metrics,
    run_registered_stages,
)


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


def run_preprocessing_pipeline(
    config: ExperimentConfig,
    raw_path: str | None = None,
) -> PreprocessingResult:
    import time

    from pns_health_research.core.run_progress import RunProgress

    pp = _sync_from_catalog(config)
    validate_pipeline(pp.resolved_pipeline())

    progress = RunProgress("pré-processamento", title="Pré-processamento")
    progress.banner(
        perfil=config.experiment.profile,
        dataset=config.data.dataset,
        ano=config.data.year,
    )

    t0 = time.perf_counter()
    progress.step_begin(1, 2, "carregar_dados")
    path = raw_path or config.data.source
    if path is None:
        raw = load_raw_pns(dataset=config.data.dataset, year=config.data.year)
    else:
        raw = load_raw_pns(path, dataset=config.data.dataset, year=config.data.year)
    raw = add_record_id(raw, config.data.record_id_columns)
    progress.step_end(f"{len(raw)} linhas")

    progress.step_begin(2, 2, "pipeline_estagios", detail=f"{len(pp.resolved_pipeline())} estágios")
    ctx = PreprocessingContext(
        pp=pp,
        experiment_profile=config.experiment.profile,
        raw=raw,
    )

    run_registered_stages(ctx, progress=progress)
    progress.step_end()

    assert ctx.binary_numeric is not None and ctx.categorical_numeric is not None
    metrics = build_result_metrics(ctx)
    metrics["data_year"] = config.data.year
    metrics["data_dataset"] = config.data.dataset

    progress.finish(
        f"{metrics['rows_out']} linhas, bin={metrics['binary_columns']}, "
        f"cat={metrics['categorical_columns']}"
    )

    return PreprocessingResult(
        binary_df=ctx.binary_numeric,
        categorical_df=ctx.categorical_numeric,
        metrics=metrics,
        filter_stats=ctx.filter_stats,
        processing_stats=ctx.processing_stats,
        survey_snapshot=ctx.survey_snapshot,
        column_catalog=list(ctx.column_catalog),
    )
