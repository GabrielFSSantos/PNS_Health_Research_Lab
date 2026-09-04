from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pns_health_research.modules.classification.cv_routing import run_model_cv
from pns_health_research.modules.classification.feature_selection import resolve_classification_columns
from pns_health_research.modules.classification.validation import (
    ModelResult,
    prepare_classification_df,
)
from pns_health_research.modules.classification.worker import deserialize_result, train_one_model
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.core.gpu import cuda_available, cuda_device_label, gpu_ready_for_training
from pns_health_research.core.run_progress import RunProgress, format_duration

logger = logging.getLogger("pns_health_research")


@dataclass
class ClassificationStageResult:
    model_results: dict[str, ModelResult | dict[str, Any]]
    metrics: dict


def _train_sequential(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    config: ExperimentConfig,
    progress: RunProgress | None = None,
) -> dict[str, ModelResult | dict[str, Any]]:
    from pns_health_research.modules.classification.registry import CLASSIFIER_REGISTRY

    model_results: dict[str, ModelResult | dict[str, Any]] = {}
    names = [n for n in config.classification.models if n in CLASSIFIER_REGISTRY]
    metric = config.evaluation.primary_metric
    for idx, name in enumerate(names, start=1):
        if progress:
            progress.step_begin(idx, len(names), name)
        try:
            model_results[name] = run_model_cv(x, y, weights, name, config)
            res = model_results[name]
            extra = ""
            if isinstance(res, ModelResult):
                score = res.mean_metrics.get(metric)
                if score is not None:
                    extra = f"{metric}={score:.4f}"
            if progress:
                progress.step_end(extra)
        except Exception as exc:
            model_results[name] = {"status": "failed", "error": str(exc)}
            if progress:
                progress.step_end(f"erro: {exc}")
    return model_results


def _train_parallel(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    config: ExperimentConfig,
    progress: RunProgress | None = None,
) -> dict[str, ModelResult | dict[str, Any]]:
    from pns_health_research.modules.classification.registry import CLASSIFIER_REGISTRY

    config_dict = config.model_dump()
    x_dict = x.to_dict(orient="list")
    y_list = y.tolist()
    w_list = weights.tolist()
    names = [n for n in config.classification.models if n in CLASSIFIER_REGISTRY]
    metric = config.evaluation.primary_metric

    model_results: dict[str, ModelResult | dict[str, Any]] = {}
    max_workers = min(config.classification.max_workers, len(names)) or 1
    if progress:
        progress.message(f"paralelo: {max_workers} workers, {len(names)} modelos")

    completed = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(train_one_model, name, x_dict, y_list, w_list, config_dict): name
            for name in names
        }
        for fut in as_completed(futures):
            name, payload = fut.result()
            model_results[name] = deserialize_result(payload)
            completed += 1
            res = model_results[name]
            extra = name
            if isinstance(res, ModelResult):
                score = res.mean_metrics.get(metric)
                if score is not None:
                    extra = f"{name} {metric}={score:.4f}"
            if progress:
                progress.message(f"[{completed}/{len(names)}] {extra} ({format_duration(time.perf_counter() - t0)})")
    return model_results


def _metrics_from_results(
    model_results: dict[str, ModelResult | dict[str, Any]],
) -> dict:
    metrics: dict = {}
    for name, res in model_results.items():
        if isinstance(res, dict) and res.get("status") == "failed":
            metrics[name] = res
        elif isinstance(res, ModelResult):
            metrics[name] = res.mean_metrics
    return metrics


def run_classification_stage(
    clustered_categorical: pd.DataFrame,
    config: ExperimentConfig,
) -> ClassificationStageResult:
    progress = RunProgress("classificação", title="Classificação")
    device_label = "cpu"
    if config.classification.device == "cuda" and gpu_ready_for_training():
        device_label = cuda_device_label()
    elif config.classification.device == "cuda":
        device_label = "cpu (CUDA indisponível para treino)"

    progress.banner(
        perfil=config.experiment.profile,
        device=device_label,
        backend=config.classification.backend,
        modelos=",".join(config.classification.models),
        paralelo=config.classification.parallel,
    )

    progress.step_begin(1, 3, "preparar_features")
    df = prepare_classification_df(clustered_categorical, config)
    feature_cols, _, weight_col = resolve_classification_columns(list(df.columns), config)
    if weight_col not in df.columns and "survey_weight" in df.columns:
        weight_col = "survey_weight"
    target_col = config.preprocessing.classification_features.target_column
    if target_col not in df.columns:
        target_col = "Serviço de Saúde"
    y = df[target_col]
    weights = df[weight_col] if weight_col in df.columns else pd.Series(1.0, index=df.index)
    x = df[feature_cols]
    progress.step_end(f"{len(x)} amostras, {len(feature_cols)} features")

    progress.step_begin(2, 3, "validacao_cruzada")
    if config.classification.parallel and len(config.classification.models) > 1:
        model_results = _train_parallel(x, y, weights, config, progress=progress)
    else:
        model_results = _train_sequential(x, y, weights, config, progress=progress)
    progress.step_end(f"{len(model_results)} modelos")

    metrics = _metrics_from_results(model_results)
    primary = config.evaluation.primary_metric
    best_name = None
    best_score = None
    for name, row in metrics.items():
        if isinstance(row, dict) and row.get("status") == "failed":
            continue
        score = row.get(primary) if isinstance(row, dict) else None
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_name = name
    summary = f"melhor={best_name} {primary}={best_score:.4f}" if best_name else ""
    progress.finish(summary)

    return ClassificationStageResult(
        model_results=model_results,
        metrics=metrics,
    )
