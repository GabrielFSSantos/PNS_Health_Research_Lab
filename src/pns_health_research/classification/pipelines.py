from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pns_health_research.classification.feature_selection import resolve_classification_columns
from pns_health_research.classification.validation import (
    ModelResult,
    prepare_classification_df,
    run_legacy_cv,
    run_research_v2_cv,
)
from pns_health_research.classification.worker import deserialize_result, train_one_model
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.core.gpu import cuda_available, cuda_device_label

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
) -> dict[str, ModelResult | dict[str, Any]]:
    from pns_health_research.classification.registry import CLASSIFIER_REGISTRY

    model_results: dict[str, ModelResult | dict[str, Any]] = {}
    for name in config.classification.models:
        if name not in CLASSIFIER_REGISTRY:
            continue
        try:
            if (
                config.experiment.profile == "research_v2"
                and config.classification.backend == "gpu"
            ):
                from pns_health_research.classification.gpu.pipelines import run_gpu_cv

                model_results[name] = run_gpu_cv(x, y, weights, name, config)
            elif config.experiment.profile == "research_v2":
                model_results[name] = run_research_v2_cv(x, y, weights, name, config)
            else:
                model_results[name] = run_legacy_cv(x, y, weights, name, config)
        except Exception as exc:
            model_results[name] = {"status": "failed", "error": str(exc)}
    return model_results


def _train_parallel(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    config: ExperimentConfig,
) -> dict[str, ModelResult | dict[str, Any]]:
    from pns_health_research.classification.registry import CLASSIFIER_REGISTRY

    config_dict = config.model_dump()
    x_dict = x.to_dict(orient="list")
    y_list = y.tolist()
    w_list = weights.tolist()
    names = [n for n in config.classification.models if n in CLASSIFIER_REGISTRY]

    model_results: dict[str, ModelResult | dict[str, Any]] = {}
    max_workers = min(config.classification.max_workers, len(names)) or 1

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(train_one_model, name, x_dict, y_list, w_list, config_dict): name
            for name in names
        }
        for fut in as_completed(futures):
            name, payload = fut.result()
            model_results[name] = deserialize_result(payload)
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
    if config.classification.device == "cuda":
        if cuda_available():
            logger.info("CUDA detectada (%s)", cuda_device_label())
        else:
            logger.warning("device=cuda configurado mas CUDA não detectada; CPU onde aplicável")

    df = prepare_classification_df(clustered_categorical, config)
    feature_cols, _, weight_col = resolve_classification_columns(list(df.columns), config)
    if weight_col not in df.columns and "survey_weight" in df.columns:
        weight_col = "survey_weight"
    y = df["Serviço de Saúde"]
    weights = df[weight_col] if weight_col in df.columns else pd.Series(1.0, index=df.index)
    x = df[feature_cols]

    if config.classification.parallel and len(config.classification.models) > 1:
        model_results = _train_parallel(x, y, weights, config)
    else:
        model_results = _train_sequential(x, y, weights, config)

    return ClassificationStageResult(
        model_results=model_results,
        metrics=_metrics_from_results(model_results),
    )
