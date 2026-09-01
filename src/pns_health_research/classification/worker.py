from __future__ import annotations

from typing import Any

from pns_health_research.classification.validation import (
    ModelResult,
    run_legacy_cv,
    run_research_v2_cv,
)
from pns_health_research.config.models import ExperimentConfig


def train_one_model(
    model_name: str,
    x_serialized: dict[str, Any],
    y_values: list,
    weight_values: list,
    config_dict: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Picklable worker for ProcessPoolExecutor."""
    import pandas as pd

    config = ExperimentConfig.model_validate(config_dict)
    x = pd.DataFrame(x_serialized)
    y = pd.Series(y_values, name="Serviço de Saúde")
    weights = pd.Series(weight_values, index=x.index)

    try:
        if (
            config.experiment.profile == "research_v2"
            and config.classification.backend == "gpu"
        ):
            from pns_health_research.classification.gpu.pipelines import run_gpu_cv

            result = run_gpu_cv(x, y, weights, model_name, config)
        elif config.experiment.profile == "research_v2":
            result = run_research_v2_cv(x, y, weights, model_name, config)
        else:
            result = run_legacy_cv(x, y, weights, model_name, config)
        return model_name, _serialize_result(result)
    except Exception as exc:
        return model_name, {"status": "failed", "error": str(exc)}


def _serialize_result(result: ModelResult) -> dict[str, Any]:
    fold_metrics = [
        {"fold": fm.fold, "metrics": fm.metrics} for fm in result.fold_metrics
    ]
    fold_search = [
        {"fold": fs.fold, "best_params": fs.best_params, "best_score": fs.best_score}
        for fs in result.fold_search_results
    ]
    payload: dict[str, Any] = {
        "status": "ok",
        "model_name": result.model_name,
        "mean_metrics": result.mean_metrics,
        "best_params": result.best_params,
        "predictions": result.predictions.to_dict(orient="records"),
        "fold_metrics": fold_metrics,
        "fold_search_results": fold_search,
        "search_config": result.search_config,
        "best_score": result.best_score,
        "best_index": result.best_index,
        "final_params": result.final_params,
        "confusion_matrix_sum": result.confusion_matrix_sum,
    }
    if not result.cv_results.empty:
        payload["cv_results"] = result.cv_results.to_dict(orient="list")
    if not result.feature_importance.empty:
        payload["feature_importance"] = result.feature_importance.to_dict(orient="records")
    if not result.permutation_importance.empty:
        payload["permutation_importance"] = result.permutation_importance.to_dict(
            orient="records"
        )
    return payload


def deserialize_result(data: dict[str, Any]) -> ModelResult | dict[str, Any]:
    if data.get("status") == "failed":
        return data
    import pandas as pd

    from pns_health_research.classification.validation import FoldMetrics, FoldSearchResult

    fold_metrics = [
        FoldMetrics(fold=fm["fold"], metrics=fm["metrics"])
        for fm in data.get("fold_metrics", [])
    ]
    fold_search = [
        FoldSearchResult(
            fold=fs["fold"],
            best_params=fs.get("best_params", {}),
            best_score=fs.get("best_score"),
        )
        for fs in data.get("fold_search_results", [])
    ]
    cv_results = pd.DataFrame(data.get("cv_results", {}))
    feature_importance = pd.DataFrame(data.get("feature_importance", []))
    permutation_importance = pd.DataFrame(data.get("permutation_importance", []))

    return ModelResult(
        model_name=data["model_name"],
        fold_metrics=fold_metrics,
        mean_metrics=data["mean_metrics"],
        best_params=data.get("best_params", {}),
        predictions=pd.DataFrame(data.get("predictions", [])),
        feature_importance=feature_importance,
        permutation_importance=permutation_importance,
        search_config=data.get("search_config", {}),
        cv_results=cv_results,
        best_score=data.get("best_score"),
        best_index=data.get("best_index"),
        final_params=data.get("final_params", {}),
        fold_search_results=fold_search,
        confusion_matrix_sum=data.get("confusion_matrix_sum", []),
    )
