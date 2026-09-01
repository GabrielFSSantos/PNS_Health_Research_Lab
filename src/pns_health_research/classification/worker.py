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
    return {
        "status": "ok",
        "model_name": result.model_name,
        "mean_metrics": result.mean_metrics,
        "best_params": result.best_params,
        "predictions": result.predictions.to_dict(orient="records"),
    }


def deserialize_result(data: dict[str, Any]) -> ModelResult | dict[str, Any]:
    if data.get("status") == "failed":
        return data
    import pandas as pd

    from pns_health_research.classification.validation import ModelResult

    return ModelResult(
        model_name=data["model_name"],
        fold_metrics=[],
        mean_metrics=data["mean_metrics"],
        best_params=data.get("best_params", {}),
        predictions=pd.DataFrame(data.get("predictions", [])),
    )
