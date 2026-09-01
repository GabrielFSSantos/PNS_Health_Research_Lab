from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from pns_health_research.classification.gpu import (  # noqa: F401
    cuml_rf,
    cuml_svc,
    torch_mlp,
    xgboost_gpu,
)
from pns_health_research.classification.gpu.registry import GPU_CLASSIFIER_REGISTRY
from pns_health_research.classification.gpu.torch_mlp import run_torch_mlp_cv
from pns_health_research.classification.tuning import load_param_grid
from pns_health_research.classification.validation import FoldMetrics, ModelResult
from pns_health_research.config.models import ExperimentConfig


def _gpu_available() -> bool:
    try:
        import cuml  # noqa: F401

        return True
    except ImportError:
        return False


def run_gpu_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    model_name: str,
    config: ExperimentConfig,
) -> ModelResult:
    if model_name == "mlp":
        return run_torch_mlp_cv(x, y, weights, config)

    if model_name not in GPU_CLASSIFIER_REGISTRY:
        raise ValueError(f"Modelo GPU não registrado: {model_name}")

    if model_name in ("random_forest", "svc") and not _gpu_available():
        raise ImportError(
            "cuML não instalado. Instale com: pip install 'pns-health-research[gpu]' "
            "(Python 3.10–3.12, CUDA compatível)."
        )

    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import NearMiss

    factory = GPU_CLASSIFIER_REGISTRY[model_name]
    param_grid = load_param_grid(model_name)
    outer = StratifiedKFold(
        n_splits=config.classification.n_splits,
        shuffle=True,
        random_state=config.experiment.seed,
    )
    fold_metrics: list[FoldMetrics] = []
    all_preds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = ImbPipeline(
            [
                ("sampler", NearMiss(version=config.preprocessing.nearmiss_version)),
                ("clf", factory(param_grid or {})),
            ]
        )
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        y_pred = np.asarray(y_pred).ravel()
        cm = confusion_matrix(y_test, y_pred)
        fold_metrics.append(
            FoldMetrics(
                fold=fold_idx,
                accuracy=float(accuracy_score(y_test, y_pred)),
                precision=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                recall=float(recall_score(y_test, y_pred, average="macro")),
                f1=float(f1_score(y_test, y_pred, average="macro")),
                confusion=cm,
            )
        )
        for i, idx in enumerate(test_idx):
            all_preds.append(
                {
                    "record_id": str(idx),
                    "fold": fold_idx,
                    "true_label": int(y_test.iloc[i]),
                    "predicted_label": int(y_pred[i]),
                    "model": model_name,
                    "dataset_partition": "outer_test",
                }
            )

    mean_metrics = {
        "accuracy": float(np.mean([m.accuracy for m in fold_metrics])),
        "precision": float(np.mean([m.precision for m in fold_metrics])),
        "recall": float(np.mean([m.recall for m in fold_metrics])),
        "f1": float(np.mean([m.f1 for m in fold_metrics])),
    }
    return ModelResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        predictions=pd.DataFrame(all_preds),
    )
