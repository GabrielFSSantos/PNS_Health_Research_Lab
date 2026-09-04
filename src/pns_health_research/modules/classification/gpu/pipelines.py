from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pns_health_research.modules.classification.gpu import (  # noqa: F401
    cuml_rf,
    cuml_svc,
    torch_mlp,
    xgboost_gpu,
)
from pns_health_research.modules.classification.gpu.registry import GPU_CLASSIFIER_REGISTRY
from pns_health_research.modules.classification.gpu.search import gpu_search_config
from pns_health_research.modules.classification.gpu.torch_mlp import run_torch_mlp_cv
from pns_health_research.modules.classification.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
)
from pns_health_research.modules.classification.tuning import load_param_grid
from pns_health_research.modules.classification.validation import FoldMetrics, ModelResult
from pns_health_research.config.models import ExperimentConfig


def _gpu_available() -> bool:
    try:
        import cuml  # noqa: F401

        return True
    except ImportError:
        return False


def _predict_proba_safe(model: Any, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(x_test)
        except Exception:
            return None
    return None


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
    param_grid = load_param_grid(model_name, config=config)
    outer = StratifiedKFold(
        n_splits=config.classification.n_splits,
        shuffle=True,
        random_state=config.experiment.seed,
    )
    fold_metrics: list[FoldMetrics] = []
    all_preds: list[dict[str, Any]] = []
    last_model: Any = None

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
        last_model = pipe
        y_pred = np.asarray(pipe.predict(x_test)).ravel()
        y_proba = _predict_proba_safe(pipe, x_test)
        m = compute_classification_metrics(y_test.values, y_pred, y_proba)
        fold_metrics.append(FoldMetrics(fold=fold_idx, metrics=m))
        for i, idx in enumerate(test_idx):
            row: dict[str, Any] = {
                "record_id": str(idx),
                "fold": fold_idx,
                "true_label": int(y_test.iloc[i]),
                "predicted_label": int(y_pred[i]),
                "model": model_name,
                "dataset_partition": "outer_test",
            }
            if y_proba is not None:
                row["proba_positive"] = float(
                    y_proba[i, 1] if y_proba.ndim > 1 else y_proba[i]
                )
            all_preds.append(row)

    mean_metrics = aggregate_fold_metrics([f.metrics for f in fold_metrics])
    clf = last_model.named_steps["clf"] if last_model else None
    final_params = dict(clf.get_params()) if clf and hasattr(clf, "get_params") else {}
    search = gpu_search_config(
        model_name,
        param_grid,
        cv_folds=config.classification.n_splits,
        random_state=config.experiment.seed,
    )

    return ModelResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        search_config=search,
        final_params=final_params,
        predictions=pd.DataFrame(all_preds),
    )
