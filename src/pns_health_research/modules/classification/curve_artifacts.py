"""Persistência de curvas ROC/PR/calibração a partir de predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from pns_health_research.modules.classification.validation import ModelResult
from pns_health_research.persistence.artifact_store import ArtifactStore


def _curve_from_predictions(
    predictions: pd.DataFrame,
) -> dict[str, Any] | None:
    if predictions.empty or "proba_positive" not in predictions.columns:
        return None
    y_true = predictions["true_label"].values
    y_score = predictions["proba_positive"].values
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
    return {
        "roc": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "auc": float(auc(fpr, tpr)),
        },
        "pr": {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "auc": float(auc(recall, precision)),
        },
        "calibration": {
            "fraction_of_positives": [float(x) for x in prob_true],
            "mean_predicted_value": [float(x) for x in prob_pred],
        },
    }


def persist_curve_artifacts(
    store: ArtifactStore,
    model_results: dict[str, ModelResult | dict[str, Any]],
) -> None:
    for model_name, res in model_results.items():
        if not isinstance(res, ModelResult) or res.predictions.empty:
            continue
        curves = _curve_from_predictions(res.predictions)
        if curves:
            store.write_json(f"metrics/roc_curves/{model_name}.json", curves)


def aggregate_predictions_summary(
    model_results: dict[str, ModelResult | dict[str, Any]],
) -> dict[str, Any]:
    from pns_health_research.modules.classification.persistence import predictions_summary

    out: dict[str, Any] = {}
    for model_name, res in model_results.items():
        if isinstance(res, ModelResult):
            out[model_name] = predictions_summary(res.predictions)
    return out
