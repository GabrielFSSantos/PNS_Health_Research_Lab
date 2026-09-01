"""Métricas de classificação (binária e multiclasse)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    positive_label: int = 1,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_positive": float(
            precision_score(
                y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0
            )
        ),
        "recall_positive": float(
            recall_score(
                y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0
            )
        ),
        "f1_positive": float(
            f1_score(y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0)
        ),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            metrics["pr_auc"] = float(average_precision_score(y_true, proba_pos))
            metrics["brier_score"] = float(brier_score_loss(y_true, proba_pos))
        except ValueError:
            pass
    # compat legacy reproduction report
    metrics["precision"] = metrics["precision_macro"]
    metrics["recall"] = metrics["recall_macro"]
    metrics["f1"] = metrics["f1_macro"]
    return metrics


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    keys = fold_metrics[0].keys() if fold_metrics else []
    out: dict[str, float] = {}
    for k in keys:
        if k == "confusion_matrix":
            continue
        vals = [m[k] for m in fold_metrics if k in m and isinstance(m[k], (int, float))]
        if vals:
            out[k] = float(np.mean(vals))
    return out
