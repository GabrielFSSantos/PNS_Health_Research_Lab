from __future__ import annotations

from dataclasses import dataclass, field
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from pns_health_research.classification.feature_selection import resolve_classification_columns
from pns_health_research.classification.registry import CLASSIFIER_REGISTRY
from pns_health_research.classification.tuning import effective_n_jobs, load_param_grid
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.preprocessing.encoding import merge_internacao_target, target_encode_legacy
from pns_health_research.preprocessing.sampling import apply_nearmiss


@dataclass
class FoldMetrics:
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray


@dataclass
class ModelResult:
    model_name: str
    fold_metrics: list[FoldMetrics]
    mean_metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def prepare_classification_df(
    clustered_categorical: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    df = clustered_categorical.copy()
    if config.preprocessing.merge_internacao_into_servico:
        df = merge_internacao_target(df)
    if config.preprocessing.target_encoding_before_split:
        _, te_cols, _ = resolve_classification_columns(list(df.columns), config)
        te_cols = [c for c in te_cols if c in df.columns]
        df = target_encode_legacy(df, te_cols, target_col="Serviço de Saúde")
    if config.preprocessing.nearmiss_before_split:
        df = apply_nearmiss(df, version=config.preprocessing.nearmiss_version)
    return df


def run_legacy_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    model_name: str,
    config: ExperimentConfig,
) -> ModelResult:
    clf_class = CLASSIFIER_REGISTRY[model_name]
    param_grid = load_param_grid(model_name)
    n_jobs = effective_n_jobs(
        model_name,
        parallel=config.classification.parallel,
        max_workers=config.classification.max_workers,
    )
    base_clf = clf_class({"n_jobs": n_jobs}, config.classification.device, model_name)
    n_splits = config.classification.n_splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.experiment.seed)

    fold_metrics: list[FoldMetrics] = []
    all_preds: list[dict[str, Any]] = []
    best_model = None
    best_params: dict[str, Any] = {}

    for fold_idx, (train_index, test_index) in enumerate(skf.split(x, y)):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        w_train = weights.iloc[train_index]

        x_tr, x_val, y_tr, y_val, w_tr, w_val = train_test_split(
            x_train,
            y_train,
            w_train,
            test_size=config.classification.inner_test_size,
            stratify=y_train,
            random_state=config.experiment.seed,
        )

        if param_grid:
            grid = GridSearchCV(
                estimator=base_clf,
                param_grid=param_grid,
                cv=skf,
                scoring="f1_macro",
                n_jobs=n_jobs,
            )
            grid.fit(x_val, y_val)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            best_model = base_clf
            best_params = {}

        fit_kwargs: dict[str, Any] = {}
        if config.classification.use_sample_weight_in_fit:
            fit_kwargs["sample_weight"] = w_tr.values
        best_model.fit(x_tr, y_tr, **fit_kwargs)

        y_pred = best_model.predict(x_test)
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
        for i, idx in enumerate(test_index):
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
        best_params=best_params,
        predictions=pd.DataFrame(all_preds),
    )


def run_research_v2_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    model_name: str,
    config: ExperimentConfig,
) -> ModelResult:
    """Nested CV: NearMiss and encoding only on training folds."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import NearMiss

    clf_class = CLASSIFIER_REGISTRY[model_name]
    param_grid = load_param_grid(model_name)
    n_jobs = effective_n_jobs(
        model_name,
        parallel=config.classification.parallel,
        max_workers=config.classification.max_workers,
    )
    outer = StratifiedKFold(
        n_splits=config.classification.n_splits,
        shuffle=True,
        random_state=config.experiment.seed,
    )
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.experiment.seed)

    fold_metrics: list[FoldMetrics] = []
    all_preds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = weights.iloc[train_idx]

        pipe_steps: list[tuple[str, Any]] = [
            ("sampler", NearMiss(version=config.preprocessing.nearmiss_version)),
            ("clf", clf_class({"n_jobs": n_jobs}, config.classification.device, model_name)),
        ]
        pipe = ImbPipeline(pipe_steps)

        prefixed_grid = {f"clf__{k}": v for k, v in param_grid.items()}
        if prefixed_grid:
            grid = GridSearchCV(
                pipe, prefixed_grid, cv=inner, scoring="f1_macro", n_jobs=n_jobs
            )
            fit_kwargs: dict[str, Any] = {}
            if config.classification.use_sample_weight_in_fit:
                fit_kwargs["clf__sample_weight"] = w_train.values
            grid.fit(x_train, y_train, **fit_kwargs)
            best = grid.best_estimator_
        else:
            fit_kwargs = {}
            if config.classification.use_sample_weight_in_fit:
                fit_kwargs["clf__sample_weight"] = w_train.values
            pipe.fit(x_train, y_train, **fit_kwargs)
            best = pipe
        y_pred = best.predict(x_test)
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
