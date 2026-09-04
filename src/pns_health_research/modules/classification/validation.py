from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from pns_health_research.modules.classification.feature_selection import resolve_classification_columns
from pns_health_research.modules.classification.gridsearch import serialize_grid_result
from pns_health_research.modules.classification.importance import (
    compute_permutation_importance,
    extract_native_importance,
)
from pns_health_research.modules.classification.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
)
from pns_health_research.modules.classification.registry import CLASSIFIER_REGISTRY
from pns_health_research.modules.classification.tuning import effective_n_jobs, load_param_grid
from pns_health_research.config.models import ExperimentConfig
from pns_health_research.modules.preprocessing.encoding import merge_internacao_target, target_encode_legacy
from pns_health_research.modules.preprocessing.sampling import apply_nearmiss


@dataclass
class FoldMetrics:
    fold: int
    metrics: dict[str, Any]


@dataclass
class FoldSearchResult:
    fold: int
    best_params: dict[str, Any]
    best_score: float | None


@dataclass
class ModelResult:
    model_name: str
    fold_metrics: list[FoldMetrics]
    mean_metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    permutation_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    search_config: dict[str, Any] = field(default_factory=dict)
    cv_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_score: float | None = None
    best_index: int | None = None
    final_params: dict[str, Any] = field(default_factory=dict)
    fold_search_results: list[FoldSearchResult] = field(default_factory=list)
    confusion_matrix_sum: list[list[int]] = field(default_factory=list)


def prepare_classification_df(
    clustered_categorical: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    df = clustered_categorical.copy()
    primary = config.preprocessing.target.primary_column
    if primary in df.columns:
        pass
    elif config.preprocessing.merge_internacao_into_servico:
        df = merge_internacao_target(df)
    if config.preprocessing.target_encoding_before_split:
        _, te_cols, _ = resolve_classification_columns(list(df.columns), config)
        te_cols = [c for c in te_cols if c in df.columns]
        df = target_encode_legacy(df, te_cols, target_col="Serviço de Saúde")
    if config.preprocessing.nearmiss_before_split:
        df = apply_nearmiss(df, version=config.preprocessing.nearmiss_version)
    return df


def _predict_proba_safe(model: Any, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(x_test)
        except Exception:
            return None
    return None


def _record_fold(
    fold_idx: int,
    test_index: np.ndarray,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    model_name: str,
) -> tuple[FoldMetrics, list[dict[str, Any]]]:
    m = compute_classification_metrics(y_test.values, y_pred, y_proba)
    preds = []
    for i, idx in enumerate(test_index):
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
        preds.append(row)
    return FoldMetrics(fold=fold_idx, metrics=m), preds


def _sum_confusion_matrices(fold_metrics: list[FoldMetrics]) -> list[list[int]]:
    total = np.zeros((2, 2), dtype=int)
    for fm in fold_metrics:
        cm = fm.metrics.get("confusion_matrix")
        if cm:
            arr = np.array(cm, dtype=int)
            if arr.shape == (2, 2):
                total += arr
    return total.tolist()


def run_legacy_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    model_name: str,
    config: ExperimentConfig,
) -> ModelResult:
    clf_class = CLASSIFIER_REGISTRY[model_name]
    param_grid = load_param_grid(model_name, config=config)
    n_jobs = effective_n_jobs(
        model_name,
        parallel=config.classification.parallel,
        max_workers=config.classification.max_workers,
        config=config,
    )
    base_clf = clf_class({"n_jobs": n_jobs}, config.classification.device, model_name)
    skf = StratifiedKFold(
        n_splits=config.classification.n_splits,
        shuffle=True,
        random_state=config.experiment.seed,
    )

    fold_metrics: list[FoldMetrics] = []
    all_preds: list[dict[str, Any]] = []
    best_params: dict[str, Any] = {}
    fold_search: list[FoldSearchResult] = []
    last_grid: GridSearchCV | None = None
    last_final_model: Any = None
    last_x_test: pd.DataFrame | None = None
    last_y_test: pd.Series | None = None

    for fold_idx, (train_index, test_index) in enumerate(skf.split(x, y)):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        w_train = weights.iloc[train_index]

        x_tr, x_val, y_tr, y_val, w_tr, _w_val = train_test_split(
            x_train,
            y_train,
            w_train,
            test_size=config.classification.inner_test_size,
            stratify=y_train,
            random_state=config.experiment.seed,
        )

        grid: GridSearchCV | None = None
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
            fold_search.append(
                FoldSearchResult(
                    fold=fold_idx,
                    best_params=dict(grid.best_params_),
                    best_score=float(grid.best_score_),
                )
            )
            last_grid = grid
        else:
            best_model = base_clf
            fold_search.append(FoldSearchResult(fold=fold_idx, best_params={}, best_score=None))

        fit_kwargs: dict[str, Any] = {}
        if config.classification.use_sample_weight_in_fit:
            fit_kwargs["sample_weight"] = w_tr.values
        best_model.fit(x_tr, y_tr, **fit_kwargs)
        last_final_model = best_model
        last_x_test = x_test
        last_y_test = y_test

        y_pred = best_model.predict(x_test)
        y_proba = _predict_proba_safe(best_model, x_test)
        fm, preds = _record_fold(fold_idx, test_index, y_test, y_pred, y_proba, model_name)
        fold_metrics.append(fm)
        all_preds.extend(preds)

    mean_metrics = aggregate_fold_metrics([f.metrics for f in fold_metrics])
    grid_data = serialize_grid_result(
        last_grid,
        param_grid,
        scoring="f1_macro",
        cv_folds=config.classification.n_splits,
        random_state=config.experiment.seed,
        final_estimator=last_final_model,
    )

    native_imp = pd.DataFrame()
    perm_imp = pd.DataFrame()
    if last_final_model is not None and last_x_test is not None and last_y_test is not None:
        native = extract_native_importance(last_final_model, list(x.columns))
        if native is not None:
            native_imp = native
        try:
            perm_imp = compute_permutation_importance(
                last_final_model,
                last_x_test,
                last_y_test,
                random_state=config.experiment.seed,
            )
        except Exception:
            pass

    return ModelResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        best_params=best_params,
        predictions=pd.DataFrame(all_preds),
        feature_importance=native_imp,
        permutation_importance=perm_imp,
        search_config=grid_data["search"],
        cv_results=grid_data["cv_results"],
        best_score=grid_data["best_score"],
        best_index=grid_data["best_index"],
        final_params=grid_data["final_params"],
        fold_search_results=fold_search,
        confusion_matrix_sum=_sum_confusion_matrices(fold_metrics),
    )


def run_research_v2_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    model_name: str,
    config: ExperimentConfig,
) -> ModelResult:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import NearMiss

    clf_class = CLASSIFIER_REGISTRY[model_name]
    param_grid = load_param_grid(model_name, config=config)
    n_jobs = effective_n_jobs(
        model_name,
        parallel=config.classification.parallel,
        max_workers=config.classification.max_workers,
        config=config,
    )
    outer = StratifiedKFold(
        n_splits=config.classification.n_splits,
        shuffle=True,
        random_state=config.experiment.seed,
    )
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.experiment.seed)

    fold_metrics: list[FoldMetrics] = []
    all_preds: list[dict[str, Any]] = []
    fold_search: list[FoldSearchResult] = []
    last_grid: GridSearchCV | None = None
    last_final_model: Any = None
    last_x_test: pd.DataFrame | None = None
    last_y_test: pd.Series | None = None

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe_steps: list[tuple[str, Any]] = [
            ("sampler", NearMiss(version=config.preprocessing.nearmiss_version)),
            ("clf", clf_class({"n_jobs": n_jobs}, config.classification.device, model_name)),
        ]
        pipe = ImbPipeline(pipe_steps)
        prefixed_grid = {f"clf__{k}": v for k, v in param_grid.items()}
        grid: GridSearchCV | None = None
        if prefixed_grid:
            grid = GridSearchCV(pipe, prefixed_grid, cv=inner, scoring="f1_macro", n_jobs=n_jobs)
            # Pesos amostrais incompatíveis com NearMiss (tamanho muda após undersampling).
            grid.fit(x_train, y_train)
            best = grid.best_estimator_
            fold_search.append(
                FoldSearchResult(
                    fold=fold_idx,
                    best_params=dict(grid.best_params_),
                    best_score=float(grid.best_score_),
                )
            )
            last_grid = grid
        else:
            pipe.fit(x_train, y_train)
            best = pipe
            fold_search.append(FoldSearchResult(fold=fold_idx, best_params={}, best_score=None))

        last_final_model = best
        last_x_test = x_test
        last_y_test = y_test

        y_pred = best.predict(x_test)
        y_proba = _predict_proba_safe(best, x_test)
        fm, preds = _record_fold(fold_idx, test_idx, y_test, y_pred, y_proba, model_name)
        fold_metrics.append(fm)
        all_preds.extend(preds)

    mean_metrics = aggregate_fold_metrics([f.metrics for f in fold_metrics])
    clf_est = last_final_model.named_steps["clf"] if last_final_model else None
    grid_data = serialize_grid_result(
        last_grid,
        param_grid,
        scoring="f1_macro",
        cv_folds=3,
        random_state=config.experiment.seed,
        final_estimator=clf_est,
    )

    native_imp = pd.DataFrame()
    perm_imp = pd.DataFrame()
    if clf_est is not None and last_x_test is not None and last_y_test is not None:
        native = extract_native_importance(clf_est, list(x.columns))
        if native is not None:
            native_imp = native
        try:
            perm_imp = compute_permutation_importance(
                last_final_model,
                last_x_test,
                last_y_test,
                random_state=config.experiment.seed,
            )
        except Exception:
            pass

    return ModelResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        predictions=pd.DataFrame(all_preds),
        feature_importance=native_imp,
        permutation_importance=perm_imp,
        search_config=grid_data["search"],
        cv_results=grid_data["cv_results"],
        best_score=grid_data["best_score"],
        best_index=grid_data["best_index"],
        final_params=grid_data["final_params"],
        fold_search_results=fold_search,
        confusion_matrix_sum=_sum_confusion_matrices(fold_metrics),
    )
