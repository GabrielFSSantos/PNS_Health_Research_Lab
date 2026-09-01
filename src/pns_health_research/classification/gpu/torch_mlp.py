from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pns_health_research.classification.gpu.registry import register_gpu_classifier
from pns_health_research.classification.gpu.search import gpu_search_config
from pns_health_research.classification.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
)
from pns_health_research.classification.tuning import load_param_grid
from pns_health_research.classification.validation import FoldMetrics, ModelResult
from pns_health_research.config.models import ExperimentConfig


class TorchMLPClassifier:
    """Minimal PyTorch MLP wrapper with sklearn-like fit/predict."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        max_iter: int = 200,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> TorchMLPClassifier:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        x_arr = self._scaler.fit_transform(np.asarray(x, dtype=np.float32))
        y_arr = np.asarray(y, dtype=np.int64)
        n_features = x_arr.shape[1]
        n_classes = len(np.unique(y_arr))

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in self.hidden_layer_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        net = nn.Sequential(*layers)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        x_t = torch.from_numpy(x_arr)
        y_t = torch.from_numpy(y_arr)
        net.train()
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            loss = criterion(net(x_t), y_t)
            loss.backward()
            optimizer.step()
        self._model = net
        return self

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        import torch

        assert self._model is not None
        x_arr = self._scaler.transform(np.asarray(x, dtype=np.float32))
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.from_numpy(x_arr))
            return logits.argmax(dim=1).numpy()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }


@register_gpu_classifier("mlp")
def create_torch_mlp(params: dict[str, Any]) -> TorchMLPClassifier:
    hidden = params.get("hidden_layer_sizes", (100,))
    if isinstance(hidden, list):
        hidden = tuple(hidden)
    return TorchMLPClassifier(
        hidden_layer_sizes=hidden,
        max_iter=int(params.get("max_iter", 200)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        random_state=int(params.get("random_state", 42)),
    )


def run_torch_mlp_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    config: ExperimentConfig,
) -> ModelResult:
    """Nested CV for PyTorch MLP (research_v2 GPU path)."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import NearMiss

    param_grid = load_param_grid("mlp", config=config)
    outer = StratifiedKFold(
        n_splits=config.classification.n_splits,
        shuffle=True,
        random_state=config.experiment.seed,
    )
    fold_metrics: list[FoldMetrics] = []
    all_preds: list[dict[str, Any]] = []
    last_pipe: Any = None

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = ImbPipeline(
            [
                ("sampler", NearMiss(version=config.preprocessing.nearmiss_version)),
                ("clf", create_torch_mlp(param_grid or {})),
            ]
        )
        pipe.fit(x_train, y_train)
        last_pipe = pipe
        y_pred = pipe.predict(x_test)
        m = compute_classification_metrics(y_test.values, y_pred, None)
        fold_metrics.append(FoldMetrics(fold=fold_idx, metrics=m))
        for i, idx in enumerate(test_idx):
            all_preds.append(
                {
                    "record_id": str(idx),
                    "fold": fold_idx,
                    "true_label": int(y_test.iloc[i]),
                    "predicted_label": int(y_pred[i]),
                    "model": "mlp",
                    "dataset_partition": "outer_test",
                }
            )

    mean_metrics = aggregate_fold_metrics([f.metrics for f in fold_metrics])
    clf = last_pipe.named_steps["clf"] if last_pipe else None
    final_params = dict(clf.get_params()) if clf else {}
    search = gpu_search_config(
        "mlp",
        param_grid,
        cv_folds=config.classification.n_splits,
        random_state=config.experiment.seed,
    )

    return ModelResult(
        model_name="mlp",
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        search_config=search,
        final_params=final_params,
        predictions=pd.DataFrame(all_preds),
    )
