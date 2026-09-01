from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from pns_health_research.core.gpu import cuda_available

ClassifierFactory = Callable[[dict[str, Any], Literal["cpu", "cuda"], str], Any]

CLASSIFIER_REGISTRY: dict[str, ClassifierFactory] = {}


def register_classifier(name: str) -> Callable[[ClassifierFactory], ClassifierFactory]:
    def decorator(fn: ClassifierFactory) -> ClassifierFactory:
        CLASSIFIER_REGISTRY[name] = fn
        return fn

    return decorator


def _cuda_available() -> bool:
    return cuda_available()


@register_classifier("random_forest")
def create_random_forest(
    params: dict[str, Any],
    device: Literal["cpu", "cuda"] = "cpu",
    model_name: str = "random_forest",
) -> RandomForestClassifier:
    n_jobs = params.pop("n_jobs", -1)
    return RandomForestClassifier(random_state=42, n_jobs=n_jobs, **params)


@register_classifier("svc")
def create_svc(
    params: dict[str, Any],
    device: Literal["cpu", "cuda"] = "cpu",
    model_name: str = "svc",
) -> SVC:
    params.pop("n_jobs", None)
    return SVC(random_state=42, **params)


@register_classifier("mlp")
def create_mlp(
    params: dict[str, Any],
    device: Literal["cpu", "cuda"] = "cpu",
    model_name: str = "mlp",
) -> MLPClassifier:
    params.pop("n_jobs", None)
    return MLPClassifier(random_state=42, max_iter=500, **params)


@register_classifier("xgboost")
def create_xgboost(
    params: dict[str, Any],
    device: Literal["cpu", "cuda"] = "cpu",
    model_name: str = "xgboost",
) -> XGBClassifier:
    use_cuda = device == "cuda" and _cuda_available()
    if use_cuda:
        import logging

        from pns_health_research.core.gpu import cuda_device_label

        logging.getLogger("pns_health_research").info(
            "XGBoost usando GPU (%s)", cuda_device_label()
        )
    elif device == "cuda":
        import logging

        logging.getLogger("pns_health_research").warning(
            "CUDA indisponível; XGBoost usará CPU."
        )
    params.pop("n_jobs", None)
    if use_cuda:
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda")
    return XGBClassifier(random_state=42, eval_metric="logloss", **params)
