from __future__ import annotations

from typing import Any

from pns_health_research.modules.classification.gpu.registry import register_gpu_classifier


@register_gpu_classifier("xgboost")
def create_xgboost_gpu(params: dict[str, Any]) -> Any:
    from xgboost import XGBClassifier

    params.setdefault("tree_method", "hist")
    params.setdefault("device", "cuda")
    return XGBClassifier(random_state=42, eval_metric="logloss", **params)
