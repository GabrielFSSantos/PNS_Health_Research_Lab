from __future__ import annotations

from typing import Any

from pns_health_research.classification.gpu.registry import register_gpu_classifier


@register_gpu_classifier("random_forest")
def create_cuml_rf(params: dict[str, Any]) -> Any:
    from cuml.ensemble import RandomForestClassifier

    return RandomForestClassifier(random_state=42, **params)
