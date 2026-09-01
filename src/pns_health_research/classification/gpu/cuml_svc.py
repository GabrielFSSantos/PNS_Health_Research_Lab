from __future__ import annotations

from typing import Any

from pns_health_research.classification.gpu.registry import register_gpu_classifier


@register_gpu_classifier("svc")
def create_cuml_svc(params: dict[str, Any]) -> Any:
    from cuml.svm import SVC

    return SVC(random_state=42, **params)
