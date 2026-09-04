"""Estágios plugáveis do pipeline de pré-processamento."""

from pns_health_research.modules.preprocessing.stages.registry import (
    STAGE_REGISTRY,
    register_stage,
    run_registered_stages,
)

__all__ = ["STAGE_REGISTRY", "register_stage", "run_registered_stages"]
