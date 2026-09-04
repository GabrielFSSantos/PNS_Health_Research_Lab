"""Conversão schema 2.x (stages flat) → schema 3.0 (pipeline ordenado)."""

from __future__ import annotations

import logging
from typing import Any

from pns_health_research.config.models import PipelineStage, PreprocessingStages

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_ORDER: tuple[str, ...] = (
    "select_columns",
    "adult_filter",
    "encode_binary",
    "gestational_reinforce",
    "attach_weights",
    "drop_design_columns",
    "derive_binary",
    "encode_categorical",
    "build_targets",
    "missing_audit",
    "impute_missing",
    "finalize_outputs",
)

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "adult_filter": ("select_columns",),
    "gestational_reinforce": ("encode_binary",),
    "attach_weights": ("gestational_reinforce",),
    "drop_design_columns": ("attach_weights",),
    "derive_binary": ("drop_design_columns",),
    "encode_categorical": ("drop_design_columns",),
    "build_targets": ("encode_binary",),
    "missing_audit": ("encode_binary",),
    "impute_missing": ("encode_binary",),
    "finalize_outputs": ("encode_binary",),
}


def _stage_params(stage_id: str, stages: PreprocessingStages) -> dict[str, Any]:
    if stage_id == "adult_filter":
        return {
            "min_age": stages.adult_filter.min_age,
            "require_imc_inputs": stages.adult_filter.require_imc_inputs,
        }
    if stage_id == "gestational_reinforce":
        return {"impute_age_by_target": stages.impute_age_by_target.enabled}
    if stage_id == "derive_binary":
        return {"one_hot": stages.one_hot_and_numeric.enabled}
    if stage_id == "encode_categorical":
        return {"one_hot": stages.one_hot_and_numeric.enabled}
    if stage_id == "impute_missing":
        return {
            "binary": stages.impute_missing.binary,
            "categorical": stages.impute_missing.categorical,
        }
    return {}


def _stage_enabled(stage_id: str, stages: PreprocessingStages) -> bool:
    mapping: dict[str, bool] = {
        "select_columns": stages.select_columns.enabled,
        "adult_filter": stages.adult_filter.enabled,
        "encode_binary": True,
        "gestational_reinforce": True,
        "attach_weights": True,
        "drop_design_columns": True,
        "derive_binary": stages.derive_features.enabled,
        "encode_categorical": stages.one_hot_and_numeric.enabled,
        "build_targets": True,
        "missing_audit": True,
        "impute_missing": stages.impute_missing.enabled,
        "finalize_outputs": True,
    }
    return mapping.get(stage_id, True)


def flat_stages_to_pipeline(stages: PreprocessingStages) -> list[PipelineStage]:
    """Converte toggles flat 2.x para lista ordenada schema 3.0."""
    logger.warning(
        "preprocessing schema 2.x: usando adapter flat stages → pipeline. "
        "Prefira schema_version 3.0 com pipeline explícito."
    )
    return [
        PipelineStage(
            id=stage_id,
            enabled=_stage_enabled(stage_id, stages),
            params=_stage_params(stage_id, stages),
        )
        for stage_id in DEFAULT_PIPELINE_ORDER
    ]


def apply_flat_stage_overrides(
    pipeline: list[PipelineStage],
    stages: PreprocessingStages,
) -> list[PipelineStage]:
    """Aplica toggles flat (overlays de experimento) sobre pipeline 3.0."""
    out: list[PipelineStage] = []
    for step in pipeline:
        updated = step.model_copy()
        if step.id == "select_columns":
            updated.enabled = stages.select_columns.enabled
        elif step.id == "adult_filter":
            updated.enabled = stages.adult_filter.enabled
            params = dict(updated.params)
            params["min_age"] = stages.adult_filter.min_age
            params["require_imc_inputs"] = stages.adult_filter.require_imc_inputs
            updated.params = params
        elif step.id == "derive_binary":
            updated.enabled = stages.derive_features.enabled
            params = dict(updated.params)
            params["one_hot"] = stages.one_hot_and_numeric.enabled
            updated.params = params
        elif step.id == "encode_categorical":
            updated.enabled = stages.one_hot_and_numeric.enabled
            params = dict(updated.params)
            params["one_hot"] = stages.one_hot_and_numeric.enabled
            updated.params = params
        elif step.id == "gestational_reinforce":
            params = dict(updated.params)
            params["impute_age_by_target"] = stages.impute_age_by_target.enabled
            updated.params = params
        elif step.id == "impute_missing":
            updated.enabled = stages.impute_missing.enabled
            params = dict(updated.params)
            params["binary"] = stages.impute_missing.binary
            params["categorical"] = stages.impute_missing.categorical
            updated.params = params
        out.append(updated)
    return out


def merge_pipeline_with_stages(
    pipeline: list[PipelineStage],
    stages: PreprocessingStages,
) -> list[PipelineStage]:
    return apply_flat_stage_overrides(pipeline, stages)


def validate_pipeline(pipeline: list[PipelineStage]) -> None:
    """Falha cedo se estágio habilitado depende de outro desabilitado."""
    enabled_ids = {s.id for s in pipeline if s.enabled}
    known = set(DEFAULT_PIPELINE_ORDER)
    for step in pipeline:
        if step.id not in known:
            raise ValueError(f"Estágio de pré-processamento desconhecido: {step.id!r}")
    for step in pipeline:
        if not step.enabled:
            continue
        for req in STAGE_DEPENDENCIES.get(step.id, ()):
            if req not in enabled_ids:
                raise ValueError(
                    f"Estágio {step.id!r} requer {req!r} habilitado no pipeline"
                )
