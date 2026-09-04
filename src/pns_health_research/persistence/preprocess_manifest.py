"""Helpers compartilhados para manifests de pré-processamento."""

from __future__ import annotations

from pns_health_research.config.models import ExperimentConfig


def methodological_warnings(config: ExperimentConfig) -> list[str]:
    warnings: list[str] = []
    pp = config.preprocessing
    adult = pp.pipeline_step("adult_filter")
    require_imc = (
        adult.params.get("require_imc_inputs", True)
        if adult is not None
        else pp.stages.adult_filter.require_imc_inputs
    )
    if require_imc:
        warnings.append(
            "PRE-05: filtro exige peso e altura (IMC); amostra analítica menor que população adulta."
        )
    derive = pp.pipeline_step("derive_binary")
    derive_on = derive.enabled if derive is not None else pp.stages.derive_features.enabled
    if derive_on:
        warnings.append(
            "PRE-08: comorbidades gestacionais fundidas em Diabetes/Hipertensão."
        )
    if pp.binary_encoding.policy == "explicit_missing":
        warnings.append(
            "PRE-01: comorbidades com -1 (ausente explícito); modelos podem exigir imputação por fold."
        )
    return warnings


def audit_resolved_for_profile(config: ExperimentConfig) -> list[str]:
    if config.preprocessing.binary_encoding.policy == "explicit_missing":
        return [
            "PRE-01",
            "PRE-02",
            "PRE-03",
            "PRE-04",
            "PRE-06",
            "PRE-07",
            "DOC-01",
            "DOC-02",
        ]
    return []


def preprocessing_config_path(config: ExperimentConfig) -> str | None:
    cfg_files = config.configuration_files
    if cfg_files is None:
        return None
    pp_path = cfg_files.preprocessing
    if pp_path is False or pp_path is None:
        return None
    return str(pp_path)
