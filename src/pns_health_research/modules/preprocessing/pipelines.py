"""Pipeline de pré-processamento — fachada pública."""

from __future__ import annotations

from pns_health_research.modules.preprocessing.contract import PreprocessingResult
from pns_health_research.modules.preprocessing.pipeline_runner import run_preprocessing_pipeline

# Reexport para compatibilidade de imports existentes
__all__ = ["PreprocessingResult", "run_preprocessing"]


def run_preprocessing(config, raw_path: str | None = None) -> PreprocessingResult:
    return run_preprocessing_pipeline(config, raw_path=raw_path)
