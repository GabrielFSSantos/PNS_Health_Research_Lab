"""Contrato de estado e resultado do pipeline de pré-processamento."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd

from pns_health_research.config.models import PreprocessingConfig

OutputFrame = Literal["binary", "categorical"]


@dataclass
class ColumnSpec:
    name: str
    output_frame: OutputFrame
    semantic_type: str = "categorical"
    source_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessingContext:
    """Estado mutável compartilhado entre estágios do pipeline."""

    pp: PreprocessingConfig
    experiment_profile: str
    raw: pd.DataFrame
    renamed: pd.DataFrame | None = None
    filtered: pd.DataFrame | None = None
    raw_for_audit: pd.DataFrame | None = None
    binary_work: pd.DataFrame | None = None
    categorical_work: pd.DataFrame | None = None
    filter_stats: dict[str, Any] = field(default_factory=dict)
    processing_stats: dict[str, Any] = field(default_factory=dict)
    survey_snapshot: dict[str, Any] = field(default_factory=dict)
    missing_audit: dict[str, Any] | None = None
    bin_imputation: dict[str, Any] = field(default_factory=dict)
    cat_imputation: dict[str, Any] = field(default_factory=dict)
    binary_numeric: pd.DataFrame | None = None
    categorical_numeric: pd.DataFrame | None = None
    pipeline_executed: list[dict[str, Any]] = field(default_factory=list)
    column_catalog: list[ColumnSpec] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    binary_df: pd.DataFrame
    categorical_df: pd.DataFrame
    metrics: dict[str, Any]
    filter_stats: dict[str, Any] = field(default_factory=dict)
    processing_stats: dict[str, Any] = field(default_factory=dict)
    survey_snapshot: dict[str, Any] = field(default_factory=dict)
    column_catalog: list[ColumnSpec] = field(default_factory=list)
