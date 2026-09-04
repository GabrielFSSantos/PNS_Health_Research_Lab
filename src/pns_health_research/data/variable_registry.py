"""Carregamento do registro semântico de variáveis PNS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from pns_health_research.core.paths import CONFIGS_DIR

SemanticType = Literal[
    "binary",
    "ordinal",
    "count",
    "categorical",
    "derived",
    "target",
    "weight",
    "design",
]

DEFAULT_REGISTRY_PATH = CONFIGS_DIR / "data" / "variable_registry.yaml"


@dataclass(frozen=True)
class VariableSpec:
    raw_code: str
    label: str
    semantic_type: str
    stages: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class VariableRegistry:
    schema_version: str
    variables: dict[str, VariableSpec]

    def get_by_code(self, raw_code: str) -> VariableSpec | None:
        return self.variables.get(raw_code)

    def get_by_label(self, label: str) -> VariableSpec | None:
        for spec in self.variables.values():
            if spec.label == label:
                return spec
        return None

    def labels_for_type(self, semantic_type: str) -> list[str]:
        return [
            spec.label
            for spec in self.variables.values()
            if spec.semantic_type == semantic_type
        ]

    def required_raw_columns(self) -> list[str]:
        return list(self.variables.keys())


def load_variable_registry(
    path: str | Path | None = None,
) -> VariableRegistry:
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    with open(registry_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    variables: dict[str, VariableSpec] = {}
    for raw_code, spec in (raw.get("variables") or {}).items():
        if not isinstance(spec, dict):
            continue
        variables[str(raw_code)] = VariableSpec(
            raw_code=str(raw_code),
            label=str(spec.get("label", raw_code)),
            semantic_type=str(spec.get("semantic_type", "categorical")),
            stages=tuple(spec.get("stages") or ()),
        )

    return VariableRegistry(
        schema_version=str(raw.get("schema_version", "1.0")),
        variables=variables,
    )


def validate_raw_columns(
    raw_columns: list[str],
    registry: VariableRegistry | None = None,
) -> list[str]:
    """Retorna códigos PNS do registry ausentes no raw."""
    reg = registry or load_variable_registry()
    missing: list[str] = []
    for code in reg.required_raw_columns():
        if code not in raw_columns:
            missing.append(code)
    return missing
