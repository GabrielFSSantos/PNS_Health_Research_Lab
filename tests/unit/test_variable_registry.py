"""Testes do variable_registry."""

from __future__ import annotations

from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.data.variable_registry import (
    load_variable_registry,
    validate_raw_columns,
)


def test_load_variable_registry():
    reg = load_variable_registry(PROJECT_ROOT / "configs/data/variable_registry.yaml")
    assert reg.schema_version == "1.0"
    j014 = reg.get_by_code("J014")
    assert j014 is not None
    assert j014.label == "Serviço de Saúde"
    assert j014.semantic_type == "binary"
    assert "encode_binary" in j014.stages


def test_get_by_label():
    reg = load_variable_registry()
    spec = reg.get_by_label("Consultas 12m")
    assert spec is not None
    assert spec.raw_code == "J012"
    assert spec.semantic_type == "count"


def test_validate_raw_columns():
    reg = load_variable_registry()
    missing = validate_raw_columns(["C008", "J014"], registry=reg)
    assert "J038" in missing
    assert "J014" not in missing
