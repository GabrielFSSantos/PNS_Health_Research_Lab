"""Testes dos presets de classificação."""

from __future__ import annotations

from pathlib import Path

import pytest

from pns_health_research.persistence.classification_presets import (
    CLASSIFICATION_PRESETS,
    resolve_classification_args,
    resolve_classification_preset,
)


def test_classification_presets_mapping():
    assert CLASSIFICATION_PRESETS["pns"].name == "pns"
    assert CLASSIFICATION_PRESETS["pns"].source_base == "pns"
    assert CLASSIFICATION_PRESETS["pns"].profile_kind == "pns"

    assert CLASSIFICATION_PRESETS["legacy"].name == "legacy"
    assert CLASSIFICATION_PRESETS["legacy"].source_base == "legacy"
    assert CLASSIFICATION_PRESETS["legacy"].profile_kind == "legacy"


def test_resolve_classification_preset_unknown():
    with pytest.raises(ValueError, match="Perfil desconhecido"):
        resolve_classification_preset("invalid")


def test_resolve_classification_args_profile_pns():
    config, name, src, kind = resolve_classification_args(
        profile="pns",
        config="configs/other.yaml",
        name="ignored",
        source_base=None,
    )
    assert config == Path("configs/profiles/pns.yaml")
    assert name == "pns"
    assert src == "pns"
    assert kind == "pns"


def test_resolve_classification_args_profile_legacy():
    config, name, src, kind = resolve_classification_args(
        profile="legacy",
        config="configs/other.yaml",
        name="ignored",
        source_base=None,
    )
    assert config == Path("configs/profiles/legacy.yaml")
    assert name == "legacy"
    assert src == "legacy"
    assert kind == "legacy"


def test_resolve_classification_args_custom():
    config, name, src, kind = resolve_classification_args(
        profile=None,
        config="configs/overrides/example.yaml",
        name="sabesp",
        source_base="sabesp",
    )
    assert config == Path("configs/overrides/example.yaml")
    assert name == "sabesp"
    assert src == "sabesp"
    assert kind == "custom"
