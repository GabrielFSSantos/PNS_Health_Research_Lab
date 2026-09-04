"""Testes dos presets de pré-processamento."""

from __future__ import annotations

from pathlib import Path

import pytest

from pns_health_research.persistence.preprocess_presets import (
    PREPROCESS_PRESETS,
    resolve_prepare_args,
    resolve_preprocess_preset,
)


def test_preprocess_presets_mapping():
    assert PREPROCESS_PRESETS["pns"].name == "pns"
    assert PREPROCESS_PRESETS["pns"].profile_kind == "pns"
    assert PREPROCESS_PRESETS["legacy"].name == "legacy"
    assert PREPROCESS_PRESETS["legacy"].profile_kind == "legacy"


def test_resolve_preprocess_preset_unknown():
    with pytest.raises(ValueError, match="Perfil desconhecido"):
        resolve_preprocess_preset("invalid")


def test_resolve_prepare_args_profile_pns():
    config, name, kind = resolve_prepare_args(
        profile="pns",
        config="configs/other.yaml",
        name="ignored",
        dataset_id=None,
    )
    assert config == Path("configs/profiles/pns.yaml")
    assert name == "pns"
    assert kind == "pns"


def test_resolve_prepare_args_profile_legacy():
    config, name, kind = resolve_prepare_args(
        profile="legacy",
        config="configs/other.yaml",
        name="ignored",
        dataset_id=None,
    )
    assert config == Path("configs/profiles/legacy.yaml")
    assert name == "legacy"
    assert kind == "legacy"


def test_resolve_prepare_args_custom():
    config, name, kind = resolve_prepare_args(
        profile=None,
        config="configs/overrides/example.yaml",
        name="sabesp",
        dataset_id=None,
    )
    assert config == Path("configs/overrides/example.yaml")
    assert name == "sabesp"
    assert kind == "custom"


def test_resolve_prepare_args_dataset_id_escape_hatch():
    config, name, kind = resolve_prepare_args(
        profile=None,
        config="configs/profiles/pns.yaml",
        name="pns",
        dataset_id="custom_id",
    )
    assert name == "custom_id"
    assert kind == "custom"
