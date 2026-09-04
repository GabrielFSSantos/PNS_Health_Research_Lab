"""Testes dos presets de agrupamento."""

from __future__ import annotations

from pathlib import Path

import pytest

from pns_health_research.persistence.clustering_presets import (
    CLUSTERING_PRESETS,
    resolve_clustering_args,
    resolve_clustering_preset,
)


def test_clustering_presets_mapping():
    assert CLUSTERING_PRESETS["pns"].name == "pns"
    assert CLUSTERING_PRESETS["pns"].source_base == "pns"
    assert CLUSTERING_PRESETS["pns"].profile_kind == "pns"

    assert CLUSTERING_PRESETS["legacy"].name == "legacy"
    assert CLUSTERING_PRESETS["legacy"].source_base == "legacy"
    assert CLUSTERING_PRESETS["legacy"].profile_kind == "legacy"


def test_resolve_clustering_preset_unknown():
    with pytest.raises(ValueError, match="Perfil desconhecido"):
        resolve_clustering_preset("invalid")


def test_resolve_clustering_args_profile_pns():
    config, name, src, kind = resolve_clustering_args(
        profile="pns",
        config="configs/other.yaml",
        name="ignored",
        source_base=None,
    )
    assert config == Path("configs/profiles/pns.yaml")
    assert name == "pns"
    assert src == "pns"
    assert kind == "pns"


def test_resolve_clustering_args_profile_legacy():
    config, name, src, kind = resolve_clustering_args(
        profile="legacy",
        config="configs/other.yaml",
        name="ignored",
        source_base=None,
    )
    assert config == Path("configs/profiles/legacy.yaml")
    assert name == "legacy"
    assert src == "legacy"
    assert kind == "legacy"


def test_resolve_clustering_args_custom():
    config, name, src, kind = resolve_clustering_args(
        profile=None,
        config="configs/overrides/example.yaml",
        name="sabesp",
        source_base="sabesp",
    )
    assert config == Path("configs/overrides/example.yaml")
    assert name == "sabesp"
    assert src == "sabesp"
    assert kind == "custom"


def test_resolve_clustering_args_custom_default_source():
    config, name, src, kind = resolve_clustering_args(
        profile=None,
        config="configs/profiles/pns.yaml",
        name="pns",
        source_base=None,
    )
    assert name == "pns"
    assert src == "pns"
    assert kind == "custom"
