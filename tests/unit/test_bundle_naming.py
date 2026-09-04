"""Testes de resolução de nomes de base."""

from __future__ import annotations

from pns_health_research.persistence.bundle_naming import parse_bundle_id, resolve_bundle_id


def test_parse_bundle_id_legacy():
    assert parse_bundle_id("sabesp_v1") == ("sabesp", 1)
    assert parse_bundle_id("pns") is None


def test_resolve_bundle_id_fixed_name():
    assert resolve_bundle_id("pns") == "pns"
    assert resolve_bundle_id("sabesp", force=True) == "sabesp"
