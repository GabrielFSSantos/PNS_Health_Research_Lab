"""Testes do serviço de dataset do dashboard."""

from __future__ import annotations

from pns_health_research.dashboard.services import dataset

RUN_ID = "pesquisa-original-legacy"


def test_fingerprint_normalizes_row_column_keys():
    fp = dataset.load_fingerprint(RUN_ID)
    assert fp.get("rows") == 87678
    assert fp.get("columns") == 161


def test_format_fingerprint_metric():
    assert dataset.format_fingerprint_metric(87678) == "87.678"
    assert dataset.format_fingerprint_metric(None) == "—"
