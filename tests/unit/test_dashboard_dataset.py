"""Testes do serviço de dataset do dashboard."""

from __future__ import annotations

from pns_health_research.dashboard.services import dataset
from pns_health_research.persistence.base_query import normalize_fingerprint

SHOWCASE_ID = "pesquisa-original-legacy"


def test_fingerprint_normalizes_row_column_keys():
    dataset.load_dataset_info.cache_clear()
    fp = dataset.load_fingerprint(SHOWCASE_ID)
    assert fp.get("rows") == 87678
    assert fp.get("columns") == 161


def test_fingerprint_modular_legacy():
    dataset.load_dataset_info.cache_clear()
    fp = dataset.load_fingerprint("legacy")
    assert fp.get("rows") == 87678
    assert fp.get("columns") == 179


def test_fingerprint_modular_pns():
    dataset.load_dataset_info.cache_clear()
    fp = dataset.load_fingerprint("pns")
    assert fp.get("rows") == 87678
    assert fp.get("columns") == 181
    assert fp.get("categorical_columns") == 90


def test_funnel_has_stages_for_modular_bases():
    dataset.load_dataset_info.cache_clear()
    for base in ("legacy", "pns"):
        funnel = dataset.load_funnel(base)
        stages = funnel.get("stages") or []
        assert len(stages) >= 6
        names = [s.get("name") for s in stages]
        assert "raw" in names
        assert "processed" in names


def test_normalize_fingerprint_modular_binary_block():
    data = {
        "binary": {
            "n_rows": 100,
            "n_columns": 50,
            "hash": "abc",
        }
    }
    flat = normalize_fingerprint(data)
    assert flat["rows"] == 100
    assert flat["columns"] == 50
    assert flat["hash"] == "abc"


def test_format_fingerprint_metric():
    assert dataset.format_fingerprint_metric(87678) == "87.678"
    assert dataset.format_fingerprint_metric(None) == "—"
