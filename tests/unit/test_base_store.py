"""Testes da camada base_store."""

from __future__ import annotations

from pathlib import Path

from pns_health_research.persistence.base_store import (
    base_path,
    clustering_path,
    load_registry,
    preprocessing_path,
    update_registry,
)


def test_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "pns_health_research.persistence.base_store.BASES_DIR",
        tmp_path / "bases",
    )
    assert base_path("pns") == tmp_path / "bases" / "pns"
    assert preprocessing_path("pns") == tmp_path / "bases" / "pns" / "preprocessing"
    assert clustering_path("pns") == tmp_path / "bases" / "pns" / "clustering"


def test_registry_run_count(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "pns_health_research.persistence.base_store.BASES_DIR",
        tmp_path / "bases",
    )
    update_registry("pns", "preprocessing", entry_config="cfg.yaml", extra={"rows_out": 100})
    reg = load_registry("pns")
    assert reg["modules"]["preprocessing"]["run_count"] == 1
    assert reg["modules"]["preprocessing"]["rows_out"] == 100
    update_registry("pns", "preprocessing")
    assert load_registry("pns")["modules"]["preprocessing"]["run_count"] == 2
