"""Testes de export showcase a partir de base modular."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pns_health_research.persistence.showcase import SHOWCASE_DIR, export_showcase_from_base


def test_export_showcase_from_base_copies_funnel_and_flat_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pns_health_research.persistence.showcase.SHOWCASE_DIR",
        tmp_path / "showcase",
    )
    label = "test-export-pns"
    dest = export_showcase_from_base("pns", label=label, name="Test PNS")
    assert dest.is_dir()
    assert (dest / "metrics" / "preprocess_funnel.json").exists()
    assert (dest / "metrics" / "dataset_distributions.json").exists()
    assert (dest / "data_fingerprint.json").exists()
    fp = json.loads((dest / "data_fingerprint.json").read_text(encoding="utf-8"))
    assert fp.get("n_rows") == 87678
    assert fp.get("n_columns") == 181
    assert fp.get("hash")
    shutil.rmtree(dest, ignore_errors=True)
