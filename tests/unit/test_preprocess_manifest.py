"""Testes do manifest e registry de bases."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.persistence.base_store import (
    load_registry,
    resolve_preprocessing_source,
    write_preprocessing_base,
)
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing


def test_preprocess_base_artifacts(synthetic_raw_path: Path, monkeypatch_bases_dir: Path):
    cfg = resolve_module_config("pns", "preprocessing")
    result = run_preprocessing(cfg, raw_path=str(synthetic_raw_path))

    out = write_preprocessing_base(
        result,
        cfg,
        base_name="testbase",
        profile_kind="pns",
    )

    assert out == monkeypatch_bases_dir / "testbase" / "preprocessing"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_kind"] == "pns"

    registry = load_registry("testbase")
    assert registry["modules"]["preprocessing"]["run_count"] == 1
    assert resolve_preprocessing_source("testbase").binary_path.exists()

    write_preprocessing_base(result, cfg, base_name="testbase")
    assert load_registry("testbase")["modules"]["preprocessing"]["run_count"] == 2

    pp_yaml = yaml.safe_load((out / "preprocessing.resolved.yaml").read_text(encoding="utf-8"))
    assert pp_yaml["schema_version"] == "3.0"
