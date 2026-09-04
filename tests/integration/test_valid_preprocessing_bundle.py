import json
from pathlib import Path

from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT
from pns_health_research.persistence.base_store import write_preprocessing_base
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing


def test_valid_preprocessing_bundle(synthetic_raw_path: Path, monkeypatch_bases_dir: Path):
    cfg = resolve_module_config("pns", "preprocessing")
    result = run_preprocessing(cfg, raw_path=str(synthetic_raw_path))
    assert "servico_j014" in result.categorical_df.columns
    assert "servico_composto" in result.categorical_df.columns
    assert result.processing_stats.get("missing_audit") is not None

    out = write_preprocessing_base(
        result,
        cfg,
        base_name="pns",
        profile_kind="pns",
    )
    assert (out / "manifest.json").exists()
    assert (out / "binary.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_kind"] == "pns"
    assert manifest["methodological_profile"] == "pns_preprocessing"
