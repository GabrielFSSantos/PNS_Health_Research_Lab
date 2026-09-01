import json
from pathlib import Path

from pns_health_research.persistence.run_registry import load_index, tag_run


def test_tag_run_updates_manifest(tmp_path: Path, monkeypatch):
    run_id = "test_run_tag"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"experiment_name": "t", "profile": "p", "tags": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr("pns_health_research.persistence.run_registry.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(
        "pns_health_research.persistence.run_registry.INDEX_PATH",
        tmp_path / "runs" / "index.json",
    )
    tag_run(run_id, preserve=True, add_tags=["legacy"], category="legacy")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["preserve"] is True
    assert "legacy" in manifest["tags"]
    idx = load_index()
    assert idx["categories"]["legacy"] == run_id
