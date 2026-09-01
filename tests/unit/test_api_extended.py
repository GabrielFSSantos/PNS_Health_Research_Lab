"""Testes estendidos da API read-only."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from pns_health_research.web.app import app

client = TestClient(app)


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    from pns_health_research.core import paths

    run_id = "test_failed_run_xyz"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "failed",
                "schema_version": 2,
                "name": "Failed Run",
                "error": {"message": "test error", "stage": "classification"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    from pns_health_research.web import query as qmod

    monkeypatch.setattr(qmod, "RUNS_DIR", tmp_path / "runs")
    return run_id


def test_dictionary_has_columns():
    resp = client.get("/api/v1/dictionary")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data.get("columns", [])) >= 50


def test_dictionary_column():
    resp = client.get("/api/v1/dictionary/Sexo")
    assert resp.status_code == 200
    assert resp.json()["internal_name"] == "Sexo"


def test_dictionary_column_not_found():
    resp = client.get("/api/v1/dictionary/__nonexistent__")
    assert resp.status_code == 404


def test_run_not_found():
    resp = client.get("/api/v1/runs/nonexistent-run-id-xyz")
    assert resp.status_code == 404


def test_failed_run_listable(failed_run):
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200
    ids = [r["run_id"] for r in resp.json()]
    assert failed_run in ids


def test_compare_clustering_empty():
    resp = client.get("/api/v1/comparisons/clustering?runs=a,b")
    assert resp.status_code == 200
    data = resp.json()
    assert "comparison" in data
    assert "alignment_warning" in data


def test_predictions_summary_missing():
    resp = client.get("/api/v1/runs/pesquisa-original-legacy/predictions/summary")
    assert resp.status_code == 200


def test_curves_present_for_showcase():
    resp = client.get("/api/v1/runs/pesquisa-original-legacy/curves/random_forest")
    assert resp.status_code == 200
    data = resp.json()
    assert "roc" in data


def test_curves_missing_model():
    resp = client.get("/api/v1/runs/pesquisa-original-legacy/curves/__nonexistent__")
    assert resp.status_code == 200
    assert resp.json().get("status") == "missing"


def test_showcase_grid_search_present():
    resp = client.get("/api/v1/runs/pesquisa-original-legacy/grid-search/random_forest")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") != "missing" or "search" in data
