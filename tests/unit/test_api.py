"""Testes da API read-only."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pns_health_research.web.app import app

client = TestClient(app)


def test_dashboard():
    resp = client.get("/api/v1/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_runs" in data


def test_list_runs():
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_run_not_found():
    resp = client.get("/api/v1/runs/nonexistent-run-id-xyz")
    assert resp.status_code == 404


def test_dictionary():
    resp = client.get("/api/v1/dictionary")
    assert resp.status_code == 200
    data = resp.json()
    assert "columns" in data


@pytest.mark.skipif(
    not __import__("pathlib").Path("artifacts/showcase/pesquisa-original-legacy").exists(),
    reason="showcase legacy não presente",
)
def test_showcase_run_detail():
    resp = client.get("/api/v1/runs/pesquisa-original-legacy")
    assert resp.status_code == 200
    data = resp.json()
    assert "manifest" in data or "summary" in data


@pytest.mark.skipif(
    not __import__("pathlib").Path("artifacts/showcase/pesquisa-original-legacy").exists(),
    reason="showcase legacy não presente",
)
def test_grid_search_missing():
    resp = client.get("/api/v1/runs/pesquisa-original-legacy/grid-search/random_forest")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "missing" or "search" in data
