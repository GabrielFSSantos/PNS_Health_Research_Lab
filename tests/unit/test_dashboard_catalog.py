"""Testes dos services do dashboard."""

from __future__ import annotations

from pns_health_research.dashboard.services import catalog


def test_list_runs_includes_completed():
    runs = catalog.list_runs()
    assert len(runs) >= 1
    for r in runs:
        assert r.status == "completed"
        assert r.display_name


def test_list_runs_includes_showcase_baseline():
    runs = catalog.list_runs()
    labels = [r.showcase_label or r.run_id for r in runs]
    assert "pesquisa-original-legacy" in labels


def test_baseline_role_from_override():
    runs = catalog.list_runs()
    baseline = next(r for r in runs if r.showcase_label == "pesquisa-original-legacy")
    assert baseline.role == "baseline"
    assert "legacy" in baseline.display_name.lower() or "tcc" in baseline.display_name.lower()


def test_showcase_query_id():
    runs = catalog.list_runs()
    legacy = next(r for r in runs if r.showcase_label == "pesquisa-original-legacy")
    assert legacy.query_id == "pesquisa-original-legacy"


def test_get_summary_showcase():
    summary = catalog.get_summary("pesquisa-original-legacy")
    assert summary.get("status") == "completed"


def test_artifacts_status_present():
    status = catalog.artifacts_status("pesquisa-original-legacy")
    assert status.get("grid_search") == "present"


def test_dashboard_stats():
    stats = catalog.dashboard_stats()
    assert stats.get("pns_year") == 2019
    assert stats.get("total_runs", 0) >= 1


def test_run_overrides_loaded():
    overrides = catalog.load_run_overrides()
    assert "pesquisa-original-legacy" in overrides
