"""Testes do motor de insights."""

from __future__ import annotations

from pns_health_research.dashboard.insights import engine
from pns_health_research.dashboard.insights.rules import InsightLevel


def test_overview_insights():
    insights = engine.generate(
        {
            "page": "overview",
            "total_runs": 3,
            "best_classification": {
                "name": "Legacy",
                "primary_metric": {"value": 0.42},
            },
        }
    )
    assert len(insights) >= 2
    assert any(i.level == InsightLevel.SUCCESS for i in insights)


def test_clustering_weak_silhouette():
    insights = engine.generate(
        {"page": "clustering", "silhouette_score": 0.1, "artifacts_status": {}}
    )
    assert any(i.level == InsightLevel.WARNING for i in insights)


def test_research_insights_no_runs():
    insights = engine.generate({"page": "research", "total_runs": 0})
    assert len(insights) >= 1
