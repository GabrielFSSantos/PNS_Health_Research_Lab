"""Configuração central do dashboard."""

from __future__ import annotations

from pns_health_research.core.paths import ARTIFACTS_DIR, CONFIGS_DIR, RUNS_DIR
from pns_health_research.dashboard import DASHBOARD_DIR
from pns_health_research.persistence.showcase import SHOWCASE_DIR

PROJECT_ROOT = DASHBOARD_DIR.parents[2]
THEME_CSS = DASHBOARD_DIR / "theme.css"
CACHE_TTL = 300

CLUSTER_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#ca8a04",
    "#9333ea",
    "#0891b2",
    "#ea580c",
    "#4f46e5",
    "#be185d",
    "#059669",
]

MODEL_COLORS: dict[str, str] = {
    "random_forest": "#2563eb",
    "svc": "#dc2626",
    "mlp": "#16a34a",
    "xgboost": "#9333ea",
}

DOCS_DIR = PROJECT_ROOT / "docs"

__all__ = [
    "ARTIFACTS_DIR",
    "CACHE_TTL",
    "CLUSTER_COLORS",
    "CONFIGS_DIR",
    "DOCS_DIR",
    "MODEL_COLORS",
    "PROJECT_ROOT",
    "RUNS_DIR",
    "SHOWCASE_DIR",
    "THEME_CSS",
]
