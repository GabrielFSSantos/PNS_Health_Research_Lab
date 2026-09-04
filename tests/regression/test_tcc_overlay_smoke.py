"""Smoke: overlay TCC carrega e expõe tolerâncias (sem rodar pipeline)."""

from __future__ import annotations

import pytest

from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROJECT_ROOT


@pytest.mark.regression_smoke
def test_tcc_overlay_loads():
    cfg = load_config(PROJECT_ROOT / "configs/_internal/tcc_overlay.yaml")
    assert cfg.experiment.profile == "legacy_reproduction"
    assert cfg.baseline.enabled is True
    assert cfg.clustering.reducer.n_components == 8
    assert cfg.preprocessing.nearmiss_before_split is True


@pytest.mark.regression_smoke
def test_tcc_overlay_classification_tolerances():
    cfg = load_config(PROJECT_ROOT / "configs/_internal/tcc_overlay.yaml")
    tol = cfg.baseline.tolerances
    clf = tol.get("classification", {})
    rf = clf.get("random_forest", {})
    xgb = clf.get("xgboost", {})
    assert abs(float(rf.get("f1", 0)) - 0.777) < 0.001
    assert abs(float(xgb.get("f1", 0)) - 0.784) < 0.001
