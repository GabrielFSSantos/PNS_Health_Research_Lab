"""Regressão de métricas de classificação com PNS real."""

from __future__ import annotations

import pytest

from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.modules.classification.pipeline_runner import run_classification_pipeline
from tests.conftest import apply_fast_regression


@pytest.mark.regression
def test_legacy_profile_classification_metrics(regression_bases):
    cf_cfg = apply_fast_regression(resolve_module_config("legacy", "classification"))
    result = run_classification_pipeline(cf_cfg, base_name="legacy", source_base="legacy")
    row = result.metrics.get("random_forest")
    assert row is not None and isinstance(row, dict)
    assert row.get("status") != "failed"
    assert float(row.get("f1", 0)) > 0.5


@pytest.mark.regression
def test_pns_profile_classification_metrics(regression_bases):
    cf_cfg = apply_fast_regression(resolve_module_config("pns", "classification"))
    result = run_classification_pipeline(cf_cfg, base_name="pns", source_base="pns")
    row = result.metrics.get("random_forest")
    assert row is not None and isinstance(row, dict)
    assert row.get("status") != "failed"
    assert row.get("f1_positive", 0) > 0
