"""Testes do roteamento legacy_cv na classificação."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from pns_health_research.modules.classification.cv_routing import run_model_cv
from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT


def _minimal_config(profile: str, legacy_cv: bool):
    cfg = resolve_module_config("pns", "classification")
    cfg.experiment.profile = profile
    cfg.classification.legacy_cv = legacy_cv
    cfg.classification.backend = "sklearn"
    return cfg


@patch("pns_health_research.modules.classification.cv_routing.run_legacy_cv")
def test_legacy_cv_flag_uses_legacy_runner(mock_legacy):
    mock_legacy.return_value = MagicMock()
    cfg = _minimal_config("research_v2", legacy_cv=True)
    x = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    w = pd.Series([1.0, 1.0])
    run_model_cv(x, y, w, "random_forest", cfg)
    mock_legacy.assert_called_once()


@patch("pns_health_research.modules.classification.cv_routing.run_research_v2_cv")
def test_research_v2_uses_v2_runner(mock_v2):
    mock_v2.return_value = MagicMock()
    cfg = _minimal_config("research_v2", legacy_cv=False)
    x = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    w = pd.Series([1.0, 1.0])
    run_model_cv(x, y, w, "random_forest", cfg)
    mock_v2.assert_called_once()


@patch("pns_health_research.modules.classification.cv_routing.run_research_v2_cv")
def test_custom_profile_defaults_to_v2(mock_v2):
    mock_v2.return_value = MagicMock()
    cfg = _minimal_config("custom", legacy_cv=False)
    x = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    w = pd.Series([1.0, 1.0])
    run_model_cv(x, y, w, "random_forest", cfg)
    mock_v2.assert_called_once()
