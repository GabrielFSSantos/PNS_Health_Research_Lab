"""Testes de classificação paralela (effective_n_jobs)."""

from pns_health_research.modules.classification.tuning import effective_n_jobs, load_model_n_jobs
from pns_health_research.config.profile_resolver import resolve_module_config
from pns_health_research.core.paths import PROJECT_ROOT


def test_effective_n_jobs_respects_parallel_flag():
    cfg = resolve_module_config("pns", "classification")
    serial = effective_n_jobs("random_forest", parallel=False, max_workers=4, config=cfg)
    parallel = effective_n_jobs("random_forest", parallel=True, max_workers=4, config=cfg)
    assert serial >= 1
    assert parallel >= serial


def test_load_model_n_jobs_from_registry():
    cfg = resolve_module_config("pns", "classification")
    n_jobs = load_model_n_jobs("random_forest", config=cfg)
    assert n_jobs >= 1
