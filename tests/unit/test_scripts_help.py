"""Smoke dos scripts de entrada."""

from __future__ import annotations

import subprocess

import pytest

from pns_health_research.core.paths import PROJECT_ROOT

SCRIPTS = [
    "setup_env.sh",
    "run_download.sh",
    "run_preprocess.sh",
    "run_clustering.sh",
    "run_classification.sh",
    "run_experiment.sh",
    "run_dashboard.sh",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_help_exits_zero(name: str):
    path = PROJECT_ROOT / "scripts" / name
    assert path.exists()
    result = subprocess.run(
        ["bash", str(path), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Uso:" in result.stdout or "Uso:" in result.stderr
