from __future__ import annotations

from pathlib import Path

import typer

from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import ensure_dirs
from pns_health_research.experiments.runner import ExperimentRunner, compare_runs
from pns_health_research.preprocessing.pipelines import run_preprocessing

app = typer.Typer(help="PNS Health ML experimental platform")
data_app = typer.Typer(help="Data ingestion and preparation")
app.add_typer(data_app, name="data")


@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(..., "--config", help="Experiment config YAML"),
) -> None:
    """Prepare preprocessing datasets."""
    ensure_dirs()
    cfg = load_config(config)
    result = run_preprocessing(cfg)
    from pns_health_research.core.paths import PROCESSED_DATA_DIR

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.binary_df.to_parquet(PROCESSED_DATA_DIR / "binary_latest.parquet")
    result.categorical_df.to_parquet(PROCESSED_DATA_DIR / "categorical_latest.parquet")
    typer.echo(
        f"Prepared {result.metrics['rows_out']} rows, "
        f"binary cols={result.metrics['binary_columns']}"
    )


@data_app.command("acquire")
def data_acquire() -> None:
    """Fetch PNS 2019 via R/PNSIBGE (requires extra [acquisition])."""
    from pns_health_research.data.acquisition import acquire_pns2019

    ensure_dirs()
    path = acquire_pns2019()
    typer.echo(f"Acquired dataset: {path}")


@app.command("clustering")
def clustering_run(
    config: Path = typer.Option(..., "--config"),
) -> None:
    cfg = load_config(config)
    cfg.stages = ["preprocessing", "clustering"]
    run_dir = ExperimentRunner(cfg).run()
    typer.echo(f"Clustering run: {run_dir}")


@app.command("classification")
def classification_run(
    config: Path = typer.Option(..., "--config"),
    from_run: str | None = typer.Option(
        None,
        "--from-run",
        help="Run ID com clustering válido; pula prep/cluster e resume classificação.",
    ),
) -> None:
    cfg = load_config(config)
    if from_run:
        cfg.classification.input_run = from_run
    cfg.stages = ["classification", "statistics"]
    run_dir = ExperimentRunner(cfg).run()
    typer.echo(f"Classification run: {run_dir}")


@app.command("experiment")
def experiment_run(
    config: Path = typer.Option(..., "--config"),
) -> None:
    cfg = load_config(config)
    run_dir = ExperimentRunner(cfg).run()
    typer.echo(f"Experiment run: {run_dir}")


@app.command("compare")
def compare(
    baseline: str = typer.Option(..., "--baseline"),
    candidate: str = typer.Option(..., "--candidate"),
) -> None:
    from pns_health_research.core.paths import RUNS_DIR

    b_dir = RUNS_DIR / baseline
    c_dir = RUNS_DIR / candidate
    df = compare_runs(b_dir, c_dir)
    out = RUNS_DIR / f"compare_{baseline}_vs_{candidate}.parquet"
    df.to_parquet(out, index=False)
    typer.echo(f"Comparison saved: {out}")


if __name__ == "__main__":
    app()
