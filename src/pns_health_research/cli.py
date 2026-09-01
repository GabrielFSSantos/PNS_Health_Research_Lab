from __future__ import annotations

from pathlib import Path

import typer

from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import PROCESSED_DATA_DIR, RUNS_DIR, ensure_dirs
from pns_health_research.data.download import download_pns, ensure_data_available
from pns_health_research.experiments.runner import ExperimentRunner, compare_runs
from pns_health_research.persistence.run_registry import list_runs, tag_run
from pns_health_research.preprocessing.pipelines import run_preprocessing

app = typer.Typer(
    help="PNS Health Research Lab — plataforma científica para pesquisa com dados da PNS"
)
data_app = typer.Typer(help="Download e preparação de dados")
clustering_app = typer.Typer(help="Agrupamento")
classification_app = typer.Typer(help="Classificação")
experiment_app = typer.Typer(help="Experimentos e comparações")
app.add_typer(data_app, name="data")
app.add_typer(clustering_app, name="clustering")
app.add_typer(classification_app, name="classification")
app.add_typer(experiment_app, name="experiment")


def _parse_models(models: str | None) -> list[str] | None:
    if not models:
        return None
    return [m.strip() for m in models.split(",") if m.strip()]


def _maybe_download(cfg, *, download_if_missing: bool, no_download: bool) -> None:
    if no_download:
        return
    if download_if_missing or cfg.data.download_if_missing:
        ensure_data_available(
            cfg.data.year,
            dataset=cfg.data.dataset,
            source=cfg.data.source,
        )


@data_app.command("download")
def data_download(
    year: int = typer.Option(2019, "--year", help="Ano da PNS"),
    dataset: str | None = typer.Option(
        None, "--dataset", help="Chave em configs/datasets.yaml (ex.: pns_2019)"
    ),
    force: bool = typer.Option(False, "--force", help="Rebaixar mesmo se existir"),
) -> None:
    """Baixa microdados PNS (ação explícita; requer extra [acquisition] para PNSIBGE)."""
    ensure_dirs()
    if dataset:
        path = download_pns(year, dataset=dataset, force=force)
    else:
        path = download_pns(year, force=force)
    typer.echo(f"Dataset disponível: {path}")


@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(
        Path("configs/experiment.yaml"), "--config", help="YAML do experimento"
    ),
    dataset: str | None = typer.Option(None, "--dataset", help="Chave em datasets.yaml"),
    download_if_missing: bool = typer.Option(
        False, "--download-if-missing", help="Força download se o parquet ausente"
    ),
    no_download: bool = typer.Option(False, "--no-download", help="Não baixa mesmo se if_missing"),
) -> None:
    """Pré-processa dados e grava em data/processed/."""
    ensure_dirs()
    cfg = load_config(config, cli_dataset=dataset)
    _maybe_download(cfg, download_if_missing=download_if_missing, no_download=no_download)
    result = run_preprocessing(cfg)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.binary_df.to_parquet(PROCESSED_DATA_DIR / "binary_latest.parquet")
    result.categorical_df.to_parquet(PROCESSED_DATA_DIR / "categorical_latest.parquet")
    typer.echo(
        f"Preparado: {result.metrics['rows_out']} linhas, "
        f"bin={result.metrics['binary_columns']}, cat={result.metrics['categorical_columns']}"
    )


@clustering_app.command("run")
def clustering_run_cmd(
    config: Path = typer.Option(Path("configs/experiment.yaml"), "--config"),
    name: str | None = typer.Option(None, "--name", help="Nome amigável da run"),
    models: str | None = typer.Option(
        None, "--models", help="Filtro temporário de classificadores (ex.: xgboost,random_forest)"
    ),
    dataset: str | None = typer.Option(None, "--dataset", help="Chave em datasets.yaml"),
    download_if_missing: bool = typer.Option(False, "--download-if-missing"),
    no_download: bool = typer.Option(False, "--no-download"),
) -> None:
    cfg = load_config(config, cli_models=_parse_models(models), cli_dataset=dataset)
    if name:
        cfg.run.name = name
    _maybe_download(cfg, download_if_missing=download_if_missing, no_download=no_download)
    cfg.stages = ["preprocessing", "clustering"]
    run_dir = ExperimentRunner(cfg).run()
    typer.echo(f"Clustering run: {run_dir}")


@classification_app.command("run")
def classification_run_cmd(
    config: Path = typer.Option(Path("configs/experiment.yaml"), "--config"),
    from_run: str | None = typer.Option(None, "--from-run"),
    name: str | None = typer.Option(None, "--name", help="Nome amigável da run"),
    models: str | None = typer.Option(
        None, "--models", help="Filtro temporário de classificadores (ex.: xgboost,random_forest)"
    ),
    dataset: str | None = typer.Option(None, "--dataset", help="Chave em datasets.yaml"),
    download_if_missing: bool = typer.Option(False, "--download-if-missing"),
    no_download: bool = typer.Option(False, "--no-download"),
) -> None:
    cfg = load_config(config, cli_models=_parse_models(models), cli_dataset=dataset)
    if from_run:
        cfg.classification.input_run = from_run
    if name:
        cfg.run.name = name
    _maybe_download(cfg, download_if_missing=download_if_missing, no_download=no_download)
    if from_run:
        cfg.stages = ["classification", "statistics"]
    else:
        cfg.stages = ["preprocessing", "classification", "statistics"]
    run_dir = ExperimentRunner(cfg).run()
    typer.echo(f"Classification run: {run_dir}")


@experiment_app.command("run")
def experiment_run_cmd(
    config: Path = typer.Option(Path("configs/experiment.yaml"), "--config"),
    name: str | None = typer.Option(None, "--name", help="Nome amigável da run"),
    models: str | None = typer.Option(
        None, "--models", help="Filtro temporário de classificadores (ex.: xgboost,random_forest)"
    ),
    dataset: str | None = typer.Option(None, "--dataset", help="Chave em datasets.yaml"),
    download_if_missing: bool = typer.Option(False, "--download-if-missing"),
    no_download: bool = typer.Option(False, "--no-download"),
) -> None:
    cfg = load_config(config, cli_models=_parse_models(models), cli_dataset=dataset)
    if name:
        cfg.run.name = name
    _maybe_download(cfg, download_if_missing=download_if_missing, no_download=no_download)
    run_dir = ExperimentRunner(cfg).run()
    typer.echo(f"Experiment run: {run_dir}")


@experiment_app.command("compare")
def experiment_compare_cmd(
    baseline: str = typer.Option(..., "--baseline"),
    candidate: str = typer.Option(..., "--candidate"),
) -> None:
    b_dir = RUNS_DIR / baseline
    c_dir = RUNS_DIR / candidate
    df = compare_runs(b_dir, c_dir)
    out = RUNS_DIR / f"compare_{baseline}_vs_{candidate}.parquet"
    df.to_parquet(out, index=False)
    typer.echo(f"Comparação salva: {out}")


@experiment_app.command("list")
def experiment_list_cmd() -> None:
    for row in list_runs():
        typer.echo(
            f"{row.get('run_id')}  profile={row.get('profile')}  "
            f"status={row.get('status')}  tags={row.get('tags', [])}"
        )


@experiment_app.command("tag")
def experiment_tag_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    preserve: bool = typer.Option(False, "--preserve"),
    add_tag: list[str] = typer.Option(None, "--add-tag"),
    category: str | None = typer.Option(
        None,
        "--category",
        help="Ex.: legacy, best_f1_positive (não sobrescreve se já existir)",
    ),
) -> None:
    tag_run(run_id, preserve=preserve, add_tags=add_tag, category=category)
    typer.echo(f"Run {run_id} atualizado.")


@app.command("web")
def web_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Inicia API read-only (requer extra [web])."""
    try:
        import uvicorn
    except ImportError as exc:
        typer.echo("Instale com: pip install 'pns-health-research-lab[web]'")
        raise typer.Exit(1) from exc
    uvicorn.run("pns_health_research.web.app:app", host=host, port=port, reload=False)


@app.command("dashboard")
def dashboard_cmd(
    port: int = typer.Option(8501, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Inicia dashboard Streamlit (requer extra [dashboard])."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        typer.echo("Instale com: pip install 'pns-health-research-lab[dashboard]'")
        raise typer.Exit(1) from exc

    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    app()
