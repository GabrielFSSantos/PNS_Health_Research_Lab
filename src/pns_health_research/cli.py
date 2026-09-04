from __future__ import annotations

from pathlib import Path

import typer

from pns_health_research.config.cli_config import load_module_config
from pns_health_research.config.loader import load_config
from pns_health_research.core.paths import ensure_dirs
from pns_health_research.data.download import download_pns, ensure_data_available
from pns_health_research.modules.preprocessing.pipelines import run_preprocessing

app = typer.Typer(
    help="PNS Health Research Lab — plataforma científica para pesquisa com dados da PNS"
)
data_app = typer.Typer(help="Download e preparação de dados")
bases_app = typer.Typer(help="Bases nomeadas em data/bases/")
data_app.add_typer(bases_app, name="bases")
clustering_app = typer.Typer(help="Agrupamento")
classification_app = typer.Typer(help="Classificação")
app.add_typer(data_app, name="data")
app.add_typer(clustering_app, name="clustering")
app.add_typer(classification_app, name="classification")


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
        None, "--dataset", help="Chave em configs/shared/datasets.yaml (ex.: pns_2019)"
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
        Path("configs/profiles/pns.yaml"),
        "--config",
        help="Perfil ou override YAML (com --profile, só overrides em configs/overrides/)",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Preset: pns (rigoroso) ou legacy (TCC); sobrescreve --name e --config",
    ),
    name: str = typer.Option("pns", "--name", help="Nome da base em data/bases/<name>/"),
    dataset: str | None = typer.Option(None, "--dataset", help="Chave em datasets.yaml"),
    dataset_id: str | None = typer.Option(
        None, "--dataset-id", help="Alias legado para --name (escape hatch)"
    ),
    download_if_missing: bool = typer.Option(
        False, "--download-if-missing", help="Força download se o parquet ausente"
    ),
    no_download: bool = typer.Option(False, "--no-download", help="Não baixa mesmo se if_missing"),
    force: bool = typer.Option(False, "--force", help="Sobrescreve preprocessing/ da base"),
) -> None:
    """Pré-processa dados e grava em data/bases/<name>/preprocessing/."""
    from pns_health_research.persistence.base_store import write_preprocessing_base
    from pns_health_research.persistence.preprocess_presets import resolve_prepare_args

    ensure_dirs()
    try:
        config_path, base_name, profile_kind = resolve_prepare_args(
            profile=profile,
            config=config,
            name=name,
            dataset_id=dataset_id,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    cfg, entry_path = load_module_config(
        profile=profile,
        module="preprocessing",
        config=config_path,
        cli_dataset=dataset,
    )
    _maybe_download(cfg, download_if_missing=download_if_missing, no_download=no_download)
    result = run_preprocessing(cfg)

    out = write_preprocessing_base(
        result,
        cfg,
        base_name=base_name,
        entry_config_path=entry_path,
        profile_kind=profile_kind,
    )
    typer.echo(f"Base: {out.parent.parent}")
    typer.echo(f"Módulo: {out}")
    typer.echo(f"Base name: {base_name}")
    typer.echo(f"Profile kind: {profile_kind}")
    prev = result.metrics.get("target_prevalence") or {}
    if prev:
        for target_name, stats in prev.items():
            typer.echo(
                f"  {target_name}: prevalência={stats.get('prevalence', 0):.3f} "
                f"(n={stats.get('n', 0)})"
            )
    typer.echo(
        f"Preparado: {result.metrics['rows_out']} linhas, "
        f"bin={result.metrics['binary_columns']}, cat={result.metrics['categorical_columns']}"
    )


@bases_app.command("list")
def data_bases_list() -> None:
    """Lista bases em data/bases/ com resumo do registry."""
    from pns_health_research.persistence.base_store import list_bases

    ensure_dirs()
    rows = list_bases()
    if not rows:
        typer.echo("Nenhuma base em data/bases/")
        return
    for row in rows:
        name = row["base_name"]
        modules = row.get("modules") or {}
        pp = modules.get("preprocessing") or {}
        cl = modules.get("clustering") or {}
        cf = modules.get("classification") or {}
        typer.echo(f"{name}")
        if pp:
            typer.echo(
                f"  preprocessing: kind={pp.get('profile_kind', '?')} "
                f"runs={pp.get('run_count', 0)} "
                f"last={pp.get('last_run_at', '?')} "
                f"rows={pp.get('rows_out', '?')}"
            )
        if cl:
            typer.echo(
                f"  clustering: kind={cl.get('profile_kind', '?')} "
                f"runs={cl.get('run_count', 0)} "
                f"last={cl.get('last_run_at', '?')} "
                f"k={cl.get('n_clusters', '?')} "
                f"silhouette={cl.get('silhouette', '?')}"
            )
        if cf:
            typer.echo(
                f"  classification: kind={cf.get('profile_kind', '?')} "
                f"runs={cf.get('run_count', 0)} "
                f"last={cf.get('last_run_at', '?')} "
                f"models={len(cf.get('models') or [])} "
                f"best={cf.get('best_model', '?')} "
                f"{cf.get('primary_metric', '?')}={cf.get('best_score', '?')}"
            )


@clustering_app.command("run")
def clustering_run_cmd(
    config: Path = typer.Option(
        Path("configs/profiles/pns.yaml"),
        "--config",
        help="Perfil ou override YAML",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Preset: pns (rigoroso) ou legacy (TCC); sobrescreve --name, --config e --source-base",
    ),
    name: str = typer.Option("pns", "--name", help="Nome da base em data/bases/<name>/"),
    source_base: str | None = typer.Option(
        None, "--source-base", help="Base de pré-processamento (default: mesmo --name)"
    ),
    dataset: str | None = typer.Option(None, "--dataset", help="Chave em datasets.yaml"),
) -> None:
    """Agrupa dados lendo preprocessing/ da base e grava em clustering/."""
    from pns_health_research.modules.clustering.pipeline_runner import run_clustering_pipeline
    from pns_health_research.persistence.base_store import write_clustering_base
    from pns_health_research.persistence.clustering_presets import resolve_clustering_args

    ensure_dirs()
    try:
        config_path, base_name, src, profile_kind = resolve_clustering_args(
            profile=profile,
            config=config,
            name=name,
            source_base=source_base,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    cfg, entry_path = load_module_config(
        profile=profile,
        module="clustering",
        config=config_path,
        cli_dataset=dataset,
    )
    src = source_base or cfg.clustering.source_base or src

    result = run_clustering_pipeline(cfg, base_name=base_name, source_base=src)
    out = write_clustering_base(
        base_name=base_name,
        result=result,
        config=cfg,
        source_base=src,
        entry_config_path=entry_path,
        profile_kind=profile_kind,
    )
    typer.echo(f"Base: {out.parent.parent}")
    typer.echo(f"Módulo clustering: {out}")
    typer.echo(f"Base name: {base_name}")
    typer.echo(f"Profile kind: {profile_kind}")
    typer.echo(f"source_base: {src}")
    typer.echo(
        f"Agrupado: k={result.metrics.get('n_clusters')}, "
        f"silhouette={result.metrics.get('silhouette', 'n/a')}"
    )


@classification_app.command("run")
def classification_run_cmd(
    config: Path = typer.Option(
        Path("configs/profiles/pns.yaml"),
        "--config",
        help="Perfil ou override YAML",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Preset: pns (rigoroso) ou legacy (TCC); sobrescreve --name, --config e --source-base",
    ),
    name: str = typer.Option("pns", "--name", help="Nome da base em data/bases/<name>/"),
    source_base: str | None = typer.Option(
        None, "--source-base", help="Base de clustering (default: mesmo --name)"
    ),
    models: str | None = typer.Option(
        None, "--models", help="Filtro temporário de classificadores (ex.: xgboost,random_forest)"
    ),
    dataset: str | None = typer.Option(None, "--dataset", help="Chave em datasets.yaml"),
) -> None:
    """Classifica dados lendo clustering/ da base e grava em classification/."""
    from pns_health_research.modules.classification.pipeline_runner import run_classification_pipeline
    from pns_health_research.persistence.base_store import write_classification_base
    from pns_health_research.persistence.classification_presets import resolve_classification_args

    ensure_dirs()
    try:
        config_path, base_name, src, profile_kind = resolve_classification_args(
            profile=profile,
            config=config,
            name=name,
            source_base=source_base,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    cfg, entry_path = load_module_config(
        profile=profile,
        module="classification",
        config=config_path,
        cli_models=_parse_models(models),
        cli_dataset=dataset,
    )
    src = source_base or cfg.classification.source_base or src

    result = run_classification_pipeline(cfg, base_name=base_name, source_base=src)
    out = write_classification_base(
        result,
        cfg,
        base_name=base_name,
        source_base=src,
        entry_config_path=entry_path,
        profile_kind=profile_kind,
    )
    typer.echo(f"Base: {out.parent.parent}")
    typer.echo(f"Módulo classification: {out}")
    typer.echo(f"Base name: {base_name}")
    typer.echo(f"Profile kind: {profile_kind}")
    typer.echo(f"source_base: {src}")
    primary = cfg.evaluation.primary_metric
    typer.echo(f"Métricas: {list(result.metrics.keys())}")
    for model_name, row in result.metrics.items():
        if isinstance(row, dict) and row.get("status") != "failed":
            typer.echo(f"  {model_name}: {primary}={row.get(primary, 'n/a')}")


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
