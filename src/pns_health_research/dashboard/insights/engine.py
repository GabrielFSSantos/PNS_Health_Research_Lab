"""Motor de insights PNS."""

from __future__ import annotations

from typing import Any

from pns_health_research.dashboard.insights.rules import SILHOUETTE_WEAK, Insight, InsightLevel


def generate(context: dict[str, Any]) -> list[Insight]:
    page = context.get("page", "")
    handlers = {
        "overview": _overview_insights,
        "dataset": _dataset_insights,
        "clustering": _clustering_insights,
        "clustering_compare": _clustering_compare_insights,
        "classification": _classification_insights,
        "classification_compare": _classification_compare_insights,
        "evolution": _evolution_insights,
        "research": _research_insights,
    }
    handler = handlers.get(page)
    return handler(context) if handler else []


def _overview_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    if ctx.get("total_runs"):
        out.append(Insight(f"{ctx['total_runs']} runs disponíveis para exploração.", InsightLevel.INFO))
    best = ctx.get("best_classification")
    if best:
        pm = best.get("primary_metric") or {}
        val = pm.get("value")
        if val is not None:
            out.append(
                Insight(
                    f"Melhor classificação: {best.get('name')} (F1+ {val:.1%}).",
                    InsightLevel.SUCCESS,
                )
            )
    missing = ctx.get("artifacts_missing") or []
    for item in missing:
        out.append(
            Insight(
                f"Artifact ausente: {item}. Execute um experimento completo para popular.",
                InsightLevel.WARNING,
            )
        )
    return out


def _dataset_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    if not ctx.get("has_fingerprint"):
        out.append(Insight("Fingerprint de dados não disponível nesta run.", InsightLevel.WARNING))
    return out


def _clustering_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    sil = ctx.get("silhouette_score")
    if sil is not None and sil < SILHOUETTE_WEAK:
        out.append(
            Insight(
                f"Silhouette {sil:.3f} abaixo de {SILHOUETTE_WEAK} — clusters pouco separados.",
                InsightLevel.WARNING,
            )
        )
    status = ctx.get("artifacts_status") or {}
    for key, val in status.items():
        if val == "missing":
            out.append(Insight(f"Artifact de clustering ausente: {key}.", InsightLevel.WARNING))
    return out


def _clustering_compare_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    if ctx.get("alignment_warning"):
        out.append(Insight(ctx["alignment_warning"], InsightLevel.WARNING))
    return out


def _classification_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    if ctx.get("grid_search_missing"):
        out.append(
            Insight(
                "GridSearch não disponível — tuning manual ou run anterior ao contrato v2.",
                InsightLevel.WARNING,
            )
        )
    return out


def _classification_compare_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    best = ctx.get("best_model")
    if best:
        out.append(Insight(f"Melhor F1+ no recorte: {best}.", InsightLevel.SUCCESS))
    return out


def _evolution_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    if ctx.get("showcase_count", 0) < 2:
        out.append(
            Insight(
                "Execute mais de uma run concluída para comparar evolução no gráfico.",
                InsightLevel.INFO,
            )
        )
    return out


def _research_insights(ctx: dict[str, Any]) -> list[Insight]:
    out: list[Insight] = []
    total = ctx.get("total_runs", 0)
    if total == 0:
        out.append(
            Insight(
                "Nenhuma execução concluída. Rode `./scripts/run_experiment.sh` para a primeira run.",
                InsightLevel.INFO,
            )
        )
    return out
