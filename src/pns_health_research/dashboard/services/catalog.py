"""Catálogo de runs e metadados."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from pns_health_research.core.paths import CONFIGS_DIR
from pns_health_research.web import query

RUN_OVERRIDES_PATH = CONFIGS_DIR / "dashboard" / "run_overrides.yaml"


@dataclass
class RunOverride:
    """Override opcional de exibição para uma run, base ou showcase."""

    key: str
    display_name: str | None = None
    role: str | None = None  # baseline | candidate
    layer: str | None = None
    subtitle: str | None = None


@dataclass
class RunSummary:
    run_id: str
    showcase_label: str | None
    name: str
    display_name: str
    type: str
    status: str
    source: str
    source_type: str  # base | showcase | runs
    role: str
    layer: str | None = None
    subtitle: str | None = None
    profile_kind: str | None = None
    primary_metric: dict[str, Any] | None = None
    tags: list[str] | None = None

    @property
    def query_id(self) -> str:
        return self.showcase_label or self.run_id


def _load_run_overrides() -> dict[str, RunOverride]:
    if not RUN_OVERRIDES_PATH.exists():
        return {}
    data = yaml.safe_load(RUN_OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
    overrides: dict[str, RunOverride] = {}
    for item in data.get("overrides", []):
        key = item.get("key") or item.get("run_id") or item.get("label")
        if not key:
            continue
        overrides[key] = RunOverride(
            key=key,
            display_name=item.get("display_name"),
            role=item.get("role"),
            layer=item.get("layer"),
            subtitle=item.get("subtitle"),
        )
    return overrides


def load_run_overrides() -> dict[str, RunOverride]:
    return _load_run_overrides()


def _infer_role(row: dict[str, Any], overrides: dict[str, RunOverride]) -> str:
    tags = row.get("tags") or []
    if "baseline" in tags:
        return "baseline"
    for key in (row.get("showcase_label"), row.get("run_id")):
        if key and key in overrides and overrides[key].role:
            return overrides[key].role  # type: ignore[return-value]
    if row.get("profile_kind") == "legacy":
        return "baseline"
    return "candidate"


def _resolve_override(row: dict[str, Any], overrides: dict[str, RunOverride]) -> RunOverride | None:
    for key in (row.get("showcase_label"), row.get("run_id")):
        if key and key in overrides:
            return overrides[key]
    return None


def _resolve_display_name(row: dict[str, Any], overrides: dict[str, RunOverride]) -> str:
    ov = _resolve_override(row, overrides)
    if ov and ov.display_name:
        return ov.display_name
    return row.get("name") or row.get("run_id", "")


def _source_type(row: dict[str, Any]) -> str:
    src = row.get("source", "runs")
    if src == "base":
        return "base"
    if src == "showcase":
        return "showcase"
    return "runs"


def list_runs(
    *,
    run_type: str | None = None,
    status: str | None = "completed",
) -> list[RunSummary]:
    overrides = _load_run_overrides()
    rows = query.list_all_runs(run_type=run_type, status=status)
    result: list[RunSummary] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        ov = _resolve_override(row, overrides)
        tags = list(row.get("tags") or [])
        profile_kind = row.get("profile_kind")
        if profile_kind and profile_kind not in tags:
            tags.append(profile_kind)
        result.append(
            RunSummary(
                run_id=row["run_id"],
                showcase_label=row.get("showcase_label"),
                name=row.get("name", row["run_id"]),
                display_name=_resolve_display_name(row, overrides),
                type=row.get("type", "experiment"),
                status=row.get("status", "unknown"),
                source=row.get("source", "runs"),
                source_type=_source_type(row),
                role=_infer_role(row, overrides),
                layer=ov.layer if ov else None,
                subtitle=ov.subtitle if ov else None,
                profile_kind=profile_kind,
                primary_metric=row.get("primary_metric"),
                tags=tags,
            )
        )
    result.sort(key=lambda r: (r.layer or "9", r.run_id), reverse=False)
    return result


def get_run(query_id: str) -> RunSummary | None:
    for run in list_runs():
        if run.query_id == query_id:
            return run
    return None


def get_baseline_f1() -> float | None:
    runs = list_runs()
    for r in runs:
        if r.role == "baseline":
            pm = r.primary_metric or {}
            val = pm.get("value")
            if val is not None:
                return float(val)
    return None


def get_summary(run_id: str) -> dict[str, Any]:
    return query.get_run_summary(run_id)


def get_detail(run_id: str) -> dict[str, Any]:
    return query.get_run_detail(run_id)


def artifacts_status(run_id: str) -> dict[str, str]:
    detail = get_detail(run_id)
    manifest = detail.get("manifest") or {}
    return manifest.get("artifacts_status") or {}


def dashboard_stats() -> dict[str, Any]:
    runs = list_runs()
    modular = [r for r in runs if r.source_type == "base"]
    showcase = [r for r in runs if r.source_type == "showcase"]
    clustering = [r for r in runs if r.type in ("experiment", "clustering")]
    classification = [r for r in runs if r.type in ("experiment", "classification")]
    best_clf = None
    best_val = -1.0
    for r in runs:
        pm = r.primary_metric or {}
        val = pm.get("value")
        if val is not None and val > best_val:
            best_val = float(val)
            best_clf = r
    return {
        "pns_year": 2019,
        "total_runs": len(runs),
        "modular_bases": len(modular),
        "showcase_count": len(showcase),
        "clustering_runs": len(clustering),
        "classification_runs": len(classification),
        "preserved_runs": len(runs),
        "best_classification": best_clf,
        "recent_runs": runs,
        "showcase_available": len(showcase) > 0,
    }
