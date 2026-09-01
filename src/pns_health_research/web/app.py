"""API HTTP read-only para o frontend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pns_health_research.web import query

app = FastAPI(
    title="PNS Health Research Lab API",
    version="1.0.0",
    description="API read-only para consulta de runs e artifacts.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/v1/dashboard")
def dashboard() -> dict:
    return query.dashboard_stats()


@app.get("/api/v1/runs")
def list_runs(
    type: str | None = Query(None, alias="type"),
    status: str | None = None,
    tags: str | None = None,
) -> list[dict]:
    tag_list = tags.split(",") if tags else None
    return query.list_all_runs(run_type=type, status=status, tags=tag_list)


@app.get("/api/v1/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    try:
        return query.get_run_detail(run_id)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/summary")
def run_summary(run_id: str) -> dict:
    try:
        return query.get_run_summary(run_id)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except query.ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/metrics")
def run_metrics(run_id: str, name: str | None = None) -> dict:
    try:
        return query.get_run_metrics(run_id, name)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except query.ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/lineage")
def run_lineage(run_id: str) -> dict:
    try:
        detail = query.get_run_detail(run_id)
        return detail.get("lineage", {})
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/grid-search/{model}")
def grid_search(run_id: str, model: str) -> dict:
    try:
        return query.get_grid_search(run_id, model)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/feature-importance/{model}")
def feature_importance(run_id: str, model: str) -> dict:
    try:
        return query.get_feature_importance(run_id, model)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/clusters")
def clusters(run_id: str) -> dict:
    try:
        return query.get_clusters(run_id)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/dataset")
def dataset(run_id: str) -> dict:
    try:
        return query.get_dataset_info(run_id)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/dictionary")
def dictionary() -> dict:
    return query.get_dictionary()


@app.get("/api/v1/dictionary/{internal_name}")
def dictionary_column(internal_name: str) -> dict:
    try:
        return query.get_dictionary_column(internal_name)
    except query.ArtifactError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/predictions/summary")
def predictions_summary(run_id: str) -> dict:
    try:
        return query.get_predictions_summary(run_id)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/runs/{run_id}/curves/{model}")
def curves(run_id: str, model: str) -> dict:
    try:
        return query.get_curves(run_id, model)
    except query.RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/comparisons/clustering")
def compare_clustering(runs: str = Query(..., description="IDs separados por vírgula")) -> dict:
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    return query.compare_clustering(run_ids)


@app.get("/api/v1/comparisons/classification")
def compare_classification(runs: str = Query(..., description="IDs separados por vírgula")) -> dict:
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    return query.compare_classification(run_ids)
