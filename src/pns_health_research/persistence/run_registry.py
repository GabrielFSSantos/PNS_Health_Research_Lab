from __future__ import annotations

import json
import logging

from pns_health_research.core.paths import RUNS_DIR

logger = logging.getLogger("pns_health_research")

INDEX_PATH = RUNS_DIR / "index.json"


def load_index() -> dict:
    if not INDEX_PATH.exists():
        return {"runs": {}, "categories": {}}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def save_index(index: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def tag_run(
    run_id: str,
    *,
    preserve: bool = False,
    add_tags: list[str] | None = None,
    category: str | None = None,
) -> None:
    """Marca run como importante. Nunca sobrescreve categorias automaticamente."""
    manifest_path = RUNS_DIR / run_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run não encontrado: {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tags = set(manifest.get("tags", []))
    if add_tags:
        tags.update(add_tags)
    if preserve:
        manifest["preserve"] = True
        tags.add("preserved")
    manifest["tags"] = sorted(tags)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    index = load_index()
    index.setdefault("runs", {})[run_id] = {
        "preserve": manifest.get("preserve", False),
        "tags": manifest["tags"],
        "experiment_name": manifest.get("experiment_name"),
        "profile": manifest.get("profile"),
    }
    if category:
        if category in index.get("categories", {}):
            logger.warning(
                "Categoria '%s' já aponta para %s; não sobrescrevendo.",
                category,
                index["categories"][category],
            )
        else:
            index.setdefault("categories", {})[category] = run_id
    save_index(index)


def list_runs() -> list[dict]:
    rows = []
    if not RUNS_DIR.exists():
        return rows
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        mf = d / "manifest.json"
        if mf.exists():
            data = json.loads(mf.read_text(encoding="utf-8"))
            data["run_id"] = d.name
            rows.append(data)
    return rows
