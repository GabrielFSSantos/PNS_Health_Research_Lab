#!/usr/bin/env python3
"""Gera configs/data/dictionary.yaml a partir do mapa de colunas PNS."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_PATH = PROJECT_ROOT / "configs" / "datasets.yaml"
OUT_PATH = PROJECT_ROOT / "configs" / "data" / "dictionary.yaml"

CLUSTERING_KEYWORDS = (
    "Limita",
    "Hipertensão",
    "Diabetes",
    "Asma",
    "Depressão",
    "Câncer",
    "Cardiopatia",
    "AVC",
    "Artrite",
    "DORT",
    "Dor na Coluna",
    "Colesterol",
    "Pulmão",
    "Renal",
    "Mental",
)


def _infer_type(name: str) -> str:
    if name in ("Sexo", "Raça", "Estado Civil", "UF", "Zona", "Extrato"):
        return "categorical"
    if name in ("Idade", "Peso", "Altura", "IMC", "Renda"):
        return "numeric"
    if name in ("Serviço de Saúde", "Internação", "Plano de Saúde"):
        return "binary"
    if name == "Estado de Saúde":
        return "ordinal"
    if "Limita" in name:
        return "binary"
    return "binary"


def main() -> None:
    raw = yaml.safe_load(DATASETS_PATH.read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets") or {}
    pns_2019 = datasets.get("pns_2019") or {}
    columns_map: dict = pns_2019.get("columns") or {}
    entries = []
    for code, label in columns_map.items():
        if not isinstance(label, str):
            continue
        used_clustering = any(k in label for k in CLUSTERING_KEYWORDS) or label in (
            "Sexo",
            "Idade",
            "Estado de Saúde",
            "IMC",
        )
        used_classification = label not in ("Peso (V0091)", "Ordem do Domicílio", "Número de Ordem")
        entries.append(
            {
                "code": code,
                "internal_name": label,
                "friendly_name": label,
                "description": f"Variável PNS mapeada de {code}",
                "type": _infer_type(label),
                "used_clustering": used_clustering,
                "used_classification": used_classification,
                "derived": label == "IMC",
            }
        )
    entries.append(
        {
            "code": "Cluster",
            "internal_name": "Cluster",
            "friendly_name": "Cluster de agrupamento",
            "description": "Label do algoritmo de clustering",
            "type": "categorical",
            "used_clustering": True,
            "used_classification": True,
            "derived": True,
        }
    )
    out = {"schema_version": 1, "columns": entries}
    OUT_PATH.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Dictionary gerado: {OUT_PATH} ({len(entries)} colunas)")


if __name__ == "__main__":
    main()
