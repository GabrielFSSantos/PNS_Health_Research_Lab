"""Catálogo de edições da PNS (abstração mínima por ano)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pns_health_research.config.registry import (
    ConfigurationError,
    DatasetEntry,
    DatasetsRegistry,
    load_datasets_registry,
)
from pns_health_research.core.exceptions import MissingDatasetError
from pns_health_research.core.paths import PROJECT_ROOT, RAW_DATA_DIR

_LEGACY_DATASET_ALIASES = frozenset({"pns", "pnsibge"})


@dataclass(frozen=True)
class PNSEdition:
    year: int
    status: str
    raw_path: Path
    columns: dict[str, str]
    record_id_columns: list[str]
    survey_weight_column: str
    download_backend: str | None
    message: str | None = None
    key: str = "pns_2019"

    @property
    def is_supported(self) -> bool:
        return self.status == "supported"

    @property
    def columns_config(self) -> dict[str, str]:
        """Compat: mapa de colunas (antes era path YAML)."""
        return self.columns


def load_catalog() -> DatasetsRegistry:
    return load_datasets_registry()


def _to_edition(entry: DatasetEntry) -> PNSEdition:
    raw_rel = entry.raw_path
    raw_path = Path(raw_rel) if Path(raw_rel).is_absolute() else PROJECT_ROOT / raw_rel
    return PNSEdition(
        year=entry.year,
        status=entry.status,
        raw_path=raw_path,
        columns=dict(entry.columns),
        record_id_columns=list(entry.record_id_columns),
        survey_weight_column=entry.survey_weight_column,
        download_backend=entry.download_backend,
        message=entry.message,
        key=entry.key,
    )


def resolve_edition(year: int, dataset: str = "pns") -> PNSEdition:
    try:
        registry = load_datasets_registry()
    except ConfigurationError as exc:
        raise MissingDatasetError(str(exc)) from exc

    entry: DatasetEntry | None = None
    if dataset in registry.datasets:
        entry = registry.get(dataset)
    elif dataset in _LEGACY_DATASET_ALIASES:
        try:
            entry = registry.by_year(year)
        except ConfigurationError as exc:
            raise MissingDatasetError(str(exc)) from exc
    else:
        raise MissingDatasetError(
            f"Dataset '{dataset}' não configurado. Disponíveis: {sorted(registry.datasets)}"
        )

    if entry.status != "supported":
        msg = entry.message or f"PNS {entry.year} ainda não suportada."
        raise MissingDatasetError(msg)

    if dataset not in _LEGACY_DATASET_ALIASES and entry.year != year:
        # Chave explícita (pns_2019) prevalece sobre year divergente.
        pass

    return _to_edition(entry)


def resolve_raw_path(
    *,
    dataset: str = "pns",
    year: int = 2019,
    source: str | Path | None = None,
) -> Path:
    """Resolve caminho do parquet bruto (override explícito ou catálogo)."""
    if source is not None:
        p = Path(source)
        return p if p.is_absolute() else PROJECT_ROOT / p
    edition = resolve_edition(year, dataset)
    return edition.raw_path


def default_raw_candidates(year: int) -> list[Path]:
    """Candidatos legados para compatibilidade."""
    return [
        RAW_DATA_DIR / f"pns_{year}.parquet",
        RAW_DATA_DIR / "pns2019.parquet" if year == 2019 else RAW_DATA_DIR / f"pns{year}.parquet",
    ]


def catalog_as_dict() -> dict[str, Any]:
    """Visão serializável do catálogo (debug / docs)."""
    registry = load_datasets_registry()
    return {
        "schema_version": registry.schema_version,
        "defaults": registry.defaults,
        "datasets": {k: vars(v) for k, v in registry.datasets.items()},
    }
