"""Download explícito de microdados PNS (nunca em import)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pns_health_research.core.exceptions import MissingDatasetError
from pns_health_research.core.paths import INTERIM_DATA_DIR, PROJECT_ROOT
from pns_health_research.data.acquisition import _parquet_safe_dataframe
from pns_health_research.data.catalog import PNSEdition, resolve_edition

logger = logging.getLogger("pns_health_research")


def _fingerprint_path(year: int) -> Path:
    return INTERIM_DATA_DIR / f"pns_{year}_raw_fingerprint.json"


def compute_file_fingerprint(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "size_bytes": path.stat().st_size}


def validate_raw_parquet(path: Path, edition: PNSEdition) -> None:
    if not path.exists():
        raise MissingDatasetError(f"Arquivo ausente: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise MissingDatasetError(f"Parquet inválido: {path}") from exc
    if len(df) < 100:
        raise MissingDatasetError(f"Parquet suspeito (poucas linhas): {path}")
    col_map = edition.columns
    required = list(col_map.keys())[:5]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingDatasetError(
            f"Colunas mínimas ausentes em {path}: {missing}"
        )


def _download_pnsibge(year: int, output: Path) -> Path:
    try:
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise MissingDatasetError(
            "Download via PNSIBGE requer o extra [acquisition] (rpy2) e R com o pacote PNSIBGE. "
            "Instale com: ./scripts/setup_env.sh --acquisition "
            "e no R: install.packages('PNSIBGE'). "
            "Veja README.md (seção Primeira vez)."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".parquet.tmp")
    try:
        pnsibge = importr("PNSIBGE")
        df_r = pnsibge.get_pns(year=year, design=False)
        import rpy2.robjects.pandas2ri as pandas2ri
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter

        with localconverter(default_converter + pandas2ri.converter):
            df = pandas2ri.rpy2py(df_r)
        df = _parquet_safe_dataframe(df)
        df.to_parquet(tmp, index=False)
        tmp.replace(output)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return output


def download_pns(
    year: int = 2019,
    *,
    dataset: str = "pns",
    force: bool = False,
) -> Path:
    """Baixa PNS do ano via backend do catálogo. Valida integridade."""
    edition = resolve_edition(year, dataset)
    output = edition.raw_path
    if output.exists() and not force:
        validate_raw_parquet(output, edition)
        fp = compute_file_fingerprint(output)
        _fingerprint_path(year).parent.mkdir(parents=True, exist_ok=True)
        _fingerprint_path(year).write_text(json.dumps(fp, indent=2), encoding="utf-8")
        logger.info("Dataset já existe e é válido: %s", output)
        return output

    backend = edition.download_backend
    if backend == "pnsibge":
        logger.info("Baixando PNS %s via PNSIBGE/R...", year)
        path = _download_pnsibge(year, output)
    else:
        raise MissingDatasetError(f"Backend de download não configurado para PNS {year}.")

    validate_raw_parquet(path, edition)
    fp = compute_file_fingerprint(path)
    _fingerprint_path(year).parent.mkdir(parents=True, exist_ok=True)
    _fingerprint_path(year).write_text(json.dumps(fp, indent=2), encoding="utf-8")
    logger.info("Download concluído: %s (sha256=%s...)", path, fp["sha256"][:12])
    return path


def ensure_data_available(
    year: int = 2019,
    *,
    dataset: str = "pns",
    source: str | Path | None = None,
) -> Path:
    """Usado por --download-if-missing: não baixa se override path existir."""
    if source is not None:
        p = Path(source)
        path = p if p.is_absolute() else PROJECT_ROOT / p
        if not path.exists():
            raise MissingDatasetError(f"source configurado mas ausente: {path}")
        return path
    edition = resolve_edition(year, dataset)
    if edition.raw_path.exists():
        validate_raw_parquet(edition.raw_path, edition)
        return edition.raw_path
    return download_pns(year, dataset=dataset)
