"""Resolução de nomes de base (sem versionamento _vN)."""

from __future__ import annotations

import re

_BUNDLE_ID_RE = re.compile(r"^(?P<prefix>.+)_v(?P<version>\d+)$")


def parse_bundle_id(bundle_id: str) -> tuple[str, int] | None:
    """Extrai (prefixo, versão) de IDs legados no formato ``prefix_vN``."""
    match = _BUNDLE_ID_RE.match(bundle_id)
    if not match:
        return None
    return match.group("prefix"), int(match.group("version"))


def resolve_bundle_id(
    prefix: str,
    *,
    force: bool = False,
    processed_dir=None,
) -> str:
    """Retorna o nome fixo da base (sem sufixo versionado)."""
    name = prefix.strip()
    if not name:
        raise ValueError("Nome da base não pode ser vazio")
    return name
