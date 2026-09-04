"""Presets de agrupamento para bases nomeadas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProfileKind = Literal["pns", "legacy", "custom"]
PresetName = Literal["pns", "legacy"]


@dataclass(frozen=True)
class ClusteringPreset:
    name: str
    source_base: str
    profile_kind: ProfileKind


CLUSTERING_PRESETS: dict[PresetName, ClusteringPreset] = {
    "pns": ClusteringPreset(name="pns", source_base="pns", profile_kind="pns"),
    "legacy": ClusteringPreset(name="legacy", source_base="legacy", profile_kind="legacy"),
}


def resolve_clustering_preset(profile: str | None) -> ClusteringPreset | None:
    if profile is None:
        return None
    key = profile.strip().lower()
    if key not in CLUSTERING_PRESETS:
        valid = ", ".join(sorted(CLUSTERING_PRESETS))
        raise ValueError(f"Perfil desconhecido: {profile!r}. Opções: {valid}")
    return CLUSTERING_PRESETS[key]  # type: ignore[index]


def resolve_clustering_args(
    *,
    profile: str | None,
    config: Path | str,
    name: str,
    source_base: str | None,
) -> tuple[Path, str, str, ProfileKind]:
    preset = resolve_clustering_preset(profile)
    if preset is not None:
        return (
            Path(f"configs/profiles/{preset.name}.yaml"),
            preset.name,
            preset.source_base,
            preset.profile_kind,
        )

    base_name = name or "pns"
    src = source_base or base_name
    return Path(config), base_name, src, "custom"
