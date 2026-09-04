"""Presets de pré-processamento para bases nomeadas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProfileKind = Literal["pns", "legacy", "custom"]
PresetName = Literal["pns", "legacy"]


@dataclass(frozen=True)
class PreprocessPreset:
    name: str
    profile_kind: ProfileKind


PREPROCESS_PRESETS: dict[PresetName, PreprocessPreset] = {
    "pns": PreprocessPreset(name="pns", profile_kind="pns"),
    "legacy": PreprocessPreset(name="legacy", profile_kind="legacy"),
}


def resolve_preprocess_preset(profile: str | None) -> PreprocessPreset | None:
    if profile is None:
        return None
    key = profile.strip().lower()
    if key not in PREPROCESS_PRESETS:
        valid = ", ".join(sorted(PREPROCESS_PRESETS))
        raise ValueError(f"Perfil desconhecido: {profile!r}. Opções: {valid}")
    return PREPROCESS_PRESETS[key]  # type: ignore[index]


def resolve_prepare_args(
    *,
    profile: str | None,
    config: Path | str,
    name: str,
    dataset_id: str | None,
) -> tuple[Path, str, ProfileKind]:
    """Resolve config path de referência, base_name e profile_kind."""
    preset = resolve_preprocess_preset(profile)
    if preset is not None:
        return (
            Path(f"configs/profiles/{preset.name}.yaml"),
            preset.name,
            preset.profile_kind,
        )

    base_name = dataset_id or name or "pns"
    return Path(config), base_name, "custom"
