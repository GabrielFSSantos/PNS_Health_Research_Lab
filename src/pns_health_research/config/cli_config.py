"""Helpers CLI para carregar configs de perfil unificado ou YAML legado."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pns_health_research.config.loader import load_config
from pns_health_research.config.profile_resolver import (
    _path_is_override,
    module_entry_path,
    resolve_module_config,
)
from pns_health_research.config.profile_resolver import ModuleName


def load_module_config(
    *,
    profile: str | None,
    module: ModuleName,
    config: Path,
    cli_models: list[str] | None = None,
    cli_dataset: str | None = None,
) -> tuple[Any, str]:
    if profile:
        override = config if _path_is_override(config) else None
        cfg = resolve_module_config(
            profile,
            module,
            override_path=override,
            cli_models=cli_models,
            cli_dataset=cli_dataset,
        )
        return cfg, module_entry_path(profile, module)
    cfg = load_config(config, cli_models=cli_models, cli_dataset=cli_dataset)
    return cfg, str(config)
