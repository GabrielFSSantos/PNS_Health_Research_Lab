#!/usr/bin/env python3
"""Remove ou arquiva runs locais antigas em artifacts/runs/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from pns_health_research.core.paths import CONFIGS_DIR, RUNS_DIR

ARCHIVE_DIR = RUNS_DIR / "_archive"
OVERRIDES_PATH = CONFIGS_DIR / "dashboard" / "run_overrides.yaml"

V2_RUNS_TO_DELETE = [
    "20260901T062634Z_research_v2_d291447",
    "20260901T063517Z_research_v2_df3010a",
    "20260901T081819Z_research_v2_80cabac",
    "20260901T095258Z_research_v2_c9b50e6",
    "20260901T095706Z_research_v2_603718f",
]


def _protected_keys() -> set[str]:
    keys: set[str] = set()
    if OVERRIDES_PATH.exists():
        data = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
        for item in data.get("overrides", []):
            key = item.get("key") or item.get("run_id") or item.get("label")
            if key:
                keys.add(key)
    return keys


def _run_profile(run_dir: Path) -> str | None:
    manifest = run_dir / "manifest.json"
    summary = run_dir / "summary.json"
    for path in (summary, manifest):
        if path.exists():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("profile")
    return None


def _should_preserve(run_dir: Path, protected: set[str]) -> bool:
    if run_dir.name in protected:
        return True
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("preserve"):
            return True
        tags = summary.get("tags") or []
        if "baseline" in tags or "preserve" in tags:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Apenas listar")
    parser.add_argument("--yes", action="store_true", help="Confirmar sem prompt")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Apagar permanentemente (padrão: mover para _archive)",
    )
    parser.add_argument(
        "--profile",
        action="append",
        help="Filtrar por profile (ex.: research_v2). Pode repetir.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Run específica para remover. Pode repetir.",
    )
    parser.add_argument(
        "--v2-failed",
        action="store_true",
        help="Remover runs research_v2 listadas no plano (abaixo do baseline)",
    )
    args = parser.parse_args()

    if not RUNS_DIR.exists():
        print("Nenhum diretório de runs.")
        return

    protected = _protected_keys()
    profiles = set(args.profile or [])
    explicit_ids = set(args.run_ids or [])
    if args.v2_failed:
        explicit_ids.update(V2_RUNS_TO_DELETE)

    candidates = [
        p
        for p in RUNS_DIR.iterdir()
        if p.is_dir() and p.name not in ("_archive",) and not p.name.startswith(".")
    ]

    to_process: list[Path] = []
    for run_dir in sorted(candidates):
        if _should_preserve(run_dir, protected):
            continue
        if explicit_ids and run_dir.name not in explicit_ids:
            continue
        if profiles:
            prof = _run_profile(run_dir)
            if prof not in profiles:
                continue
        if not explicit_ids and not profiles:
            continue
        to_process.append(run_dir)

    if not to_process:
        print("Nenhuma run corresponde aos critérios.")
        return

    action = "apagar" if args.delete else "arquivar"
    print(f"Runs a {action} ({len(to_process)}):")
    for p in to_process:
        print(f"  - {p.name}")

    if args.dry_run:
        return
    if not args.yes:
        print(f"\nUse --yes para confirmar. Nada foi {action[:-1]}ado.")
        return

    if not args.delete:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for run_dir in to_process:
        if args.delete:
            shutil.rmtree(run_dir)
            print(f"Apagado: {run_dir.name}")
        else:
            dest = ARCHIVE_DIR / run_dir.name
            if dest.exists():
                print(f"Pulando {run_dir.name} (já existe em _archive)")
                continue
            shutil.move(str(run_dir), str(dest))
            print(f"Arquivado: {run_dir.name}")


if __name__ == "__main__":
    main()
