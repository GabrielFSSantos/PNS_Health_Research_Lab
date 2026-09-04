#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG="${CONFIG:-configs/profiles/pns.yaml}"
NAME="${NAME:-pns}"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_preprocess.sh [--profile pns|legacy] [--name NOME] [--config PATH] [--dataset KEY] [--no-download]

Presets (recomendado):
  ./scripts/run_preprocess.sh --profile pns      # base rigorosa (PRE-*)
  ./scripts/run_preprocess.sh --profile legacy   # reprodução TCC

Fluxo mínimo (defaults: --name pns):
  ./scripts/run_preprocess.sh

Perfil customizado:
  ./scripts/run_preprocess.sh --name sabesp --config configs/profiles/pns.yaml

Grava/sobrescreve data/bases/<NOME>/preprocessing/
Registra última execução em data/bases/<NOME>/registry.json
Baixa PNS automaticamente se o parquet não existir.
EOF
  exit 0
fi

mapfile -t FLAGS < <(pns_cli_flags)
pns_python -m pns_health_research data prepare "${FLAGS[@]}"
