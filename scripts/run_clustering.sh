#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG="${CONFIG:-configs/profiles/pns.yaml}"
NAME="${NAME:-pns}"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_clustering.sh [--profile pns|legacy] [--name NOME] [--source-base BASE] [--config PATH]

Presets (recomendado):
  ./scripts/run_clustering.sh --profile pns      # lê pns/preprocessing/
  ./scripts/run_clustering.sh --profile legacy   # lê legacy/preprocessing/ (TCC)

Fluxo mínimo (defaults: --name pns):
  ./scripts/run_clustering.sh

Perfil customizado:
  ./scripts/run_clustering.sh --name sabesp --source-base sabesp --config configs/profiles/pns.yaml

Lê data/bases/<source-base>/preprocessing/ e grava/sobrescreve
data/bases/<NOME>/clustering/
EOF
  exit 0
fi

mapfile -t FLAGS < <(pns_cli_flags)
pns_python -m pns_health_research clustering run "${FLAGS[@]}"
