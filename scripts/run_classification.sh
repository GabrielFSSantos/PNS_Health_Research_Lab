#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG="${CONFIG:-configs/profiles/pns.yaml}"
NAME="${NAME:-pns}"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_classification.sh [--profile pns|legacy] [--name NOME] [--source-base BASE] [--config PATH] [--models a,b]

Presets (recomendado):
  ./scripts/run_classification.sh --profile pns      # lê pns/clustering/
  ./scripts/run_classification.sh --profile legacy   # TCC

Fluxo mínimo (defaults: --name pns):
  ./scripts/run_classification.sh
EOF
  exit 0
fi

mapfile -t FLAGS < <(pns_cli_flags)
pns_python -m pns_health_research classification run "${FLAGS[@]}"
