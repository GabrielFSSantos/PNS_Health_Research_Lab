#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_experiment.sh [--config PATH] [--dataset KEY] [--models a,b] [--name TEXTO] [--no-download]

Pipeline completa conforme experiment.yaml (estágios enabled).

Reprodução TCC:
  ./scripts/run_experiment.sh --config configs/experiments/legacy.yaml --name "Reprodução TCC"
EOF
  exit 0
fi

mapfile -t FLAGS < <(pns_cli_flags)
pns_python -m pns_health_research experiment run "${FLAGS[@]}"
