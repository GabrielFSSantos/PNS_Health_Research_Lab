#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_clustering.sh [--config PATH] [--dataset KEY] [--name TEXTO] [--no-download]

Roda agrupamento (PCA + K-Means). Pré-processa na mesma run.
EOF
  exit 0
fi

mapfile -t FLAGS < <(pns_cli_flags)
pns_python -m pns_health_research clustering run "${FLAGS[@]}"
