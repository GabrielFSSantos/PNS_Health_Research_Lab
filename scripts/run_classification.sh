#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_classification.sh [--config PATH] [--models a,b] [--from-run ID] [--name TEXTO] [--dataset KEY] [--no-download]

Roda classificação. Sem --from-run, pré-processa antes.
Com --from-run, reutiliza clustered_categorical da run indicada.
EOF
  exit 0
fi

mapfile -t FLAGS < <(pns_cli_flags)
if [[ -n "$FROM_RUN" ]]; then
  FLAGS+=(--from-run "$FROM_RUN")
fi
pns_python -m pns_health_research classification run "${FLAGS[@]}"
