#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_preprocess.sh [--config PATH] [--dataset KEY] [--no-download]

Baixa o parquet PNS se if_missing=true no YAML e roda o pré-processamento
(somente estágios enabled em configs/preprocessing.yaml).
EOF
  exit 0
fi

mapfile -t FLAGS < <(pns_cli_flags)
pns_python -m pns_health_research data prepare "${FLAGS[@]}"
