#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

FORCE=0
YEAR=2019

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --year)
      YEAR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Uso: ./scripts/run_download.sh [--dataset pns_2019] [--year 2019] [--force]

Baixa microdados PNS via PNSIBGE (R + rpy2). Requer:
  ./scripts/setup_env.sh --acquisition
  R com install.packages("PNSIBGE")

O parquet é gravado em data/raw/ conforme configs/datasets.yaml.
EOF
      exit 0
      ;;
    *)
      echo "Flag desconhecida: $1" >&2
      exit 1
      ;;
  esac
done

pns_activate
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

args=(data download --year "$YEAR")
if [[ -n "${DATASET:-}" ]]; then
  args+=(--dataset "$DATASET")
fi
if [[ "$FORCE" == "1" ]]; then
  args+=(--force)
fi

python -m pns_health_research "${args[@]}"
