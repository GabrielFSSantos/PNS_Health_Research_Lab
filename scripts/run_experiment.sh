#!/usr/bin/env bash
set -euo pipefail

# Orquestrador modular: preprocess → clustering → classification por --profile.
# Default: --profile pns (pipeline rigoroso; ver docs/trajetoria.md §2.1).

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PROFILE="${PROFILE:-pns}"
RAW="$ROOT/data/raw/pns2019.parquet"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_experiment.sh [opções]

Orquestra os módulos ativos para a base --profile (default: pns):
  run_preprocess.sh → run_clustering.sh → run_classification.sh

Presets:
  ./scripts/run_experiment.sh                    # --profile pns
  ./scripts/run_experiment.sh --profile legacy   # reprodução TCC (bases modulares)
  ./scripts/run_experiment.sh --all-bases        # pns + legacy em sequência

Flags:
  --profile pns|legacy     Base e configs (default: pns)
  --all-bases              Executa pns e legacy
  --download               Baixa raw se ausente (antes do preprocess)
  --skip-preprocess        Resume após preprocess
  --skip-clustering        Resume após clustering
  --skip-classification    Só preprocess + clustering
  --models a,b             Repassa à classificação
  --no-download            Não tenta download automático no preprocess

Reprodução TCC (base modular):
  ./scripts/run_experiment.sh --profile legacy

Configurações: configs/profiles/ (ver configs/README.md).
EOF
  exit 0
fi

run_profile_pipeline() {
  local profile="$1"
  echo "==> Pipeline modular (--profile $profile)"

  if [[ "$DOWNLOAD" == "1" ]] && [[ ! -f "$RAW" ]]; then
    echo "Parquet ausente: $RAW — download (--download)..."
    "$ROOT/scripts/run_download.sh"
  fi

  local pp_flags=(--profile "$profile")
  local cl_flags=(--profile "$profile")
  local cf_flags=(--profile "$profile")

  if [[ "$NO_DOWNLOAD" == "1" ]]; then
    pp_flags+=(--no-download)
  fi
  if [[ -n "$MODELS" ]]; then
    cf_flags+=(--models "$MODELS")
  fi

  if [[ "$SKIP_PREPROCESS" != "1" ]]; then
    "$ROOT/scripts/run_preprocess.sh" "${pp_flags[@]}"
  fi
  if [[ "$SKIP_CLUSTERING" != "1" ]]; then
    "$ROOT/scripts/run_clustering.sh" "${cl_flags[@]}"
  fi
  if [[ "$SKIP_CLASSIFICATION" != "1" ]]; then
    "$ROOT/scripts/run_classification.sh" "${cf_flags[@]}"
  fi
}

if [[ "$ALL_BASES" == "1" ]]; then
  run_profile_pipeline pns
  run_profile_pipeline legacy
else
  run_profile_pipeline "${PROFILE:-pns}"
fi

echo "==> Bases materializadas:"
pns_python -m pns_health_research data bases list
