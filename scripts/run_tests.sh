#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODE="${1:-fast}"

case "$MODE" in
  fast)
    # Suíte rápida (unit + integração sintética), paralela quando xdist disponível
    if pns_python -c "import xdist" 2>/dev/null; then
      pns_python -m pytest -n auto -m "not regression and not regression_smoke"
    else
      pns_python -m pytest -m "not regression and not regression_smoke"
    fi
    ;;
  smoke)
    pns_python -m pytest -m regression_smoke
    ;;
  full|regression)
    pns_python -m pytest -m regression
    ;;
  all)
    if pns_python -c "import xdist" 2>/dev/null; then
      pns_python -m pytest -n auto -m "not regression and not regression_smoke"
    else
      pns_python -m pytest -m "not regression and not regression_smoke"
    fi
    pns_python -m pytest -m regression_smoke
    pns_python -m pytest -m regression
    ;;
  --help|-h|help)
    cat <<'EOF'
Uso: ./scripts/run_tests.sh [fast|smoke|full|all]

  fast        (default) unit + integração sintética; usa pytest-xdist se instalado
  smoke       regression_smoke — lê artefatos em data/bases/ (~segundos)
  full        regression E2E no PNS real (enxuto; ~minutos)
  all         fast + smoke + full
EOF
    exit 0
    ;;
  *)
    echo "Modo desconhecido: $MODE (use fast|smoke|full|all)" >&2
    exit 1
    ;;
esac
