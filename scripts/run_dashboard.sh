#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if ! pns_parse_common "$@"; then
  cat <<'EOF'
Uso: ./scripts/run_dashboard.sh [--port 8501]

Abre o dashboard Streamlit em http://127.0.0.1:<porta>
EOF
  exit 0
fi

pns_activate
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
exec python -m streamlit run "$ROOT/src/pns_health_research/dashboard/app.py" \
  --server.port "$PORT" \
  --server.address 127.0.0.1
