#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACQUISITION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --acquisition)
      ACQUISITION=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Uso: ./scripts/setup_env.sh [--acquisition]

Cria .venv e instala o pacote com extras [dev,dashboard].
R (PNSIBGE) não é necessário para o dashboard nem para o pipeline
se o parquet já existir em data/raw/.

  --acquisition   também instala rpy2 (requer R + pacote PNSIBGE)
EOF
      exit 0
      ;;
    *)
      echo "Flag desconhecida: $1" >&2
      exit 1
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"
if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON" -m venv "$ROOT/.venv"
fi
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"
pip install -U pip
if [[ "$ACQUISITION" == "1" ]]; then
  pip install -e ".[dev,dashboard,acquisition]"
else
  pip install -e ".[dev,dashboard]"
fi
echo "Ambiente pronto. Ative com: source .venv/bin/activate"
if [[ "$ACQUISITION" == "1" ]]; then
  echo ""
  echo "Próximos passos (primeira vez com dados reais):"
  echo "  1. No R: install.packages('PNSIBGE')"
  echo "  2. ./scripts/run_download.sh"
  echo "  3. ./scripts/run_experiment.sh --name \"Minha primeira run\""
  echo "  4. ./scripts/run_dashboard.sh"
else
  echo ""
  echo "Instalação sem [acquisition]: adequada para dashboard + showcase."
  echo "Para baixar a PNS via R, recrie o ambiente com: ./scripts/setup_env.sh --acquisition"
  echo "Primeira execução (com parquet): ./scripts/run_experiment.sh"
fi
