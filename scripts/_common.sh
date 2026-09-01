# shellcheck shell=bash
# Funções compartilhadas pelos scripts de entrada (não executar direto).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pns_activate() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    return 0
  fi
  if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
  elif [[ -f "$ROOT/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/venv/bin/activate"
  else
    echo "Ambiente virtual não encontrado. Crie com: ./scripts/setup_env.sh" >&2
    exit 1
  fi
}

pns_python() {
  pns_activate
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
  python "$@"
}

pns_parse_common() {
  CONFIG="${CONFIG:-configs/experiment.yaml}"
  NAME="${NAME:-}"
  MODELS="${MODELS:-}"
  DATASET="${DATASET:-}"
  FROM_RUN="${FROM_RUN:-}"
  NO_DOWNLOAD="${NO_DOWNLOAD:-0}"
  PORT="${PORT:-8501}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG="$2"
        shift 2
        ;;
      --name)
        NAME="$2"
        shift 2
        ;;
      --models)
        MODELS="$2"
        shift 2
        ;;
      --dataset)
        DATASET="$2"
        shift 2
        ;;
      --from-run|--run-id)
        FROM_RUN="$2"
        shift 2
        ;;
      --no-download)
        NO_DOWNLOAD=1
        shift
        ;;
      --port)
        PORT="$2"
        shift 2
        ;;
      --help|-h)
        return 2
        ;;
      *)
        echo "Flag desconhecida: $1" >&2
        return 2
        ;;
    esac
  done
}

pns_cli_flags() {
  local extra=()
  extra+=(--config "$CONFIG")
  if [[ -n "$NAME" ]]; then
    extra+=(--name "$NAME")
  fi
  if [[ -n "$MODELS" ]]; then
    extra+=(--models "$MODELS")
  fi
  if [[ -n "$DATASET" ]]; then
    extra+=(--dataset "$DATASET")
  fi
  if [[ "$NO_DOWNLOAD" == "1" ]]; then
    extra+=(--no-download)
  fi
  printf '%s\n' "${extra[@]}"
}
