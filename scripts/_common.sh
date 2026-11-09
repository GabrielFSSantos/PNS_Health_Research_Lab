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
  local config_explicit=0
  local name_explicit=0

  CONFIG="${CONFIG:-configs/profiles/pns.yaml}"
  NAME="${NAME:-}"
  PROFILE="${PROFILE:-}"
  MODELS="${MODELS:-}"
  DATASET="${DATASET:-}"
  FROM_RUN="${FROM_RUN:-}"
  SOURCE_BASE="${SOURCE_BASE:-}"
  NO_DOWNLOAD="${NO_DOWNLOAD:-0}"
  FORCE="${FORCE:-0}"
  PORT="${PORT:-8501}"
  ALL_BASES="${ALL_BASES:-0}"
  DOWNLOAD="${DOWNLOAD:-0}"
  SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
  SKIP_CLUSTERING="${SKIP_CLUSTERING:-0}"
  SKIP_CLASSIFICATION="${SKIP_CLASSIFICATION:-0}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG="$2"
        config_explicit=1
        shift 2
        ;;
      --profile)
        PROFILE="$2"
        shift 2
        ;;
      --name)
        NAME="$2"
        name_explicit=1
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
      --source-base)
        SOURCE_BASE="$2"
        shift 2
        ;;
      --no-download)
        NO_DOWNLOAD=1
        shift
        ;;
      --download)
        DOWNLOAD=1
        shift
        ;;
      --all-bases)
        ALL_BASES=1
        shift
        ;;
      --skip-preprocess)
        SKIP_PREPROCESS=1
        shift
        ;;
      --skip-clustering)
        SKIP_CLUSTERING=1
        shift
        ;;
      --skip-classification)
        SKIP_CLASSIFICATION=1
        shift
        ;;
      --force)
        FORCE=1
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

  # --profile define config e nome da base salvo override explícito na linha de comando.
  if [[ -n "$PROFILE" ]]; then
    if [[ "$config_explicit" -eq 0 ]]; then
      CONFIG="configs/profiles/${PROFILE}.yaml"
    fi
    if [[ "$name_explicit" -eq 0 ]]; then
      NAME="$PROFILE"
    fi
  elif [[ -z "$NAME" ]]; then
    NAME="pns"
  fi
}

pns_cli_flags() {
  local extra=()
  if [[ -n "$PROFILE" ]]; then
    extra+=(--profile "$PROFILE")
  fi
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
  if [[ -n "$SOURCE_BASE" ]]; then
    extra+=(--source-base "$SOURCE_BASE")
  fi
  if [[ "$NO_DOWNLOAD" == "1" ]]; then
    extra+=(--no-download)
  fi
  if [[ "$FORCE" == "1" ]]; then
    extra+=(--force)
  fi
  printf '%s\n' "${extra[@]}"
}
