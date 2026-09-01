# Contributing — PNS Health Research

## Pré-requisitos

- Python 3.12+ (`.venv`) e opcionalmente Python 3.12 em `.venv-gpu` para cuML
- [uv](https://github.com/astral-sh/uv) recomendado
- Dados PNS 2019 em `data/raw/pns2019.parquet` (ver `docs/data_required.md`)

## Setup

```bash
cd pns-health-research
uv sync --all-extras
uv pip install -e ".[dev]"
```

## Ordem de execução (runbook)

1. `pns-research data prepare --config configs/experiments/legacy_reproduction.yaml`
2. `pns-research experiment --config configs/experiments/legacy_reproduction.yaml`
3. Inspecionar `artifacts/runs/<run_id>/reproduction_report.json`

Resume (classificação após clustering válido):

```bash
pns-research classification --config configs/experiments/legacy_reproduction.yaml --from-run <run_id>
```

## Testes

```bash
make lint
make test
make typecheck
```

## Adicionar modelo ou coluna

Ver `docs/adding_algorithms.md` e `docs/runbook.md`.
