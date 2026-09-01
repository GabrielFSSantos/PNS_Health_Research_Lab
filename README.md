# PNS Health Research

Plataforma experimental reproduzível para pesquisa com PNS 2019 — laboratório de ML em saúde.

[![CI](https://github.com/OWNER/pns-health-research/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/pns-health-research/actions/workflows/ci.yml)

## Setup

```bash
cd pns-health-research
make install
# ou: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

## Runbook

Ver [docs/runbook.md](docs/runbook.md) para ordem canônica de execução.

```bash
.venv/bin/pns-research experiment --config configs/experiments/legacy_reproduction.yaml
```

## GPU (opcional)

```bash
make install-gpu
make run-v2-gpu
```

Ver [docs/gpu_setup.md](docs/gpu_setup.md).

## Publicar no Git remoto

```bash
git remote add origin <url-do-repositorio>
git push -u origin main
```

Substitua `OWNER` no badge CI após criar o repositório no GitHub.

## Documentação

- [Architecture](docs/architecture.md)
- [Runbook](docs/runbook.md)
- [GPU setup](docs/gpu_setup.md)
- [Baseline contract](docs/baseline_contract.md)
- [Known discrepancies](docs/known_discrepancies.md)

## Testes

```bash
make test
make lint
```
