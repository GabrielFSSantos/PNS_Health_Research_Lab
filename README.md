# PNS Health Research Lab

Plataforma para rodar experimentos com dados da **Pesquisa Nacional de Saúde (PNS)**, comparar configurações e visualizar resultados no dashboard — em apoio à pesquisa sobre **utilização dos serviços de saúde**.

## Entendendo a pesquisa

Perfis de saúde (comorbidades e hábitos) se agrupam de forma estável e ajudam a prever o uso de serviços de saúde na PNS 2019. O laboratório reproduz o baseline do TCC e permite evoluir o protocolo via perfis YAML.

- **Narrativa completa:** [docs/trajetoria.md](docs/trajetoria.md)
- **Referência técnica:** [docs/documentacao.md](docs/documentacao.md)
- **Mapa de configs:** [configs/README.md](configs/README.md)

## Primeira vez

```bash
./scripts/setup_env.sh --acquisition    # + --gpu para CUDA/cuML
./scripts/run_download.sh
./scripts/run_experiment.sh --profile pns --download
```

Reprodução TCC:

```bash
./scripts/run_experiment.sh --profile legacy
```

## Configurações (simples)

Só dois perfis oficiais em [`configs/profiles/`](configs/profiles/):

| Perfil | Arquivo | Uso |
|--------|---------|-----|
| `pns` | `configs/profiles/pns.yaml` | Rigoroso; GPU auto |
| `legacy` | `configs/profiles/legacy.yaml` | TCC; CPU |

Customização: [`configs/overrides/`](configs/overrides/). Detalhes: [configs/README.md](configs/README.md).

## Testes

```bash
./scripts/run_tests.sh          # rápido (~1 min)
./scripts/run_tests.sh smoke    # artefatos em data/bases/ (~30 s)
./scripts/run_tests.sh full     # E2E PNS real (enxuto)
```

Equivalente: `pytest`, `pytest -m regression_smoke`, `pytest -m regression`.

Durante a execução dos módulos, o terminal mostra progresso por etapa e tempo (`PNS_QUIET=1` desliga; `PNS_PROGRESS=1` força em ambientes sem TTY).

## Scripts principais

| Script | Função |
|--------|--------|
| `./scripts/run_experiment.sh` | Pipeline completo (`--profile pns\|legacy`) |
| `./scripts/run_tests.sh` | Testes em 3 níveis |
| `./scripts/setup_env.sh` | Venv; `--gpu` e `--acquisition` opcionais |

## Estrutura

```
configs/profiles/     # pns.yaml, legacy.yaml (único entry point por perfil)
src/pns_health_research/modules/
data/bases/           # saída modular
```

Documentação: [docs/README.md](docs/README.md).
