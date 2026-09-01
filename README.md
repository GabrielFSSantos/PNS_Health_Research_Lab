# PNS Health Research Lab

Plataforma para rodar experimentos com dados da **Pesquisa Nacional de Saúde (PNS)**, comparar configurações e visualizar resultados no dashboard — em apoio à pesquisa sobre **utilização dos serviços de saúde**.

## Hipótese

Perfis de saúde (comorbidades e hábitos) se agrupam de forma estável e ajudam a prever o uso de serviços de saúde na PNS 2019. O laboratório reproduz o baseline do TCC e permite ligar/desligar etapas, modelos e variáveis derivadas via YAML.

## O que o lab produz

1. Pré-processa microdados PNS (colunas, filtros, features derivadas)  
2. Agrupa perfis de saúde (PCA + K-Means)  
3. Classifica uso de serviços (vários modelos, validação cruzada)  
4. Grava artefatos em `artifacts/runs/` e exibe **todas as runs concluídas** no dashboard  

## Requisitos

- Python 3.12+
- `pip` + venv (script `setup_env.sh`)
- Para **baixar** a PNS: R + pacote **PNSIBGE** + `./scripts/setup_env.sh --acquisition`
- GPU opcional: extra `[gpu]`

## Primeira vez (do zero)

1. Clone o repositório e entre na pasta:
   ```bash
   git clone <url>
   cd PNS_Health_Research_Lab
   ```

2. Instale **R** e o pacote PNSIBGE (uma vez no R):
   ```r
   install.packages("PNSIBGE")
   ```

3. Crie o ambiente Python com suporte a download:
   ```bash
   ./scripts/setup_env.sh --acquisition
   ```

4. Baixe os microdados (parquet em `data/raw/`):
   ```bash
   ./scripts/run_download.sh
   ```
   Outro ano/dataset: `--dataset pns_2019` (veja `configs/datasets.yaml`).

5. Rode um experimento completo:
   ```bash
   ./scripts/run_experiment.sh --name "Minha primeira run"
   ```
   O passo 4 pode ser omitido: se o parquet não existir, o experimento tenta baixar automaticamente (`if_missing: true` no YAML).

6. Abra o dashboard:
   ```bash
   ./scripts/run_dashboard.sh
   ```
   Acesse `http://127.0.0.1:8501` — a run aparece no seletor.

**Só explorar o dashboard?** Sem R nem parquet, o showcase em `artifacts/showcase/` já alimenta o seletor. Instale com `./scripts/setup_env.sh` (sem `--acquisition`) e rode `./scripts/run_dashboard.sh`.

Reprodução do TCC:
```bash
./scripts/run_experiment.sh \
  --config configs/experiments/legacy.yaml \
  --name "Reprodução TCC — baseline"
```

## Scripts

| Script | O que faz |
|--------|-----------|
| `./scripts/setup_env.sh` | Cria `.venv` e instala `[dev,dashboard]` |
| `./scripts/setup_env.sh --acquisition` | Inclui `rpy2` para download PNSIBGE |
| `./scripts/run_download.sh` | Baixa parquet PNS via R (requer acquisition) |
| `./scripts/run_preprocess.sh` | Download se faltar + pré-processamento |
| `./scripts/run_clustering.sh` | Agrupamento (pré-processa na mesma run) |
| `./scripts/run_classification.sh` | Classificação (`--from-run` opcional) |
| `./scripts/run_experiment.sh` | Pipeline completa segundo o YAML |
| `./scripts/run_dashboard.sh` | Streamlit em `http://127.0.0.1:8501` |

Flags comuns: `--config`, `--dataset`, `--models`, `--name`, `--no-download`.

## Como escolher parâmetros

| YAML | Efeito |
|------|--------|
| `configs/datasets.yaml` | Ano PNS, colunas, download |
| `configs/preprocessing.yaml` | Etapas `enabled`, derivadas, pesos |
| `configs/clustering.yaml` | PCA, K-Means, `selection_method` |
| `configs/models.yaml` | Classificadores, `param_grid`, gridsearch |
| `configs/experiment.yaml` | Orquestração, seed, estágios do pipeline |
| `configs/experiments/legacy.yaml` | Overlay só com diffs do TCC |

Detalhes: [docs/documentacao.md](docs/documentacao.md).

## Como comparar runs

- **Dashboard → Evolução** — gráfico F1+ entre as runs  
- **Dashboard → Comparar classificação / agrupamento**  
- Artefatos em `artifacts/runs/`

## Limpar runs antigas

```bash
python scripts/prune_test_runs.py --profile custom --dry-run
python scripts/prune_test_runs.py --profile custom --delete --yes
```

## Baseline commitada

A reprodução do TCC está em `artifacts/showcase/pesquisa-original-legacy/` para quem clona sem reexecutar.

## Testes

```bash
source .venv/bin/activate
ruff check src tests
pytest tests/ -m "not gpu" -q
```

## Documentação

- **[docs/README.md](docs/README.md)** — índice  
- **[docs/documentacao.md](docs/documentacao.md)** — módulos, YAML, scripts, dashboard, testes  
- **[docs/trajetoria.md](docs/trajetoria.md)** — TCC, reprodução e decisões  

## Estrutura

```
configs/                 # um YAML por módulo + overlay legacy
scripts/                 # entrada principal
artifacts/runs/          # suas execuções
artifacts/showcase/      # baseline TCC versionada no Git
src/pns_health_research/ # pacote Python
docs/
```
