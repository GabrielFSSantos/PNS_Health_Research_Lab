# Documentação técnica

Pacote Python em `src/pns_health_research/`. A entrada principal para quem usa o laboratório são os scripts em `scripts/`; a CLI Typer (`python -m pns_health_research`) é shim interno.

## 1. Hipótese e fluxo

A PNS 2019 descreve saúde, hábitos e uso de serviços. O pipeline:

1. Seleciona e renomeia colunas (`configs/datasets.yaml`)
2. Filtra adultos e gera bases binária e categórica (`configs/preprocessing.yaml`)
3. Agrupa comorbidades via PCA + K-Means (`configs/clustering.yaml`)
4. Classifica uso de serviço de saúde (`configs/models.yaml`)
5. Persiste artefatos em `artifacts/runs/<run_id>/` e alimenta o dashboard

Dimensões do baseline TCC (PNS 2019, overlay legado): 87.678 linhas, 161 colunas binárias, 61 categóricas.

## 2. Módulos

| Módulo | Papel |
|--------|--------|
| `config/` | Loader YAML, registries, Pydantic |
| `data/` | Catálogo, download PNSIBGE, loader, fingerprint |
| `preprocessing/` | Filtros, one-hot, `derived.py`, pesos |
| `dimensionality/` | PCA |
| `clustering/` | Registry de clusterers, K-Means, avaliação |
| `classification/` | Registry de classificadores, CV, gridsearch |
| `statistics/` | Qui-quadrado, Cramér's V, IC bootstrap |
| `experiments/` | Runner e relatório de reprodução |
| `persistence/` | Manifest, artifacts, showcase |
| `dashboard/` | Streamlit (páginas → services → query) |

### Extensibilidade

- **Novo classificador:** `@register_classifier` em `classification/registry.py` + entrada em `configs/models.yaml` (`enabled`, `param_grid`).
- **Novo clusterer:** `@register_clusterer` em `clustering/registry.py` + `estimator.name` em `configs/clustering.yaml`.
- **Nova variável derivada:** função em `preprocessing/derived.py`, chave no dict `BINARY_DERIVED` / `CATEGORICAL_DERIVED`, e `derived.<chave>.enabled` no YAML. Sem arquivo YAML extra.
- **Nova edição PNS:** entrada em `configs/datasets.yaml` (`year`, `columns`, `download`). Marque `enabled: true` ou use `--dataset`.

## 3. Configuração YAML

Um arquivo por módulo. `enabled: false` **pula** a etapa (não é flag cosmética).

| Arquivo | Papel |
|---------|--------|
| `configs/datasets.yaml` | Anos PNS, paths, mapa de colunas, download |
| `configs/preprocessing.yaml` | Estágios, pesos, derivadas, features de classificação |
| `configs/clustering.yaml` | `reducer` (PCA), `estimator` (K-Means), `selection_method: fixed \| elbow` |
| `configs/models.yaml` | Classificadores, `param_grid` inline, gridsearch |
| `configs/experiment.yaml` | Orquestração: `configuration_files`, dataset, seed, estágios |
| `configs/experiments/legacy.yaml` | Overlay (`extends`) só com diffs do TCC |

Mantidos como dados/UI (não são experimento): `configs/data/dictionary.yaml`, `configs/data/survey_metadata.yaml`, `configs/dashboard/run_overrides.yaml`.

O loader faz: YAML → `extends` (deep merge) → merge dos quatro módulos → `stages` enabled → lista. `--dataset` e `--models` nos scripts filtram sem editar o arquivo.

### Pré-processamento: essenciais vs melhorias

| Estágio | Default | Nota |
|---------|---------|------|
| `select_columns` | on | Mapa bruto → nome interno |
| `adult_filter` | on | `min_age: 18` |
| `derive_features` | on | Liga o registry `derived.*` |
| `one_hot_and_numeric` | on | Demografia e limita-atividades |
| `impute_missing` | on | Binário 0 / categórico -1 |
| `merge_internacao_into_servico` | on | Target composto (classificação) |
| `impute_age_by_target` | off | Melhoria / vazamento legado |
| `nearmiss` | off | Antes do split no legado |
| `target_encoding_before_split` | off | Vazamento legado |

`weights.legacy_formula: false` usa `V00291`; `true` usa a fórmula R do TCC.

### Modelos

A matriz da run = modelos com `enabled: true`. Gridsearch respeita `defaults.gridsearch.enabled` e pode ser desligado por modelo. Filtro temporário: `--models xgboost,random_forest`.

## 4. Scripts

| Script | Comportamento |
|--------|----------------|
| `scripts/setup_env.sh` | `.venv` + `pip install -e ".[dev,dashboard]"` (`--acquisition` opcional) |
| `scripts/run_download.sh` | Download explícito via PNSIBGE/R (requer `--acquisition`) |
| `scripts/run_preprocess.sh` | Download se `if_missing` + pré-processamento |
| `scripts/run_clustering.sh` | Clustering; pré-processa na mesma run |
| `scripts/run_classification.sh` | Classificação; `--from-run` reutiliza artefato |
| `scripts/run_experiment.sh` | Pipeline conforme `experiment.yaml` |
| `scripts/run_dashboard.sh` | Streamlit |

Flags comuns: `--config`, `--dataset`, `--models`, `--name`, `--no-download`.

Instalação padrão **não** inclui `[acquisition]`. R só é necessário para baixar a PNS via PNSIBGE.

### Primeira execução (resumo)

1. `git clone` + `./scripts/setup_env.sh --acquisition`
2. No R: `install.packages("PNSIBGE")`
3. `./scripts/run_download.sh` (ou deixar `run_experiment.sh` baixar se faltar parquet)
4. `./scripts/run_experiment.sh --name "..."` → `./scripts/run_dashboard.sh`

Sem parquet, o showcase em `artifacts/showcase/` permite explorar o dashboard sem R.

## 5. Dados

| Pasta | Conteúdo | Versionado |
|-------|----------|------------|
| `data/raw/` | Parquet PNS (`pns_2019.parquet`) | Não |
| `data/interim/` | Fingerprints | Não |
| `data/processed/` | Último pré-processamento | Não |

Download é ação explícita (`ensure_data_available` / script de preprocess). Backend 2019: PNSIBGE via R. Valida parquet, colunas mínimas e SHA-256.

Peso amostral: `survey_weight` (desenho) ≠ `ml_sample_weight` (normalizado para o fit).

## 6. Agrupamento e classificação

- 15 comorbidades → PCA (8 componentes, opcional) → K-Means (`k` fixo ou `selection_method: elbow`).
- Métricas: silhouette, WCSS, Davies-Bouldin, Calinski-Harabasz.
- Classificadores: Random Forest, XGBoost, MLP (SVC no YAML, `enabled: false`).
- CV: `legacy_cv: true` reproduz o notebook; `false` usa CV aninhada.
- Métrica principal no dashboard: `evaluation.primary_metric` (ex. `f1_positive`).

GPU é opcional (`classification.backend: gpu` / `clustering.backend: cuml`, extra `[gpu]`, CUDA, Python 3.12). Métricas cuML não são comparáveis ao baseline sklearn do TCC. XGBoost pode usar `device: cuda` mesmo no overlay legado.

## 7. Artefatos e dashboard

```
artifacts/runs/<run_id>/
├── manifest.json
├── summary.json
├── config.resolved.yaml
├── data_fingerprint.json
├── reproduction_report.json
├── logs/run.log
├── metrics/
├── tables/
└── predictions/
```

O dashboard lista runs `completed` em `artifacts/runs/` e o showcase em `artifacts/showcase/`. Overrides de exibição: `configs/dashboard/run_overrides.yaml`. Inventário de gráficos: [dashboard/chart_inventory.md](dashboard/chart_inventory.md).

Quando `baseline.enabled: true`, `reproduction_report.json` compara métricas com `baseline.tolerances` (inline no overlay legado).

## 8. Testes

```bash
source .venv/bin/activate
ruff check src tests
pytest tests/ -m "not gpu" -q
```

Marcador `regression`: exige parquet PNS real. Marcador `gpu`: extra `[gpu]`.

## 9. Decisões

- Python principal; R isolado em `[acquisition]`
- Artefatos em filesystem (sem banco)
- Scripts como interface; Typer permanece como implementação
- Pacote permanece em `src/pns_health_research/` (não migrar para `modules/`)
- `survey_weight` ≠ `ml_sample_weight`
