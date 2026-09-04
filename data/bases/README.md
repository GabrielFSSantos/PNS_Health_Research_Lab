# Bases de dados (`data/bases/`)

Cada **base** é uma pasta nomeada com módulos independentes (pré-processamento, agrupamento, classificação). Mesmo nome → sobrescreve só o módulo reexecutado. Nome diferente → pasta nova.

## Layout

```
data/bases/<nome>/
├── registry.json           # última execução e contagem por módulo
├── base_manifest.json      # módulos presentes na base
├── preprocessing/          # entrada do agrupamento
│   ├── binary.parquet
│   ├── categorical.parquet
│   ├── manifest.json
│   └── ...
├── clustering/             # saída do agrupamento
│   ├── clustered_binary.parquet
│   ├── clustered_categorical.parquet
│   ├── manifest.json
│   └── metrics/ ...
└── classification/         # saída da classificação
    ├── manifest.json
    ├── metrics/classification.json
    ├── models/
    ├── predictions/
    └── tables/ ...
```

## Comandos mínimos

```bash
# Presets oficiais (recomendado)
./scripts/run_preprocess.sh --profile pns      # base rigorosa (PRE-*)
./scripts/run_preprocess.sh --profile legacy   # reprodução TCC

# Pipeline completo
./scripts/run_experiment.sh --profile pns --download
./scripts/run_experiment.sh --profile legacy
./scripts/run_experiment.sh --all-bases

# Listar bases materializadas
python -m pns_health_research data bases list
```

| Preset `--profile` | Base (`--name`) | Config (perfil unificado) | Metodologia |
|--------------------|-----------------|---------------------------|-------------|
| `pns` | `pns` | `configs/profiles/pns.yaml` | rigoroso (research_v2; GPU auto) |
| `legacy` | `legacy` | `configs/profiles/legacy.yaml` | TCC (CPU) |
| (omitido) | qualquer | `--config` override + `--name` | `custom` |

Customização: copie [`configs/overrides/example.yaml`](../../configs/overrides/example.yaml) e use `--config` com `--profile pns`.

## Regenerar após mudança de config

```bash
./scripts/run_experiment.sh --all-bases
```

Isso atualiza manifests (`entry_config_path`) e métricas em `registry.json`.

## Configuração

Mapa completo: [configs/README.md](../../configs/README.md).

| Módulo | Perfil | Pipeline interno |
|--------|--------|-------------------|
| Pré-processamento | `configs/profiles/pns.yaml` | `configs/_internal/pipelines/preprocessing/pns.yaml` |
| Agrupamento | idem | `configs/_internal/pipelines/clustering/pns.yaml` |
| Classificação | idem | `configs/shared/models.yaml` |

Substitua `pns` por `legacy` para a trilha TCC.

## Registry (`registry.json`)

Exemplo após preprocess + clustering + classification:

```json
{
  "base_name": "pns",
  "modules": {
    "preprocessing": {
      "profile_kind": "pns",
      "methodological_profile": "pns_preprocessing",
      "entry_config": "configs/profiles/pns.yaml#preprocessing"
    },
    "clustering": {
      "profile_kind": "pns",
      "source_base": "pns",
      "n_clusters": 6,
      "silhouette": 0.379
    },
    "classification": {
      "profile_kind": "pns",
      "methodological_profile": "research_v2",
      "primary_metric": "f1_positive"
    }
  }
}
```

Documentação: [docs/documentacao.md](../../docs/documentacao.md)
