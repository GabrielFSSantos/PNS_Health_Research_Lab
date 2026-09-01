# Adding Algorithms

## Adicionar coluna raw (PNS)

1. Edite `configs/data/pns2019_columns.yaml` — mapa `codigo_PNS: label_pt`
2. Referencie em `preprocessing.columns_config` no YAML do experimento
3. Se a coluna entrar em features binárias/categóricas, estenda `preprocessing/features_tcc.py` (ou blocos YAML em `configs/features/`)

## Adicionar bloco de feature

Features TCC legadas estão em `features_tcc.py` (espelho do notebook). Para novos blocos declarativos, use `configs/features/legacy_tcc_binary.yaml` e `legacy_tcc_categorical.yaml` como referência documental; a engenharia efetiva permanece em Python até migração completa.

## Classificação — colunas de entrada

Edite `configs/features/classification_default.yaml` (`target`, `exclude_columns`, `target_encode_columns`) ou sobrescreva em `classification.features_config` / `exclude_columns` no experimento.

## Classifier

1. Add factory in `src/pns_health_research/classification/registry.py`:

```python
@register_classifier("lightgbm")
def create_lightgbm(params):
    ...
```

2. Add `configs/classifiers/lightgbm.yaml` with `param_grid` and optional `n_jobs`
3. List model in experiment YAML `classification.models`

For GPU (research_v2 only), register in `classification/gpu/registry.py` and add factory module.

## Parallel classification

Set `classification.parallel: true` and `max_workers` in experiment YAML. Per-model `n_jobs` in classifier YAML controls CPU contention via `effective_n_jobs()`.

## Resume classification

```bash
pns-research classification --config configs/experiments/legacy_reproduction.yaml --from-run <run_id>
```

Requires `tables/clustered_categorical.parquet` in the source run.

## Clusterer

- **sklearn** (padrão): `clustering.backend: sklearn` — reproduz baseline TCC
- **cuML** (opcional): `clustering.backend: cuml` — métricas não comparáveis ao TCC; use perfil `research_v2` com `baseline.enabled: false`

Extend `clustering/kmeans.py` pattern or add `clustering/registry.py` (future).

## Reducer

Extend `dimensionality/pca.py` or add MCA module under optional `[mca]` extra.
