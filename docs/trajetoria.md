# Trajetória da pesquisa

Três camadas distintas — **não misturar** números de uma com os de outra.

## 1. Monografia (TCC)

Trabalho original (UFOP): modelo preditivo de uso de serviços de saúde com PNS 2019. Pré-processamento no R, clustering e classificação no Python (notebook).

| Dimensão | Valor na monografia |
|----------|---------------------|
| Linhas pós-filtro | 87.678 |
| Colunas binárias | 161 |
| Colunas categóricas | 61 |
| Silhouette (8 PCA, k=6) | ~0,5139 |
| RF accuracy | 0,778 |
| MLP F1 (macro) | 0,779 |
| XGBoost F1 (macro) | 0,784 |

A monografia original (UFOP) é a referência bibliográfica do trabalho; o showcase versionado no repositório espelha a reprodução computacional.

## 2. Reprodução no software (`legacy.yaml`)

O overlay `configs/experiments/legacy.yaml` liga o que o TCC fazia e o default atual desliga:

- peso pela fórmula R (`weights.legacy_formula`)
- imputação de idade pelo target
- NearMiss e target encoding **antes** do split (vazamento documentado)
- CV do notebook (`legacy_cv: true`)
- k=6, PCA sem scaler
- tolerâncias inline em `baseline.tolerances`

A run commitada no Git está em `artifacts/showcase/pesquisa-original-legacy/`. Snapshots JSON em `artifacts/baselines/legacy_reproduction/`.

| Dimensão | Software (reprodução) |
|----------|------------------------|
| Linhas / colunas | 87.678 / 161 / 61 |
| Silhouette | ~0,5127 (tolerância absoluta 0,005) |
| Classificação | pode ficar **acima** do TCC após features completas; contrato ±0,01 |

Métricas de classificação do software não substituem a tabela da monografia. O `reproduction_report.json` registra PASS/FAIL contra as tolerâncias.

## 3. Pesquisa evolutiva (`experiment.yaml`)

Default do laboratório:

- peso amostral PNS (`V00291`)
- sem NearMiss global, sem target encoding antes do split
- sem imputar idade pelo target
- CV aninhada
- PCA com StandardScaler
- modelos `enabled` em `configs/models.yaml` (RF, XGBoost, MLP)

Essas runs **não** devem ser comparadas numericamente ao TCC sem análise: mudam peso, leakage e validação.

O campo `experiment.profile` (`legacy_reproduction`, `custom`) é metadado para filtrar runs (`scripts/prune_test_runs.py --profile`), não altera o pipeline por si só.

## 4. Entrega da plataforma

- Contrato de runs v2 (`manifest.json`, lineage, `artifacts_status`)
- Dashboard Streamlit (visão geral, dataset, agrupamento, classificação, comparações, evolução, research)
- Showcase legado para clone sem reexecução
- Configs modulares (um YAML por módulo + overlay)

## 5. Como ler resultados

1. Escolha a camada (monografia, reprodução, ou run custom).
2. Confira `config.resolved.yaml` da run: estágios `enabled`, modelos, seed.
3. Use o dashboard para gráficos; os números oficiais da run estão em `metrics/*.json`.
4. Para o TCC, o showcase `pesquisa-original-legacy` é a referência visual versionada.
