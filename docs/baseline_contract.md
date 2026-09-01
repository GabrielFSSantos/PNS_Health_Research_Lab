# Baseline Contract

Valores de referência reportados pela monografia e notebooks históricos.
Usados em `reproduction_report` e testes de regressão — **não são hardcoded como resultados da pipeline**.

Tolerâncias: [`configs/baseline/tolerances.yaml`](../configs/baseline/tolerances.yaml).

## Pré-processamento

| Métrica | Valor esperado | Fonte |
|---------|----------------|-------|
| Registros finais | 87.678 | Tabela 2 monografia |
| Colunas base binária | 161 | Tabela 2 |
| Colunas base categórica | 61 | Tabela 2 |
| Target=1 (J014, antes OR/NearMiss) | 18.267 | Tabela 3 |
| Target=0 (antes OR/NearMiss) | 69.411 | Tabela 3 |
| Target=1 (após OR Internação + NearMiss) | 22.254 | Tabela 4 |
| Target=0 (após NearMiss) | 22.254 | Tabela 4 |

## Agrupamento

| Métrica | Valor esperado | Notas |
|---------|----------------|-------|
| PCA n_components | 8 | Código + Figura 5 |
| K-Means n_clusters | 6 | TCC |
| Silhouette (global) | 0.5138811018387510 | Análises Posteriores (8-6) |
| WCSS (inertia) | 37821.75 | Monografia §4.3.7 |
| Erro médio por ponto | 0.4314 | WCSS / n |
| Cluster 0 count | 7.341 | Tabela 5 — alinhar labels antes de comparar |
| Cluster 1 count | 40.569 | |
| Cluster 2 count | 11.585 | |
| Cluster 3 count | 8.523 | |
| Cluster 4 count | 8.415 | |
| Cluster 5 count | 11.245 | |

Parâmetros K-Means legado: `init=k-means++`, `n_init=50`, `max_iter=1000`, `random_state=42`.

## Classificação (base Total, após NearMiss)

Métricas macro médias sobre 5 folds (monografia cap. 5):

| Modelo | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| Random Forest | 0.778 | 0.781 | 0.778 | 0.777 |
| SVC | 0.721 | 0.731 | 0.721 | 0.718 |
| MLP | 0.779 | 0.782 | 0.779 | 0.779 |
| XGBoost | 0.785 | 0.790 | 0.785 | 0.784 |

## Não incluídos no contrato

- Tempos de execução (dependem de hardware).
- Importâncias exatas por variável (variam por seed/fold).
- Labels de cluster sem alinhamento (Hungarian).

## Comportamento em falha

Se `actual != expected` dentro da tolerância, o relatório marca `FAIL` e registra diferença absoluta/relativa. **Expected não é alterado silenciosamente.**
