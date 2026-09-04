# Inventário de gráficos — Dashboard Streamlit

Mapeamento notebook → artifact → função Python → status.

| Notebook | Gráfico | Artifact | Função (`components/charts.py`) | Status |
|----------|---------|----------|----------------------------------|--------|
| Agrupamento | Elbow WCSS×k | `tables/elbow_results.parquet` | `elbow_plot` | Corrigido (pt-BR, k ótimo) |
| Agrupamento | Variância PCA + acumulada | `tables/pca_explained_variance.parquet` | `pca_variance_plot` | Corrigido (eixo secundário) |
| Agrupamento | Scatter PCA | `tables/pca_scatter_sample.parquet` | `pca_scatter_plot` | Corrigido |
| Agrupamento | Loadings PCA | `tables/pca_loadings.parquet` | `pca_loadings_heatmap` | Corrigido |
| Agrupamento | Silhouette dual | `silhouette_samples` + `pca_scatter_sample` | `silhouette_dual_plot` | Corrigido (subplot duplo) |
| Agrupamento | Radar perfis | `tables/cluster_profiles.parquet` | `cluster_radar_plot` | Corrigido (todas comorbidades) |
| Agrupamento | Heatmap perfis | `tables/cluster_profiles.parquet` | `profile_heatmap` | Corrigido (formato largo) |
| Agrupamento | Bolhas configuráveis | `tables/clustered_binary.parquet` + seletores | `bubble_cross_plot` | Implementado (default saúde×serviço) |
| Agrupamento | Taxa variável/cluster | `tables/clustered_binary.parquet` ou `cluster_profiles` | `variable_rate_bar` | Implementado (seletor) |
| Agrupamento | Cruzamento cluster×categoria | `tables/clustered_binary.parquet` | `cluster_cross_heatmap` | Implementado (opcional) |
| Agrupamento | Bolhas legado | `tables/bubble_health_service.parquet` | `bubble_health_service_plot` | Fallback |
| Agrupamento | Uso serviços/cluster (legado) | `tables/cross_analysis.parquet` | `cross_analysis_bar` | Compatibilidade |
| Coleta/Preproc | Funil | `metrics/preprocess_funnel.json` | `preprocess_funnel` | 8 estágios + cards explicativos |
| Coleta/Preproc | Variáveis da pesquisa | `metrics/survey_snapshot.json` | `survey_explorer` | Pré-one-hot |
| Coleta/Preproc | Distribuições (pós-proc) | `metrics/dataset_distributions.json` | `distribution_plot` | Implementado |
| Classificação | ROC | `metrics/roc_curves/{model}.json` | `roc_plot` | Corrigido (eixos pt-BR) |
| Classificação | PR | `metrics/pr_curves/{model}.json` | `pr_plot` | Implementado |
| Classificação | Calibração | `metrics/calibration/{model}.json` | `calibration_plot` | Implementado |
| Classificação | CM | `metrics/confusion_matrices.json` | `confusion_heatmap` | Corrigido (Negativo/Positivo) |
| Classificação | Feature importance | `tables/feature_importance/*.parquet` | `importance_bar` | Implementado |
| Classificação | GridSearch | `tables/grid_search/*.parquet` | `gridsearch_scatter` | Implementado |
| Compare | Barras agrupadas | compare services | `grouped_metric_bar` | Implementado |
| Evolução | F1+ por execução | catálogo curado | `metric_bar` | Corrigido (nomes amigáveis) |

## Componentes de UX

| Componente | Arquivo | Uso |
|------------|---------|-----|
| Textos pt-BR | `content/pt_br.py` | Navegação, abas, modelos |
| Ajuda por gráfico | `content/chart_help.py` | Expander "O que isto mostra?" (gráficos e seções) |
| Catálogo de métricas | `metrics/catalog.py` | Expander por card em `metric_card` / `metric_row` |
| Bloco padrão | `components/chart_block.py` | Título + ajuda + render |
| Cards de métrica | `components/metric_card.py` | `metric_card` + `metric_row` (expander por card) |
| Seção padronizada | `components/section_block.py` | Título + ajuda de bloco + conteúdo |

## Paleta

Definida em `dashboard/config.py`: `CLUSTER_COLORS`, `MODEL_COLORS`.

Layout padrão: altura 560–620 px, `use_container_width=True`.

## Páginas

| Página | Abas / gráficos principais |
|--------|---------------------------|
| Conjunto de dados | Variáveis da pesquisa · Após pré-processamento · Dicionário técnico |
| Agrupamento | Panorama · Qualidade · Como agrupamos · Perfis · Variáveis por cluster |
| Classificação | Panorama · Desempenho · Diagnóstico · Treinamento · Variáveis |
| Comparar agrupamento / classificação | `grouped_metric_bar` |
| Evolução | `metric_bar` |

## Runs visíveis

1. **Bases modulares** completas em `data/bases/<nome>/` (preprocessing + clustering + classification)
2. **Showcases** em `artifacts/showcase/` (ex.: `pesquisa-original-legacy`)
3. **Runs legadas** em `artifacts/runs/` (se existirem)

Overrides de exibição: `configs/dashboard/run_overrides.yaml` (camadas 1–3).
