"""Textos da interface em português (Brasil)."""

NAV_DATASET = "Conjunto de dados"
NAV_CLUSTERING = "Agrupamento"
NAV_CLUSTERING_CMP = "Comparar agrupamento"
NAV_CLASSIFICATION = "Classificação"
NAV_CLASSIFICATION_CMP = "Comparar classificação"
NAV_EVOLUTION = "Evolução"
NAV_RESEARCH = "Pesquisa"

MODEL_NAMES: dict[str, str] = {
    "random_forest": "Floresta Aleatória (Random Forest)",
    "svc": "SVM (Support Vector Machine)",
    "mlp": "Rede Neural (MLP)",
    "xgboost": "XGBoost",
}


def model_label(name: str) -> str:
    return MODEL_NAMES.get(name, name.replace("_", " ").title())

TAB_OVERVIEW = "Visão geral"
TAB_PANORAMA = "Panorama"
TAB_QUALITY = "Qualidade"
TAB_METHOD = "Como agrupamos"
TAB_USAGE = "Uso de serviços"
TAB_VARIABLES_BY_CLUSTER = "Variáveis por cluster"
TAB_PERFORMANCE = "Desempenho"
TAB_DIAGNOSIS = "Diagnóstico"
TAB_TRAINING = "Treinamento"
TAB_VARIABLES = "Variáveis"
TAB_SURVEY = "Variáveis da pesquisa"
TAB_POST_PREPROCESS = "Após pré-processamento"
TAB_DICTIONARY = "Dicionário técnico"
TAB_PREPROCESSING = "Pré-processamento"
TAB_METRICS = "Métricas"
TAB_PCA = "Redução de dimensionalidade (PCA)"
TAB_ELBOW = "Método do cotovelo"
TAB_PROFILES = "Perfis dos clusters"
TAB_SILHOUETTE = "Análise de silhueta"
TAB_HEATMAP = "Mapa de calor"
TAB_CURVES = "Curvas de desempenho"
TAB_GRIDSEARCH = "Busca em grade (Grid Search)"
TAB_CM = "Matriz de confusão"
TAB_IMPORTANCE = "Importância das variáveis"
TAB_PREDICTIONS = "Resumo das predições"

TECHNICAL_DETAILS = "Detalhes técnicos"
TECHNICAL_DETAILS_DEV = "JSON bruto da execução (desenvolvedores)"
HELP_EXPANDER = "O que isto mostra?"
CHART_HELP_EXPANDER = HELP_EXPANDER

LABEL_BUBBLE_SPLIT = "Divisão vertical (eixo Y)"
LABEL_BUBBLE_CATEGORY = "Categorias (eixo X)"
LABEL_PRIMARY_VARIABLE = "Variável principal"
LABEL_CROSS_VARIABLE = "Cruzamento (opcional)"
MSG_CLUSTERED_BINARY_UNAVAILABLE = (
    "Base binária com cluster indisponível — exibindo bolhas pré-agregadas. "
    "Reexecute o experimento para seletores dinâmicos."
)
MSG_PROFILES_RATE_FALLBACK = (
    "Taxa estimada a partir dos perfis médios do cluster (comorbidades do agrupamento)."
)
MSG_PROFILES_ONLY_VARIABLES = (
    "Seletores limitados — apenas variáveis presentes nos perfis de comorbidade."
)
MSG_CROSS_UNAVAILABLE = "Cruzamento indisponível sem base binária com cluster nesta execução."
