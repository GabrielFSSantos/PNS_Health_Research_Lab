"""Catálogo de métricas PNS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInfo:
    key: str
    label: str
    description: str
    interpretation_guide: str
    higher_is_better: bool
    bands: tuple[tuple[float, float, str], ...] = ()


METRICS: dict[str, MetricInfo] = {
    "silhouette": MetricInfo(
        key="silhouette",
        label="Silhueta",
        description=(
            "**O que é:** coeficiente de silhueta (−1 a +1) que mede, para cada indivíduo, "
            "o quão bem ele se encaixa no próprio cluster em relação aos clusters vizinhos.\n\n"
            "**Para que serve nesta pesquisa:** validar se os perfis de multimorbidade formados "
            "pelo K-Means são coesos internamente e separados entre si — base para interpretar "
            "os clusters antes de cruzar com uso de serviços de saúde."
        ),
        interpretation_guide=(
            "Valores próximos de **+1** indicam clusters densos e bem separados; próximos de "
            "**0** sugerem sobreposição; **negativos** indicam possível atribuição inadequada."
        ),
        higher_is_better=True,
        bands=((-1, 0.25, "fraco"), (0.25, 0.5, "moderado"), (0.5, 1.01, "bom")),
    ),
    "davies_bouldin": MetricInfo(
        key="davies_bouldin",
        label="Davies-Bouldin",
        description=(
            "**O que é:** índice que relaciona a dispersão média **dentro** de cada cluster "
            "com a distância entre os centróides dos clusters.\n\n"
            "**Para que serve nesta pesquisa:** complementar a silhueta — quanto menor o valor, "
            "mais compactos e distintos tendem a ser os perfis de comorbidade encontrados."
        ),
        interpretation_guide="Quanto **menor**, melhor a separação entre clusters. Não tem escala fixa universal.",
        higher_is_better=False,
        bands=((0, 1.0, "bom"), (1.0, 2.0, "moderado"), (2.0, 100.0, "fraco")),
    ),
    "calinski_harabasz": MetricInfo(
        key="calinski_harabasz",
        label="Calinski-Harabasz",
        description=(
            "**O que é:** razão entre a variância **entre** clusters e a variância **dentro** "
            "dos clusters (também chamado índice de variância).\n\n"
            "**Para que serve nesta pesquisa:** indicar se a partição em k grupos explica bem "
            "a estrutura das comorbidades após a redução de dimensionalidade (PCA)."
        ),
        interpretation_guide="Quanto **maior**, melhor a estrutura encontrada. Compare entre execuções com o mesmo k.",
        higher_is_better=True,
        bands=(),
    ),
    "f1_positive": MetricInfo(
        key="f1_positive",
        label="F1+ (classe positiva)",
        description=(
            "**O que é:** F1-score da classe positiva — média harmônica entre **precisão** e "
            "**recall**: F1 = 2 × (precisão × recall) / (precisão + recall).\n\n"
            "**Para que serve nesta pesquisa:** resumir o desempenho do classificador na tarefa "
            "de identificar **uso de serviços de saúde** (classe de interesse), equilibrando "
            "acertos entre quem realmente usou e quem foi corretamente excluído."
        ),
        interpretation_guide=(
            "Penaliza modelos que só acertam uma das dimensões (muitos falsos positivos ou "
            "muitos falsos negativos)."
        ),
        higher_is_better=True,
        bands=((0, 0.5, "fraco"), (0.5, 0.7, "moderado"), (0.7, 1.01, "bom")),
    ),
    "recall_positive": MetricInfo(
        key="recall_positive",
        label="Recall+ (sensibilidade)",
        description=(
            "**O que é:** recall (sensibilidade) da classe positiva — proporção de indivíduos "
            "que **realmente usaram** serviços de saúde e foram corretamente classificados como positivos: "
            "TP / (TP + FN).\n\n"
            "**Para que serve nesta pesquisa:** medir a **cobertura** do modelo sobre quem de fato "
            "utilizou serviços; alto recall significa capturar a maior parte dos usuários reais."
        ),
        interpretation_guide="Útil quando o custo de **não identificar** um usuário de serviços é alto.",
        higher_is_better=True,
        bands=((0, 0.5, "baixo"), (0.5, 0.75, "moderado"), (0.75, 1.01, "alto")),
    ),
    "precision_positive": MetricInfo(
        key="precision_positive",
        label="Precisão+",
        description=(
            "**O que é:** precisão da classe positiva — entre todos classificados como usuários "
            "de serviços, quantos realmente são: TP / (TP + FP).\n\n"
            "**Para que serve nesta pesquisa:** avaliar a **confiabilidade** das predições positivas; "
            "alta precisão reduz falsos alarmes."
        ),
        interpretation_guide="Complementa o recall: um modelo pode ter alto recall classificando quase tudo como positivo.",
        higher_is_better=True,
        bands=((0, 0.5, "baixa"), (0.5, 0.75, "moderada"), (0.75, 1.01, "alta")),
    ),
    "roc_auc": MetricInfo(
        key="roc_auc",
        label="ROC-AUC",
        description=(
            "**O que é:** área sob a curva ROC (Receiver Operating Characteristic), que traça "
            "**sensibilidade (TPR)** no eixo Y versus **taxa de falso positivo (FPR)** no eixo X "
            "ao variar o limiar de decisão.\n\n"
            "**Para que serve nesta pesquisa:** medir a capacidade discriminativa global do modelo "
            "para separar quem usa de quem não usa serviços de saúde, independentemente de um "
            "limiar fixo de 0,5."
        ),
        interpretation_guide="**0,5** ≈ classificador aleatório; **próximo de 1** = melhor discriminação.",
        higher_is_better=True,
        bands=((0, 0.6, "fraco"), (0.6, 0.8, "moderado"), (0.8, 1.01, "bom")),
    ),
    "pr_auc": MetricInfo(
        key="pr_auc",
        label="PR-AUC",
        description=(
            "**O que é:** área sob a curva **precisão-recall**, com recall no eixo X e precisão "
            "no eixo Y.\n\n"
            "**Para que serve nesta pesquisa:** avaliar desempenho quando a classe positiva (uso "
            "de serviços) é **desbalanceada** — a curva PR é mais informativa que a ROC nesse cenário."
        ),
        interpretation_guide="Valores maiores indicam melhor equilíbrio entre cobertura e confiabilidade na classe rara.",
        higher_is_better=True,
        bands=((0, 0.4, "fraco"), (0.4, 0.6, "moderado"), (0.6, 1.01, "bom")),
    ),
    "n_samples": MetricInfo(
        key="n_samples",
        label="Amostras avaliadas",
        description=(
            "**O que é:** número total de observações usadas na avaliação do modelo "
            "(conjunto de teste ou validação cruzada agregada).\n\n"
            "**Para que serve nesta pesquisa:** contextualizar a robustez estatística das métricas "
            "— amostras maiores tornam F1, recall e AUC mais estáveis."
        ),
        interpretation_guide="Compare entre execuções para verificar se filtros de elegibilidade alteraram o tamanho da amostra.",
        higher_is_better=True,
        bands=(),
    ),
    "n_positive": MetricInfo(
        key="n_positive",
        label="Predições positivas",
        description=(
            "**O que é:** contagem de indivíduos classificados como **usuários de serviços de saúde** "
            "pelo modelo.\n\n"
            "**Para que serve nesta pesquisa:** verificar se o classificador não está enviesado "
            "para prever sempre positivo ou sempre negativo."
        ),
        interpretation_guide="Compare com a taxa de predição positiva e com a prevalência real na PNS.",
        higher_is_better=True,
        bands=(),
    ),
    "positive_rate": MetricInfo(
        key="positive_rate",
        label="Taxa de predição positiva",
        description=(
            "**O que é:** proporção de predições positivas em relação ao total avaliado "
            "(predições positivas ÷ amostras).\n\n"
            "**Para que serve nesta pesquisa:** indicar o **viés de decisão** do modelo — "
            "taxas muito altas ou baixas podem sinalizar limiar inadequado ou desbalanceamento."
        ),
        interpretation_guide="Confronte com a prevalência observada de uso de serviços na amostra PNS filtrada.",
        higher_is_better=True,
        bands=(),
    ),
}


def get_metric(key: str) -> MetricInfo | None:
    return METRICS.get(key)


def interpret_value(key: str, value: float | None) -> str:
    info = get_metric(key)
    if info is None or value is None:
        return "Sem interpretação disponível."
    if not info.bands:
        return ""
    for low, high, label in info.bands:
        if low <= value < high:
            return f"**Faixa atual:** {label} ({low:.2f}–{high:.2f})"
    return "Fora das faixas definidas."
