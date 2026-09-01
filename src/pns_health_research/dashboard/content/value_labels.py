"""Rótulos legíveis para valores codificados nas visualizações."""

from __future__ import annotations

# Variáveis binárias comuns na base TCC/PNS
BINARY_LABELS: dict[str, dict[str, str]] = {
    "Serviço de Saúde": {"0": "Não utilizou", "1": "Utilizou"},
    "Internação": {"0": "Não", "1": "Sim"},
    "Plano de Saúde": {"0": "Não possui", "1": "Possui"},
    "Homem": {"1": "Homem"},
    "Mulher": {"1": "Mulher"},
}

# Ordem lógica para categorias de saúde (bolhas / one-hot)
HEALTH_CATEGORY_ORDER = [
    "Muito Ruim",
    "Ruim",
    "Regular",
    "Bom",
    "Muito Bom",
]

# Colunas dummy one-hot: valor 1 significa presença da categoria
ONE_HOT_COLUMNS = {
    "Homem": "Homem",
    "Mulher": "Mulher",
    "Saúde Regular": "Saúde regular",
    "Saúde Muito Ruim": "Saúde muito ruim",
    "Saúde Bom": "Saúde boa",
    "Saúde Muito Bom": "Saúde muito boa",
    "Saúde Ruim": "Saúde ruim",
}


def label_for_value(column_name: str, raw_value: str | int | float) -> str:
    """Converte valor codificado em rótulo legível."""
    key = str(raw_value).strip()
    if column_name in BINARY_LABELS and key in BINARY_LABELS[column_name]:
        return BINARY_LABELS[column_name][key]
    if column_name in ONE_HOT_COLUMNS and key == "1":
        return ONE_HOT_COLUMNS[column_name]
    if column_name.startswith("Idade ") and key == "1":
        return column_name.replace("Idade ", "")
    if column_name.startswith("Saúde ") and key == "1":
        return column_name
    return key


def friendly_column_name(column_name: str) -> str:
    return ONE_HOT_COLUMNS.get(column_name, column_name)


def health_category_label(raw: str) -> str:
    text = raw if raw.startswith("Saúde") else f"Saúde {raw}"
    return text


def sort_health_categories(categories: list[str]) -> list[str]:
    order = {c: i for i, c in enumerate(HEALTH_CATEGORY_ORDER)}

    def sort_key(cat: str) -> tuple[int, str]:
        base = cat.replace("Saúde ", "").strip()
        return (order.get(base, order.get(cat, 99)), cat)

    return sorted(categories, key=sort_key)


# Ordem do notebook DCC (Muito Bom → Muito Ruim, esquerda para direita)
HEALTH_CATEGORY_ORDER_BUBBLE = [
    "Muito Bom",
    "Bom",
    "Regular",
    "Ruim",
    "Muito Ruim",
]


def sort_health_categories_bubble(categories: list[str]) -> list[str]:
    order = {c: i for i, c in enumerate(HEALTH_CATEGORY_ORDER_BUBBLE)}

    def sort_key(cat: str) -> tuple[int, str]:
        base = cat.replace("Saúde ", "").strip()
        return (order.get(base, order.get(cat, 99)), cat)

    return sorted(categories, key=sort_key)


FUNNEL_STAGE_LABELS: dict[str, str] = {
    "raw": "Dados brutos (PNS 2019)",
    "columns_selected": "Variáveis selecionadas",
    "adult_filter": "Filtro de elegibilidade",
    "gestational_imputation": "Imputação gestacional / idade",
    "weights_attached": "Peso amostral",
    "binary_encoding": "Codificação binária (one-hot)",
    "missing_imputation": "Imputação de ausentes",
    "processed": "Base processada",
}

FUNNEL_STAGE_CAPTIONS: dict[str, str] = {
    "raw": "Microdados completos antes de filtros",
    "columns_selected": "Colunas do TCC renomeadas a partir dos códigos IBGE",
    "adult_filter": "Remove <18 anos, sem peso/altura ou sem resposta sobre serviços de saúde",
    "gestational_imputation": "Consolida comorbidades gestacionais; pode imputar idade (legacy)",
    "weights_attached": "Calcula peso para análises ponderadas",
    "binary_encoding": "Sexo, idade, IMC, comorbidades e demais atributos viram colunas binárias",
    "missing_imputation": "NaN restantes viram 0 (binário) ou -1 (categórico) — sem remover linhas",
    "processed": "Matriz binária (agrupamento) + categórica (classificação)",
}


def funnel_stage_label(stage_id: str) -> str:
    return FUNNEL_STAGE_LABELS.get(stage_id, stage_id.replace("_", " ").title())


def funnel_stage_caption(stage_id: str) -> str:
    return FUNNEL_STAGE_CAPTIONS.get(stage_id, "")
