# Monografia TCC — Santos (2025)

**PDF:** [../../research/monografia.pdf](../../research/monografia.pdf)

## Identificação

| Campo | Valor |
|-------|-------|
| Título | Desenvolvimento de um modelo preditivo para identificação de uso dos serviços de saúde com base nas características de brasileiros |
| Autor | Gabriel Felipe Souza Santos |
| Instituição | UFOP — Sistemas de Informação |
| Orientadora | Helen de Cássia Sousa da Costa Lima |
| Coorientadora | Érica de Matos Reis Ferreira |
| Aprovação | 10/04/2025 |

## Problema de pesquisa

> Como desenvolver um modelo preditivo eficiente para identificar padrões de uso dos serviços de saúde com base nas características demográficas, socioeconômicas e de saúde da população brasileira?

## Objetivos

**Geral:** desenvolver e validar modelo preditivo capaz de identificar indivíduos ou grupos propensos a demandar serviços de saúde.

**Específicos:**
- Analisar padrões nos dados públicos de saúde (PNS 2019)
- Construir modelo com ML e mineração de dados
- Avaliar eficácia (métricas de predição e impacto na alocação de recursos)

## Contribuições declaradas

1. **Segmentação** — 6 perfis via PCA + K-Means sobre comorbidades
2. **Classificação** — RF, SVM, MLP, XGBoost; melhor: XGBoost (acurácia 78,5%, F1 78,4%)

## Metodologia (resumo)

| Etapa | Ferramenta | Detalhe |
|-------|------------|---------|
| Coleta | R + PNSIBGE | PNS 2019 |
| Pré-processamento | R | Filtros, one-hot, derivadas, peso amostral, target encoding, NearMiss |
| Agrupamento | Python (sklearn) | 15 comorbidades → PCA 8 → K-Means k=6 |
| Classificação | Python | Grid search + CV; bases binária e categórica |
| Avaliação | Métricas clássicas | Acurácia, precisão, recall, F1, matriz de confusão |

## Números oficiais (monografia)

| Dimensão | Valor |
|----------|-------|
| Linhas pós-filtro | 87.678 |
| Colunas binárias | 161 |
| Colunas categóricas | 61 |
| Silhouette (PCA=8, k=6) | ~0,5139 |
| WCSS | ~37.821,75 |
| Random Forest — acurácia | 0,778 |
| MLP — F1 (macro) | 0,779 |
| XGBoost — F1 (macro) | 0,784 |
| SVC — F1 (macro) | 0,718 |

## Variáveis centrais

- **Target:** uso de serviços de saúde (`J014`), composto com internação (`J037`)
- **Comorbidades:** 15 condições + flags de limitação de atividades
- **Demografia:** idade (faixas), sexo, raça, zona, renda, escolaridade
- **Comportamento:** álcool, tabaco, exercício, IMC derivado

## Técnicas com implicação metodológica

- **Target encoding antes do split** — vazamento documentado no software
- **NearMiss-1** — balanceamento antes do split (legado)
- **Imputação de idade pelo target** — estágio legado
- **Peso amostral** — fórmula R específica (não idêntica a `V00291` direto)
- **PCA sem StandardScaler** — no clustering do TCC

## Uso no lab

- Baseline numérico em `configs/legacy/tcc_overlay.yaml` → `baseline.tolerances`
- Showcase versionado: `artifacts/showcase/pesquisa-original-legacy/`
- Reprodução: `overall_status: FAIL` em classificação (métricas acima do TCC); PASS em pré-processamento e silhouette

## O que não repetimos literalmente

- Notebooks Jupyter legados (removidos); pipeline em `src/pns_health_research/`
- SVC desabilitado por padrão no `models.yaml` atual (`enabled: false`)
