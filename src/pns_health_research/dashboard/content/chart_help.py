"""Textos explicativos dos gráficos (adaptados dos notebooks legados)."""

CHART_HELP: dict[str, str] = {
    "elbow": """
**O que é / para que serve**

O **método do cotovelo (Elbow)** executa K-Means para vários valores de *k* (número de clusters)
e calcula a **soma dos quadrados intra-cluster (WCSS)** — medida de quão espalhados estão os
pontos dentro de cada grupo. Quanto menor o WCSS, mais compactos são os clusters.

**Eixos e variáveis**

- **Eixo X:** número de clusters (*k*)
- **Eixo Y:** WCSS (within-cluster sum of squares)
- **Linha vermelha:** *k* escolhido nesta execução

**Como interpretar**

Procure o ponto em que a queda do WCSS **desacelera** (formato de “cotovelo”): indica equilíbrio
entre poucos clusters muito heterogêneos e muitos clusters pouco interpretáveis. No notebook de
agrupamento, o declínio mais acentuado costuma ocorrer até *k* ≈ 4–6.
""",
    "pca_variance": """
**O que é / para que serve**

A **Análise de Componentes Principais (PCA)** reduz a dimensionalidade das comorbidades antes
do K-Means, preservando o máximo de variância possível em poucos eixos ortogonais (PCs).

**Eixos e variáveis**

- **Barras azuis:** variância explicada por cada componente principal (PC1, PC2, …)
- **Linha vermelha:** variância **acumulada** (% do total retido somando PCs)

**Como interpretar**

Some componentes até atingir ~**80–90%** da variância acumulada — no TCC, os 10 primeiros PCs
capturam a maior parte da estrutura. Mais PCs = mais informação, porém clusters mais difíceis
de visualizar.
""",
    "pca_scatter": """
**O que é / para que serve**

Projeção de cada indivíduo no espaço das **duas primeiras componentes principais**, colorido
pelo cluster atribuído pelo K-Means.

**Eixos e variáveis**

- **Eixo X:** PC1 (combinação linear das comorbidades com maior variância)
- **Eixo Y:** PC2
- **Cor:** cluster (0, 1, 2, …)
- **Marcador X:** centróide do cluster

**Como interpretar**

Clusters **visualmente separados** no plano PC1×PC2 tendem a ser mais interpretáveis. Sobreposição
indica perfis semelhantes ou necessidade de ajustar *k* ou número de componentes.
""",
    "pca_loadings": """
**O que é / para que serve**

Os **loadings** mostram quanto cada comorbidade contribui para formar cada componente principal —
ou seja, quais doenças “puxam” cada eixo da redução de dimensionalidade.

**Eixos e variáveis**

- **Linhas:** comorbidades (hipertensão, diabetes, etc.)
- **Colunas:** componentes principais (PC1, PC2, …)
- **Cor/intensidade:** magnitude e sinal da contribuição

**Como interpretar**

Valores absolutos **maiores** indicam variáveis que mais definem aquele PC. Sinais opostos
sugerem comorbidades que variam em direções contrárias na população.
""",
    "silhouette": """
**O que é / para que serve**

A **análise de silhueta** valida a qualidade do agrupamento: para cada pessoa, mede se está
mais próxima do próprio cluster ou de um cluster vizinho (coeficiente de −1 a +1).

**Eixos e variáveis**

- **Painel esquerdo:** distribuição dos coeficientes por cluster (largura = densidade de pontos)
- **Linha vermelha tracejada:** silhueta média global
- **Painel direito:** mesma amostra no espaço PCA (PC1 × PC2), colorida por cluster

**Como interpretar**

Silhueta média **≥ 0,5** sugere estrutura forte; **0,25–0,5** estrutura razoável; abaixo disso,
clusters podem se sobrepor. Barras largas e positivas por cluster indicam coesão interna.
""",
    "radar_profiles": """
**O que é / para que serve**

Resume o **perfil médio de multimorbidade** de cada cluster — prevalência média de cada
comorbidade naquele grupo.

**Eixos e variáveis**

- **Eixos radiais:** comorbidades (0 = ausente em média, 1 = presente em todos)
- **Polígonos coloridos:** um por cluster

**Como interpretar**

Compare **formatos** dos polígonos: picos em hipertensão + diabetes, por exemplo, caracterizam
perfis cardiometabólicos. Clusters com polígonos baixos e uniformes indicam baixa multimorbidade.
""",
    "heatmap_profiles": """
**O que é / para que serve**

Tabela visual das **médias de cada comorbidade por cluster** — alternativa numérica ao gráfico radar.

**Eixos e variáveis**

- **Linhas:** clusters
- **Colunas:** comorbidades
- **Cor/número:** prevalência média (0–1)

**Como interpretar**

Células mais escuras destacam comorbidades que definem aquele perfil. Use para comparar clusters
lado a lado e nomear interpretativamente cada grupo (ex.: “multimorbidade cardiovascular”).
""",
    "bubble_usage": """
**O que é / para que serve**

Cruza duas variáveis escolhidas nos seletores: uma define **faixas Sim/Não no eixo Y** e a outra
**categorias no eixo X**, mostrando a composição percentual de cada cluster em cada célula.

**Eixos e variáveis**

- **Eixo X:** categorias da variável escolhida (ex.: autoavaliação de saúde: Muito Bom, Bom, …)
- **Eixo Y:** blocos Sim/Não da variável de divisão × linhas por cluster
- **Cor:** cluster
- **Tamanho/rótulo da bolha:** % daquele cluster na célula

**Como interpretar**

Padrão default: **uso de serviços × autoavaliação de saúde**. Bolhas grandes revelam onde cada
perfil de multimorbidade se concentra em termos de saúde percebida e utilização de serviços.
""",
    "cross_analysis_bar": """
**O que é / para que serve**

Mostra a **taxa da variável principal** em cada cluster — por exemplo, prevalência de uso de
serviços de saúde ou de uma comorbidade específica.

**Eixos e variáveis**

- **Eixo X:** cluster (rótulo amigável)
- **Eixo Y:** taxa (0–100%), ponderada pelo desenho amostral PNS quando disponível
- **Rótulo na barra:** percentual e tamanho amostral (n)

**Como interpretar**

Compare alturas entre clusters para ver quais perfis de multimorbidade concentram mais a variável
escolhida. Taxas ponderadas refletem estimativas populacionais, não apenas a amostra bruta.
""",
    "cluster_cross_heatmap": """
**O que é / para que serve**

Cruzamento opcional: taxa da **variável principal** em cada combinação **cluster × categoria**
da segunda variável.

**Eixos e variáveis**

- **Linhas:** clusters
- **Colunas:** categorias da variável de cruzamento (ex.: faixa etária, sexo)
- **Cor:** taxa da variável principal naquele subgrupo

**Como interpretar**

Identifique células escuras: indicam subgrupos onde a variável principal é mais prevalente
dentro daquele perfil de saúde. Útil para ver heterogeneidade **dentro** de um mesmo cluster.
""",
    "cluster_size_bar": """
**O que é / para que serve**

Tamanho amostral de cada cluster após o K-Means.

**Eixos e variáveis**

- **Eixo X:** cluster
- **Eixo Y:** número de indivíduos (n)

**Como interpretar**

Clusters muito pequenos podem ser instáveis ou difíceis de generalizar. Verifique se o tamanho
é compatível com análises subsequentes (classificação, cruzamentos).
""",
    "funnel": """
**O que é / para que serve**

Documenta o **funil de pré-processamento** dos microdados PNS 2019 — quantas linhas permanecem
após cada filtro e transformação.

**Eixos e variáveis**

- **Etapas (de cima para baixo):** dados brutos → seleção de colunas → filtro adultos → IMC →
  one-hot → imputação → base processada
- **Cartões:** explicação textual de cada fase

**Como interpretar**

Quedas bruscas indicam filtros restritivos (idade, peso amostral, altura, uso de saúde).
A etapa final deve coincidir com a base usada no agrupamento e na classificação.
""",
    "survey_variable": """
**O que é / para que serve**

Explora uma **variável da pesquisa original** (pré-one-hot): pergunta, respostas possíveis e
frequências na amostra já filtrada.

**Eixos e variáveis**

- **Gráfico:** categorias da variável × contagem (n)
- **Metadados:** código IBGE, tipo, respostas possíveis

**Como interpretar**

Use para entender o significado das colunas antes de ver a matriz binária/categórica derivada.
Compare com o dicionário técnico para o mapeamento código → nome interno.
""",
    "distribution": """
**O que é / para que serve**

Distribuição de uma variável **após pré-processamento** (base binária ou agregada).

**Eixos e variáveis**

- **Eixo X:** categoria ou intervalo (bins)
- **Eixo Y:** contagem de observações

**Como interpretar**

Verifique desbalanceamento (ex.: poucos positivos para uso de serviços) — relevante para escolha
de métricas (F1, PR-AUC) na classificação.
""",
    "roc": """
**O que é / para que serve**

A **curva ROC** avalia o classificador em **todos os limiares** de probabilidade, não só em 0,5.

**Eixos e variáveis**

- **Eixo X:** FPR — taxa de falso positivo (1 − especificidade)
- **Eixo Y:** TPR — sensibilidade (recall)
- **AUC no título:** área sob a curva (resumo global)

**Como interpretar**

Curva **afastada da diagonal** = melhor discriminação entre quem usa e quem não usa serviços.
AUC = 0,5 é aleatório; AUC > 0,8 é considerado bom em muitos contextos clínicos/epidemiológicos.
""",
    "pr": """
**O que é / para que serve**

A curva **precisão-recall (PR)** é especialmente útil quando a classe positiva (uso de serviços)
é **minoritária** na amostra.

**Eixos e variáveis**

- **Eixo X:** recall (sensibilidade) — fração de positivos reais capturados
- **Eixo Y:** precisão — fração de predições positivas que são verdadeiras

**Como interpretar**

Curva alta e estável indica bom desempenho na classe de interesse. Compare com ROC: em
desbalanceamento forte, a PR é mais informativa que a ROC.
""",
    "calibration": """
**O que é / para que serve**

Avalia se as **probabilidades preditas** pelo modelo correspondem às frequências observadas
(calibração).

**Eixos e variáveis**

- **Eixo X:** probabilidade média predita em cada bin
- **Eixo Y:** fração real de positivos naquele bin
- **Diagonal:** calibração perfeita

**Como interpretar**

Pontos sobre a diagonal = previsões bem calibradas. Abaixo da diagonal = modelo superestima risco;
acima = subestima.
""",
    "confusion_matrix": """
**O que é / para que serve**

A **matriz de confusão** confronta rótulos **reais** e **preditos** do classificador de uso
de serviços de saúde.

**Eixos e variáveis**

- **Linhas:** classe real (Negativo / Positivo)
- **Colunas:** classe predita
- **Células:** contagem ou % (seletor no gráfico)
- **TP, TN, FP, FN:** verdadeiros/falsos positivos e negativos

**Como interpretar**

Diagonal principal = acertos. Falsos negativos (FN) = usuários reais não identificados;
falsos positivos (FP) = não usuários classificados como usuários. Base para precisão, recall e F1.
""",
    "importance": """
**O que é / para que serve**

Ranking das variáveis que mais influenciam as predições do modelo.

**Eixos e variáveis**

- **Eixo Y:** nome da variável (sociodemográfica, saúde, cluster, etc.)
- **Eixo X:** score de importância

**Como interpretar**

Barras maiores = maior contribuição. Use para interpretabilidade — quais fatores o modelo
associa ao uso de serviços.
""",
    "importance_native": """
**O que é / para que serve**

**Importância nativa** extraída do próprio algoritmo (ex.: redução de impureza na Floresta Aleatória,
pesos no MLP).

**Eixos e variáveis**

- **Barras:** variáveis ordenadas por importância relativa

**Como interpretar**

Rápida e barata computacionalmente, mas pode favorecer variáveis com muitas categorias.
Compare com importância por permutação para validar.
""",
    "importance_permutation": """
**O que é / para que serve**

**Importância por permutação:** embaralha cada variável e mede a **queda de desempenho** (ex.: F1).
Mais custosa, porém model-agnostic.

**Eixos e variáveis**

- **Barras:** queda média de score ao permutar cada atributo

**Como interpretar**

Valores maiores = variáveis sem as quais o modelo perde mais capacidade preditiva — evidência
mais robusta de relevância causal associativa.
""",
    "gridsearch": """
**O que é / para que serve**

Resultado da **busca em grade (Grid Search)** com validação cruzada: testa combinações de
hiperparâmetros e escolhe a de melhor F1 médio.

**Eixos e variáveis**

- **Eixo X:** tempo médio de treino por combinação (segundos)
- **Eixo Y:** F1 médio na validação cruzada
- **Ponto dourado:** melhor combinação (`best_params`)
- **Tabela:** hiperparâmetros de cada candidato

**Como interpretar**

Prefira combinações com alto F1 e tempo aceitável. O ponto destacado é o usado na execução final.
""",
    "metrics_bar": """
**O que é / para que serve**

Visão comparativa das **principais métricas de classificação** do modelo selecionado.

**Eixos e variáveis**

- **Eixo X:** nome da métrica (F1+, recall, precisão, ROC-AUC, etc.)
- **Eixo Y:** valor (percentual ou escore)

**Como interpretar**

Use junto com as curvas ROC/PR e a matriz de confusão. F1+ resume precisão e recall na classe
de uso de serviços; ROC-AUC resume discriminação global.
""",
    "compare_bar": """
**O que é / para que serve**

Compara a **mesma métrica** entre execuções ou modelos lado a lado.

**Eixos e variáveis**

- **Barras agrupadas:** uma cor por execução, uma posição por modelo
- **Métrica:** selecionada no painel (F1+, recall, silhueta, etc.)

**Como interpretar**

Identifique rapidamente qual execução/modelo supera o baseline do TCC na métrica escolhida.
""",
    "evolution": """
**O que é / para que serve**

Evolução da **métrica primária (F1+)** entre execuções de referência do laboratório.

**Eixos e variáveis**

- **Eixo X:** execução (baseline TCC vs. candidatas)
- **Eixo Y:** F1+ do melhor modelo

**Como interpretar**

Barras mais altas indicam melhoria metodológica. Use para acompanhar iterações da pesquisa
sem misturar runs incompletas ou descartadas.
""",
}
