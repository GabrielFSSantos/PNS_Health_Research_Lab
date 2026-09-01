# Source Audit

Mapeamento entre monografia, notebooks históricos e notebook DCC-UFOP.
**Decisão:** `legacy_reproduction` segue notebooks históricos (R + Python); `research_v2` corrige problemas metodológicos documentados.

| Etapa | Monografia | Notebook original | Notebook DCC | Decisão |
|-------|------------|-------------------|--------------|---------|
| Aquisição PNS | PNSIBGE 2019 | R `get_pns(2019, design=FALSE)` | Python + rpy2 + cache Parquet | Legacy: R ou Parquet canônico; DCC como referência de evolução |
| Seleção variáveis | ~54 atributos Tabela 1 | 54 colunas PNS renomeadas | ~82 colunas (J/K/H extras) | Legacy: 54 colunas do R |
| Filtro idade | ≥18 anos | `(is.na(Idade)\|Idade>=18)` | `Idade>=18` estrito | Legacy: manter NA de idade + imputação condicional |
| Remoção registros | NA em Peso, Altura, Serviço | Mesmo | NA em Colesterol, Hipertensão, Diabetes, Peso antes da idade | Legacy: só Peso/Altura/Serviço → 87.678 |
| Peso | Pesos PNS / V0091 | `1/rowSums(UF,Extrato,UPA_PNS,Ordem,NºOrdem)` | V00291 se existir; senão mesma fórmula; depois Peso ML | Legacy: fórmula R; v2: V00291 como survey_weight |
| Serviço de Saúde | J014 | Binário Sim/Não | J014 "Buscou Atendimento (2sem)" | Legacy: J014 |
| Internação | J037 | Binário separado | J037 "Internação 12m" | Classificação: OR com Serviço |
| Diabetes gestacional | Incorporar à Diabetes | `Diabetes na Gestação=="Sim" → 1` | Similar | Legacy: preservar |
| Hipertensão gestacional | Incorporar à Hipertensão | Idem | Idem | Legacy: preservar |
| IMC | Peso/altura², faixas | Binário: Baixo/Normal/Sobrepeso/Obesidade | Classe IMC + flag Obesidade | Legacy: faixas binárias |
| Idade/faixas | Faixas etárias | 9 faixas one-hot | Contínua na numérica | Legacy: one-hot na binária |
| Renda | Faixas per capita | 7 faixas one-hot | Faixa Renda PC ordinal | Legacy: one-hot |
| Regiões | 5 regiões | One-hot (sem Tocantins no Norte) | Grandes Regiões | Legacy: incluir bug Tocantins omitido |
| Raça | 5 categorias | 5 one-hot | Ordinal numérica | Legacy: one-hot |
| Estado de saúde | 5 níveis | 5 one-hot | Saúde Geral ordinal | Legacy: one-hot |
| Missing binário | NA → 0 | `replace_na_empty_numeric` → -1 depois 0 implícito | Recodificação explícita | Legacy: NA=0 na binária |
| Missing categórico | NA → -1 | Idem | Idem | Legacy: NA=-1 |
| Pesos finais | Peso amostral | `Peso Amostral` fórmula legada | Peso Pop/Norm/ML | Separar survey_weight vs ml_sample_weight |
| Target Encoding | Zona,Sexo,Raça,Estado Civil | No dataset inteiro | Não implementado | Legacy: leakage; v2: só no treino por fold |
| Base binária | 161 vars, 87.678 linhas | Export xlsx | 78.999 após filtros DCC | Legacy: 87.678 × 161 |
| Base categórica | 61 vars | Export xlsx | 78.999 × 88 | Legacy: 87.678 × 61 |
| NearMiss | NearMiss-1, 50/50 | Antes do split, version=1 | Não | Legacy: antes split; v2: só treino |
| Features agrupamento | 15 comorbidades | Lista fixa | NZV + φ + normalização 0-1 | Legacy: 15 cols sem scaler |
| Normalização PCA | MinMaxScaler citado | **Não aplicado** | StandardScaler | Legacy: sem scaler |
| PCA | 8 componentes (~90% var) | n=8 fixo | n testado até 15, scaler | Legacy: 8, sem scaler |
| MCA | Não | Não | prince (erro na execução) | Opcional extra [mca] |
| Elbow | WCSS, cotovelo 4-6 | k=1..15, n_init=10 | Mediana seeds + Kneedle | Legacy: elbow persistido |
| Kneedle | Não | Não | k=6 detectado, hardcode k=2 | v2: auditável; legacy: k=6 fixo |
| K-Means | k=6, k-means++, n_init=50 | random_state=42 | sample_weight, n_init=100 | Legacy: params TCC |
| LCA/LPA | Não | Não | StepMix (erro) | Opcional extra [lpa] |
| Silhouette | 0.5139 (8 PCA, 6 k) | silhouette_score | 0.3423 (config DCC k=2) | Legacy: ~0.513881 |
| WCSS | 37821.75 | inertia kmeans | Diferente (ponderado) | Legacy: baseline |
| Perfil clusters | 6 clusters, radar/heatmap | groupby mean | Similar | Persistir cluster_profiles.parquet |
| Cruzamento uso/cluster | Bolhas | Plotly | Plotly | cross_analysis layer |
| Classificação prep | Base categórica + cluster | DataBaseClustered.xlsx | Não | Legacy pipeline |
| Train/val/test | CV 5 folds | StratifiedKFold | N/A | Legacy: outer 5 folds |
| GridSearchCV | F1 macro | Inner split + GridSearch bugado | N/A | Legacy: reproduzir bugs |
| Random Forest | 0.778 acc | Grid grande | N/A | Baseline contrato |
| SVC | 0.721 | C/kernel/gamma/degree | N/A | Baseline |
| MLP | 0.779 | param_grid={} | N/A | Baseline |
| XGBoost | 0.785 | Grid 8 params | N/A | Baseline |
| Importância | Permutation + nativa | X completo | N/A | v2: permutation no val fold |
| Confusion matrix | Por fold, somada | Sim | N/A | Persistir parquet |
| Análises posteriores | Figuras TCC | Hardcoded plots | Não | Tabelas only |
