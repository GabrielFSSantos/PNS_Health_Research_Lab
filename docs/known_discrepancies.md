# Known Discrepancies

Divergências documentadas entre fontes. **Nenhum valor foi ajustado artificialmente para coincidir com a monografia.**

## Pré-processamento

### Contagem de registros
- **Monografia / notebooks legados:** 87.678 linhas após filtros.
- **Notebook DCC:** 80.393 após NA em comorbidades-chave; 78.999 após idade ≥18.
- **Causa:** filtros adicionais (Colesterol, Hipertensão, Diabetes, Peso) e definição diferente de variável de serviço.
- **Status:** discrepância mantida; DCC não é baseline de reprodução.

### Target: 18.267 vs 22.254 positivos
- **Tabela 3 (monografia):** 18.267 usaram serviço (J014), 69.411 não — **antes** de unir Internação e NearMiss.
- **Tabela 4:** 22.254 / 22.254 — **após** `Serviço OR Internação` e NearMiss-1.
- **Conclusão:** estágios diferentes da mesma pipeline, não erro de contagem.

### Peso amostral
- **Monografia:** "pesos fornecidos pela PNS", menciona V0091.
- **Código R:** `UPA_PNS` renomeado para `Peso (V0091)`; fórmula `1/rowSums(UF, Extrato, UPA_PNS, Ordem, Nº Ordem)`.
- **Problema:** `UPA_PNS` é unidade amostral, não peso do morador (`V00291`).
- **Legacy:** preserva fórmula para reprodução; **research_v2:** usa `V00291` quando disponível.

## Agrupamento

### PCA componentes
- Texto monografia menciona "10 componentes" e linha 90% no gráfico; **código e resultados usam 8**.
- Grid de 20 combinações (Análises Posteriores) confirma melhor silhouette em **8 PCA × 6 clusters**.

### MinMaxScaler
- Citado na monografia (seção tecnologias); **não aplicado** no `Agrupamento.ipynb` (importado, não usado).

### Descrição textual clusters
- Seção 4.4.7 da monografia atribui papéis a clusters 0/1 invertidos em relação a 4.4.1–4.4.2 (possível permutação de labels na redação).
- `Agrupamento.ipynb` markdown descreve apenas clusters 0–3 com k=6 no código.

### Notebook DCC
- Kneedle detecta k=6; código define `n_clusters_242 = 2`.
- Silhouette DCC: 0.3423 vs TCC ~0.5139.

## Classificação

### Pesos no treino
- Monografia afirma incorporação de pesos amostrais; código tem `sample_weight` **comentado** no `fit`.

### Target Encoding e NearMiss
- Aplicados na base inteira antes da CV → leakage documentado.

### GridSearch / CV
- `GridSearchCV` usa o mesmo `StratifiedKFold` do outer loop mas treina em ~1/3 do fold de treino (inner validation).
- `best_model` retornado é do último fold, não agregado.

### MLP
- Grade de hiperparâmetros definida e sobrescrita por `param_grid = {}`.

### Random Forest (snapshot notebook)
- Célula de treino com erro sklearn (`set_params`); métricas da monografia provavelmente de execução Colab anterior (arquivos `.pkl`).

### Cluster como feature
- Target encoding inclui `Cluster`; variável `x_total` **exclui** Cluster do modelo reportado na base Total.

### SVC no perfil legacy_reproduction
- **Baseline TCC** inclui métricas de SVC na monografia.
- **Perfil `legacy_reproduction` atual:** SVC **excluído** da lista de modelos por custo computacional (`GridSearchCV` em ~44k linhas pós-NearMiss).
- Métricas SVC no `tolerances.yaml` permanecem como referência histórica; não são checadas neste perfil.

## Região Norte
- One-hot legado omite **Tocantins** na lista de UFs do Norte.
