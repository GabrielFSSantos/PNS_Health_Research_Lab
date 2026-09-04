# Tratamento de ausentes e códigos especiais — PNS 2019

Nota de apoio à [trajetoria §2.1](../trajetoria.md#21-decisões-legacy-vs-pns-justificativas) (itens PRE-01, PRE-02, PRE-07). Dicionário do lab: [`configs/data/dictionary.yaml`](../../../configs/data/dictionary.yaml), metadados: [`configs/data/survey_metadata.yaml`](../../../configs/data/survey_metadata.yaml).

## Contexto

A PNS 2019 (IBGE) registra, para cada variável, não apenas respostas categóricas (`Sim` / `Não`), mas também situações em que a pergunta **não se aplica**, foi **ignorada** ou **não respondida**. O microdado entrega esses estados como categorias textuais ou valores ausentes (`NA`), conforme o processamento do `PNSIBGE` / exportação usada no TCC.

Boas práticas em análise de inquéritos (Barros & Victora, 2005; manual PNS/IBGE) recomendam:

1. **Distinguir** “não possui a condição” de “não sabe / não respondeu / não aplicável”.
2. **Documentar** a regra de recodificação e reportar **análise de sensibilidade** quando ausentes forem tratados como zero.
3. Usar **peso amostral oficial** (`V00291`) e estrutura de amostragem quando a inferência for populacional.

## Códigos e estados relevantes no lab

| Estado na base bruta | Tratamento no legado (`_yes_no_binary`) | Tratamento categórico (`fill_missing`) | Risco |
|---------------------|----------------------------------------|----------------------------------------|-------|
| `Sim` | `1` | mantém valor numérico | — |
| `Não` | `0` | mantém valor numérico | — |
| `NA` / célula vazia | `0` | `-1` (numérico) | **Binário:** equivale a “sem doença” |
| Resposta diferente de Sim/Não (ex.: “Ignorado”, “Não aplicável”) | `0` | depende da codificação prévia | **Binário:** colapsa com “não” |

A função `_yes_no_binary` em `features.py` implementa literalmente:

```python
return series.apply(lambda x: 0 if pd.isna(x) else int(x == yes_value))
```

Qualquer valor que não seja exatamente `Sim` (ou o `yes_value` configurado) vira `0`, **incluindo** ausência e códigos especiais.

## Comorbidades (15 condições + limitações)

No TCC e no software, as colunas de doença crônica passam por `_yes_no_binary` **antes** do estágio `impute_missing`. Por isso o funil do showcase registra `cells_imputed: 0` na etapa de imputação: os `NA` já foram convertidos em `0` na binarização.

**Argumento informal do TCC:** “se a pessoa não respondeu, tratamos como ausência da condição”. Esse argumento:

- Não distingue **não aplicável** (ex.: gestante em variável não pertinente) de **desconhecido**.
- Não aparece com análise de sensibilidade na monografia.
- Contradiz o tratamento das variáveis categóricas na mesma pipeline (`-1` para ausente).

## Variáveis com lógica invertida (sem documentação explícita)

| Variável | Regra no código | Observação |
|----------|-----------------|------------|
| Agente Comunitário | `0` se `Nunca recebeu` | Inversão semântica (presença = não recebeu) |
| Tabaco | `0` se `Não fumo atualmente` | Idem |

Essas regras reproduzem o notebook R, mas a monografia não detalha a tabela de recodificação completa.

## Idade e gestação

- **`impute_age_by_target`:** preenche `Idade` ausente pela média **dentro de cada classe do target** (`Serviço de Saúde`). Isso usa informação do desfecho e constitui **vazamento** em modelos supervisionados (ver PRE-03 na auditoria).
- **Gestação:** hipertensão/diabetes gestacional são fundidas nas comorbidades principais e as colunas gestacionais são removidas. A monografia menciona comorbidades, mas não o protocolo de fusão linha a linha.

## Peso amostral

| Modo | Implementação | Uso |
|------|---------------|-----|
| Legado | `1 / soma(UF, Extrato, UPA, Ordem domicílio, Nº ordem)` | Reprodução TCC |
| Evolutivo | `V00291` (`Peso Amostral`) | Alinhado ao manual PNS |

Para inferência sobre a população brasileira, a fórmula legada **não** substitui o peso oficial sem validação empírica.

## Recomendações para texto acadêmico

1. Tabela “estado original → código analítico” por família de variáveis (comorbidade, demografia, comportamento).
2. Subseção **Limitações — tratamento de ausentes**, citando percentual de `NA` por comorbidade antes da binarização.
3. Análise de sensibilidade: (a) `NA` → `0`; (b) `NA` → exclusão listwise; (c) categoria explícita “ausente”.
4. Não usar o argumento “pulou = sem doença” sem evidência clínica ou epidemiológica para aquela variável PNS.

## Implementação no lab

O perfil `pns` (`configs/_internal/pipelines/preprocessing/pns.yaml`) aplica `binary_encoding.policy: explicit_missing` com `missing_value: -1`. Ver [trajetoria §2.1](../trajetoria.md#21-decisões-legacy-vs-pns-justificativas).

## Referências

- IBGE — PNS 2019: manual e dicionário de variáveis.
- Barros, A. J. D.; Victora, C. G. (2005). Indicadores socioeconômicos… *Rev. Saúde Pública*.
- Monografia TCC: [nota](monografia_tcc.md) · [PDF](../../research/monografia.pdf).
- Trajetória e decisões: [trajetoria.md](../trajetoria.md).
