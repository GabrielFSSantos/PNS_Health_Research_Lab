# Data Required

## Arquivo ausente

A Pesquisa Nacional de Saúde (PNS) 2019 **não está incluída** neste repositório.

## O que obter

1. **Microdados PNS 2019** (IBGE), ou
2. Executar `uv run pns-research data acquire` com extra `[acquisition]` (R + PNSIBGE).

Dimensão de referência (notebook DCC): ~279.382 linhas × ~1.082 colunas brutas.

## Onde colocar

| Caminho | Formato |
|---------|---------|
| `data/raw/pns2019.parquet` | Preferido |
| `data/raw/pns2019.csv` | Alternativa |

## Colunas mínimas (legacy)

Variáveis PNS originais usadas no notebook R (54 colunas após seleção). Ver `configs/data/pns2019.yaml` e `src/pns_health_research/data/schema.py`.

**Peso do morador:** preferir `V00291` (peso amostral calibrado IBGE). Não confundir com `UPA_PNS` (unidade amostral).

## Intermediários históricos (opcionais)

Estes arquivos **não são necessários** se você executar a pipeline completa:

- `DataBaseProcessedBinary.xlsx`
- `DataBaseProcessedReal.xlsx`
- `DataBaseClustered.xlsx`

## Comandos após disponibilizar os dados

```bash
cd pns-health-research
uv sync

# Preparar dados (legacy)
uv run pns-research data prepare --config configs/experiments/legacy_reproduction.yaml

# Reproduzir TCC
uv run pns-research experiment run --config configs/experiments/legacy_reproduction.yaml

# Metodologia corrigida
uv run pns-research experiment run --config configs/experiments/research_v2.yaml
```

## Sem dados

Com fixture sintética em `tests/fixtures/`, a pipeline roda em CI sem a PNS completa. Testes de regressão contra o baseline são **skipped** até `data/raw/pns2019.parquet` existir.
