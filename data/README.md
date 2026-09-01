# Dados

Microdados PNS **não** são versionados neste repositório.

## Baixar (primeira vez)

Requisitos: R + `PNSIBGE`, ambiente com `./scripts/setup_env.sh --acquisition`.

```bash
./scripts/run_download.sh
```

O parquet fica em `data/raw/` conforme `configs/datasets.yaml`. Outro dataset: `--dataset pns_2019`.

Alternativa: `./scripts/run_experiment.sh` baixa automaticamente se o arquivo não existir e `defaults.download.if_missing: true`.

Documentação: [docs/documentacao.md](../docs/documentacao.md).
