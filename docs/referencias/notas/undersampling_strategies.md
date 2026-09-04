# Estratégias de undersampling (NearMiss)

**PDF versionado:** [../pdfs/undersampling_strategies.pdf](../pdfs/undersampling_strategies.pdf)

## Resumo para o lab

Revisão de técnicas de **undersampling** para classes desbalanceadas, incluindo **NearMiss-1**: seleciona amostras da classe majoritária mais próximas da minoritária.

## Uso nas fases

| Fase | Aplicação |
|------|-----------|
| F0 (TCC) | `nearmiss.enabled: true` no overlay legado |
| F2 (default) | NearMiss **desligado** — avaliação sem vazamento de balanceamento global |

## O que tomamos / o que difere

- **Tomamos:** NearMiss reproduz o notebook quando `legacy.yaml` está ativo
- **Diferente:** no default evolutivo, o desbalanceamento é tratado via métricas (`f1_positive`) e pesos, não via undersampling pré-split

## Referência canônica

Mani & Zhang (2003) — ver [bibliografia.bib](../bibliografia.bib).
