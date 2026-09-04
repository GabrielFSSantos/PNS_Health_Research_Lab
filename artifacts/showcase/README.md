# Showcase — baseline TCC

Esta pasta contém **apenas** a reprodução do TCC versionada no Git (`pesquisa-original-legacy`), para que o dashboard funcione ao clonar o repositório sem reexecutar o pipeline.

## Conteúdo

| Métrica | Valor | Modelo |
|---------|-------|--------|
| F1+ | **0,792** | XGBoost |
| Silhueta | 0,481 | K=6 |

Run fonte: `20260901T053650Z_legacy_reproduction_e4f5337`

## Novas execuções

**Não é necessário exportar para esta pasta.** Use o pipeline modular:

```bash
./scripts/run_experiment.sh --profile legacy   # → data/bases/legacy/
./scripts/run_experiment.sh --profile pns      # → data/bases/pns/
```

Tolerâncias TCC para regressão: `configs/_internal/tcc_overlay.yaml`.

Para renomear ou marcar a baseline no dashboard, edite `configs/dashboard/run_overrides.yaml`.

Veja [docs/documentacao.md](../../docs/documentacao.md) e [configs/README.md](../../configs/README.md).
