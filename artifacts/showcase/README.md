# Showcase — baseline TCC

Esta pasta contém **apenas** a reprodução do TCC versionada no Git (`pesquisa-original-legacy`), para que o dashboard funcione ao clonar o repositório sem reexecutar o pipeline.

## Conteúdo

| Métrica | Valor | Modelo |
|---------|-------|--------|
| F1+ | **0,792** | XGBoost |
| Silhueta | 0,481 | K=6 |

Run fonte: `20260901T053650Z_legacy_reproduction_e4f5337`

## Novas execuções

**Não é necessário exportar para esta pasta.** Rode experimentos normalmente:

```bash
./scripts/run_experiment.sh --config configs/experiment.yaml --name "Minha run"
```

Runs concluídas em `artifacts/runs/` aparecem automaticamente no dashboard.

Para renomear ou marcar a baseline, edite `configs/dashboard/run_overrides.yaml`.

Veja [docs/documentacao.md](../../docs/documentacao.md).
