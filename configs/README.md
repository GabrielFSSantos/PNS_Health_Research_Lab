# Configurações — PNS Health Research Lab

No dia a dia, use **`--profile pns`** ou **`--profile legacy`**. Você só precisa abrir esta pasta se quiser customizar algo.

## O que usar

| Perfil | Comando | Saída |
|--------|---------|-------|
| **pns** (rigoroso; GPU auto se disponível) | `./scripts/run_experiment.sh --profile pns` | `data/bases/pns/` |
| **legacy** (TCC; sempre CPU) | `./scripts/run_experiment.sh --profile legacy` | `data/bases/legacy/` |

## Arquivos visíveis

| Pasta/arquivo | Editar? | Função |
|---------------|---------|--------|
| [`profiles/pns.yaml`](profiles/pns.yaml) | Raramente | Perfil rigoroso completo (3 módulos) |
| [`profiles/legacy.yaml`](profiles/legacy.yaml) | Raramente | Reprodução TCC |
| [`overrides/example.yaml`](overrides/example.yaml) | Copiar para customizar | Diffs sobre um perfil |
| [`shared/datasets.yaml`](shared/datasets.yaml) | Novo dataset | Catálogo PNS |
| [`shared/models.yaml`](shared/models.yaml) | Novos modelos | Registro de classificadores |
| `_internal/` | **Não** | Pipelines, schema_root, overlay TCC |

O resolver (`profile_resolver.py`) monta a config de cada módulo a partir de `profiles/*.yaml`.

## Customizar

```bash
# Mesmo protocolo, outro nome — sem YAML novo
./scripts/run_experiment.sh --profile pns --name sabesp

# Tweaks pontuais
cp configs/overrides/example.yaml configs/overrides/sabesp.yaml
./scripts/run_experiment.sh --profile pns --config configs/overrides/sabesp.yaml --name sabesp
```

## Testes

```bash
./scripts/run_tests.sh          # rápido (default)
./scripts/run_tests.sh smoke    # lê data/bases/ (~segundos)
./scripts/run_tests.sh full     # E2E PNS real (enxuto)
```

## GPU

```bash
./scripts/setup_env.sh --gpu
./scripts/run_experiment.sh --profile pns
```

Perfil `pns`: `device: auto`, `clustering_backend: auto` (cuML se instalado). Perfil `legacy`: sempre CPU/sklearn.
