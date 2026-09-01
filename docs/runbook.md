# Runbook — PNS Health Research

## Pré-requisitos

- Python 3.12+ (`.venv`)
- Opcional GPU: Python 3.12 em `.venv-gpu` + `pip install -e ".[gpu]"`
- `data/raw/pns2019.parquet`

## Ordem canônica

### 1. Setup

```bash
cd pns-health-research
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. Preparar dados

```bash
.venv/bin/pns-research data prepare --config configs/experiments/legacy_reproduction.yaml
```

Esperado: `rows_out=87678`, `binary_columns=161`, `categorical_columns=61`.

### 3. Experimento completo (legacy TCC)

```bash
.venv/bin/pns-research experiment --config configs/experiments/legacy_reproduction.yaml
```

Estágios internos: preprocessing → clustering → classification → statistics.

### 4. Resume (classificação após clustering válido)

```bash
.venv/bin/pns-research classification \
  --config configs/experiments/legacy_reproduction.yaml \
  --from-run <run_id_com_clustering>
```

### 5. Research v2 (metodologia corrigida)

```bash
.venv/bin/pns-research experiment --config configs/experiments/research_v2.yaml
```

### 6. GPU (classificação research_v2)

```bash
make install-gpu
make run-v2-gpu
```

### 7. Comparar runs

```bash
.venv/bin/pns-research compare --baseline <legacy_run> --candidate <v2_run>
```

## Configuração via YAML

| Arquivo | Uso |
|---------|-----|
| `configs/experiments/*.yaml` | Perfil, estágios, modelos, paralelo, device |
| `configs/classifiers/*.yaml` | `param_grid`, `n_jobs` |
| `configs/features/classification_default.yaml` | Colunas de classificação, target encoding |
| `configs/data/pns2019_columns.yaml` | Mapa colunas PNS |
| `configs/baseline/tolerances.yaml` | Contrato de reprodução |

## Publicar no Git

```bash
git remote add origin <url>
git push -u origin main
```

Não versionar: `data/raw/`, `.venv*`, `.r-env/`, `artifacts/runs/*`.
