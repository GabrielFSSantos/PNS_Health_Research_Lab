# Architecture

## Flow

```
RawPNSData → Validation → Preprocessing → ClusteringDataset / ClassificationDataset
    → Dimensionality (PCA) → Clustering (K-Means) → Classification → Artifacts
```

## Modules

| Module | Responsibility |
|--------|----------------|
| `config/` | Pydantic experiment configs |
| `data/` | Load, schema, validation, fingerprint |
| `preprocessing/` | Legacy and v2 transforms |
| `dimensionality/` | PCA |
| `clustering/` | K-Means, elbow, alignment |
| `classification/` | Registry, legacy/research CV |
| `analysis/` | Cross-analysis tables |
| `statistics/` | Descriptive stats |
| `experiments/` | Runner, reproduction report |
| `persistence/` | Artifact store |

## Profiles

- **legacy_reproduction**: preserves historical behavior including documented leakage.
- **research_v2**: corrected CV, no global NearMiss/TE.

## Artifacts

Each run: `artifacts/runs/<run_id>/` with manifest, metrics JSON, tables Parquet, predictions.

See [metrics_contract.md](metrics_contract.md).
