# Metrics Contract

## Run layout

```
artifacts/runs/<run_id>/
├── manifest.json
├── config.resolved.yaml
├── data_fingerprint.json
├── reproduction_report.json
├── metrics/
│   ├── preprocessing.json
│   ├── clustering.json
│   ├── classification.json
│   ├── cross_analysis.json
│   └── performance.json
├── tables/
│   ├── elbow_results.parquet
│   ├── pca_explained_variance.parquet
│   ├── cluster_profiles.parquet
│   └── cross_analysis.parquet
└── predictions/
    └── <model>.parquet
```

## Predictions schema

| Column | Type |
|--------|------|
| record_id | string |
| fold | int |
| true_label | int |
| predicted_label | int |
| model | string |
| dataset_partition | string |

## Manifest

See `persistence/artifact_store.py` — run_id, config_hash, data_fingerprint, git_commit, seeds.
