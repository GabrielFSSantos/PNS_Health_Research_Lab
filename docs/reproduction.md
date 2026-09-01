# Reproduction

1. Place PNS 2019 at `data/raw/pns2019.parquet`
2. `uv run pns-research experiment run --config configs/experiments/legacy_reproduction.yaml`
3. Inspect `artifacts/runs/<run_id>/reproduction_report.json`

Without raw data, run integration tests with synthetic fixture only.

Expected values: [baseline_contract.md](baseline_contract.md)

Tolerances: [configs/baseline/tolerances.yaml](../configs/baseline/tolerances.yaml)
