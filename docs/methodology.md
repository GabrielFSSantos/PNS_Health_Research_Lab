# Methodology

## Legacy (`legacy_reproduction`)

Reproduces notebooks R + Python as documented in `docs/source_audit.md`.

## Research v2 (`research_v2`)

- Survey weight from `V00291` when available
- NearMiss only inside training folds
- Nested CV for tuning
- StandardScaler before PCA
- Permutation importance on validation data (planned extension)

## Statistical layer

Analyses marked `unweighted`, `weighted`, or `survey_incomplete` per `docs/known_discrepancies.md`.
