# ADR 001: Python as primary language

R isolated to optional `[acquisition]` extra; all pipelines run from Parquet.

# ADR 002: Filesystem artifact store

JSON/YAML/Parquet on disk; no database in v1.

# ADR 003: Legacy vs research_v2 profiles

Separate configs and run directories; never overwrite artifacts.

# ADR 004: Survey weight vs ML sample weight

`survey_weight` and `ml_sample_weight` are distinct columns.

# ADR 005: Cross-validation strategy

Legacy reproduces notebook bugs via `legacy_cv: true`; research_v2 uses nested CV.
