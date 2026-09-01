from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def fingerprint_dataframe(df: pd.DataFrame, logical_path: str) -> dict[str, Any]:
  schema = {c: str(df[c].dtype) for c in df.columns}
  payload = f"{len(df)}|{len(df.columns)}|{sorted(schema.items())}"
  digest = hashlib.sha256(payload.encode()).hexdigest()
  return {
      "logical_path": logical_path,
      "n_rows": len(df),
      "n_columns": len(df.columns),
      "schema": schema,
      "hash": digest,
  }


def save_fingerprint(path: Path, fp: dict[str, Any]) -> None:
    path.write_text(json.dumps(fp, indent=2), encoding="utf-8")
