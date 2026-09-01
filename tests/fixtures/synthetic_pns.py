"""Synthetic minimal PNS-like fixture for integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pns_health_research.data.schema import LEGACY_RAW_COLUMNS


def make_synthetic_pns(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict = {}
    for col in LEGACY_RAW_COLUMNS:
        data[col] = rng.integers(0, 2, n)
    df = pd.DataFrame(data)
    df["V0001"] = rng.choice(["São Paulo", "Bahia", "Minas Gerais"], n)
    df["V0024"] = rng.integers(1, 5, n)
    df["V0026"] = rng.choice(["Urbano", "Rural"], n)
    df["C006"] = rng.choice(["Homem", "Mulher"], n)
    df["C008"] = rng.integers(18, 80, n)
    df["C009"] = rng.choice(["Branca", "Parda", "Preta"], n)
    df["C011"] = rng.choice(["Casado(a)", "Solteiro(a)"], n)
    df["J014"] = rng.choice(["Sim", "Não"], n)
    df["J037"] = rng.choice(["Sim", "Não"], n)
    df["P00104"] = rng.integers(50, 100, n)
    df["P00404"] = rng.integers(150, 190, n)
    for q in [
        "Q060",
        "Q068",
        "Q074",
        "Q079",
        "Q084",
        "Q088",
        "Q092",
        "Q00201",
        "Q03001",
        "Q06306",
        "Q120",
        "Q124",
        "Q11006",
        "Q11604",
    ]:
        if q in df.columns:
            df[q] = rng.choice(["Sim", "Não"], n)
    return df
