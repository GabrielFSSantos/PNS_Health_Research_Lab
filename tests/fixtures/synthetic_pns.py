"""Synthetic minimal PNS-like fixture for integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pns_health_research.data.schema import LEGACY_RAW_COLUMNS

SYNTHETIC_PNS_ROWS = 10


def make_synthetic_pns(n: int = SYNTHETIC_PNS_ROWS, seed: int = 42) -> pd.DataFrame:
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
    df["V00291"] = rng.uniform(500, 5000, n)
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

    for col in [
        "J024",
        "J025",
        "J02901",
        "J035",
        "J042",
        "J043",
        "J046",
        "J047",
        "J048",
        "J051",
        "K052",
    ]:
        if col in df.columns:
            df[col] = rng.choice(["Sim", "Não"], n)

    for col in ["J026", "J044", "J049", "K04301"]:
        if col in df.columns:
            df[col] = rng.choice(["Sim", "Não", "Não sabe/Não lembra"], n)

    for col in ["J03001", "J032", "J03301", "J034"]:
        if col in df.columns:
            df[col] = rng.choice(["Sim, todos", "Sim, alguns", "Não, nenhum"], n)

    for col in ["J012", "J038", "K04302"]:
        if col in df.columns:
            df[col] = rng.integers(0, 5, n)

    if "J01101" in df.columns:
        df["J01101"] = rng.choice(
            ["Até 1 ano", "Mais de 1 ano a 2 anos", "Nunca foi ao médico"], n
        )
    if "H001" in df.columns:
        df["H001"] = rng.choice(["Até 15 dias", "Mais de 6 meses até 1 ano"], n)
    if "H003" in df.columns:
        df["H003"] = rng.choice(
            ["Doença ou outro problema de saúde", "Exame médico periódico"], n
        )
    if "H004" in df.columns:
        df["H004"] = rng.choice(
            [
                "Unidade básica de saúde (posto ou centro de saúde ou unidade de saúde da família)",
                "UPA (Unidade de Pronto Atendimento)",
            ],
            n,
        )
    if "J03102" in df.columns:
        df["J03102"] = rng.choice(
            ["Não tinha dinheiro para comprar", "Não achou necessário"], n
        )

    # Fixtures pequenas: alterna J014/J037 para classes mistas no alvo de utilização.
    if n <= SYNTHETIC_PNS_ROWS:
        for i in range(n):
            label = "Sim" if i % 2 == 0 else "Não"
            df.loc[i, "J014"] = label
            df.loc[i, "J037"] = label

    return df
