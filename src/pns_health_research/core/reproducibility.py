from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class SeedBundle:
    seed: int


def set_global_seed(seed: int) -> SeedBundle:
    random.seed(seed)
    np.random.seed(seed)
    return SeedBundle(seed=seed)
