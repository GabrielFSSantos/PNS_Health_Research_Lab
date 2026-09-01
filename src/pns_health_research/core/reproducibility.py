from __future__ import annotations

import logging
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


def configure_logging(level: int = logging.INFO, run_id: str | None = None) -> logging.Logger:
    logger = logging.getLogger("pns_health_research")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    fmt = "%(levelname)s"
    if run_id:
        fmt += f" run={run_id}"
    fmt += " %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
