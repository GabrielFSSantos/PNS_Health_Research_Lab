"""Regras de insights."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class InsightLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"
    DANGER = "danger"


@dataclass
class Insight:
    message: str
    level: InsightLevel = InsightLevel.INFO


SILHOUETTE_WEAK = 0.25
