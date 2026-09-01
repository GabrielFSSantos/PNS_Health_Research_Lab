from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    level: int = logging.INFO,
    run_id: str | None = None,
    log_file: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger("pns_health_research")
    logger.handlers.clear()
    logger.propagate = False
    fmt = "%(levelname)s"
    if run_id:
        fmt += f" run={run_id}"
    fmt += " %(message)s"
    formatter = logging.Formatter(fmt)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.setLevel(level)
    return logger
