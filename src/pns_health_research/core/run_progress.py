"""Progresso legível no terminal durante execução dos módulos."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator


def _enabled() -> bool:
    if os.environ.get("PNS_QUIET") == "1":
        return False
    if os.environ.get("PNS_PROGRESS") == "1":
        return True
    if "pytest" in sys.modules:
        return False
    return sys.stderr.isatty()


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def terminal_line(message: str, *, end: str = "\n") -> None:
    if not _enabled():
        return
    sys.stderr.write(message + end)
    sys.stderr.flush()


class RunProgress:
    """Reporter de progresso por módulo/estágio (stderr, flush imediato)."""

    def __init__(self, module: str, *, title: str | None = None) -> None:
        self.module = module
        self.title = title or module
        self._started = time.perf_counter()
        self._step_started: float | None = None
        self._current_step: str | None = None

    def banner(self, **details: str | int | float) -> None:
        if not _enabled():
            return
        terminal_line("")
        terminal_line(f"── {self.title} ──")
        for key, value in details.items():
            label = key.replace("_", " ")
            terminal_line(f"   {label}: {value}")

    def step_begin(
        self,
        index: int,
        total: int,
        step_id: str,
        detail: str = "",
        *,
        indent: int = 0,
    ) -> None:
        if not _enabled():
            return
        self._step_started = time.perf_counter()
        self._current_step = step_id
        prefix = "   " + "  " * indent
        suffix = f" — {detail}" if detail else ""
        terminal_line(f"{prefix}[{index}/{total}] {step_id}{suffix} ...", end="")

    def step_skip(self, index: int, total: int, step_id: str, *, indent: int = 0) -> None:
        if not _enabled():
            return
        prefix = "   " + "  " * indent
        terminal_line(f"{prefix}[{index}/{total}] {step_id} (desabilitado)")

    def step_end(self, extra: str = "", *, indent: int = 0) -> None:
        if not _enabled():
            return
        elapsed = time.perf_counter() - (self._step_started or time.perf_counter())
        suffix = f" — {extra}" if extra else ""
        if indent:
            prefix = "   " + "  " * indent
            terminal_line(f"{prefix}↳ {format_duration(elapsed)}{suffix}")
        else:
            terminal_line(f" {format_duration(elapsed)}{suffix}")

    def message(self, text: str, *, indent: int = 0) -> None:
        prefix = "   " + "  " * indent
        terminal_line(f"{prefix}· {text}")

    def finish(self, summary: str = "") -> None:
        if not _enabled():
            return
        total = time.perf_counter() - self._started
        line = f"   ✓ {self.module} concluído em {format_duration(total)}"
        if summary:
            line += f" — {summary}"
        terminal_line(line)
        terminal_line("")


@contextmanager
def timed_step(
    progress: RunProgress,
    index: int,
    total: int,
    step_id: str,
    *,
    detail: str = "",
    enabled: bool = True,
) -> Iterator[None]:
    if not enabled:
        progress.step_skip(index, total, step_id)
        yield
        return
    progress.step_begin(index, total, step_id, detail=detail)
    try:
        yield
    finally:
        progress.step_end()
