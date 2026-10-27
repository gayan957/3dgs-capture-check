"""Time each pipeline step, so the 60 s per-capture budget can be checked."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from capturecheck.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Timings:
    """Accumulates durations per named step."""

    steps: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.steps[name] = self.steps.get(name, 0.0) + seconds

    @property
    def total(self) -> float:
        return float(sum(self.steps.values()))

    def as_dict(self) -> dict[str, float]:
        out = {k: round(v, 3) for k, v in self.steps.items()}
        out["total"] = round(self.total, 3)
        return out

    def report(self) -> str:
        rows = [f"  {name:<22} {sec:6.2f} s" for name, sec in self.steps.items()]
        rows.append(f"  {'TOTAL':<22} {self.total:6.2f} s")
        return "\n".join(rows)

    @contextmanager
    def step(self, name: str, log_it: bool = True):
        """Time a block and record it under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.add(name, elapsed)
            if log_it:
                log.info("%s took %.2f s", name, elapsed)


@contextmanager
def timed(name: str, log_it: bool = True):
    """Standalone timer for a block of code; yields a one-element list with the duration."""
    holder: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
        if log_it:
            log.info("%s took %.2f s", name, holder[0])
