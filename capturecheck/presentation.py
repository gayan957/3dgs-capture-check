"""Turning a result into something a person reads: score bands and safe JSON.

The CLI and the HTTP API both report the same capture, so the band a score
falls into and the way a missing measurement is written live here rather than
in either of them. Two copies of ``0.75`` would eventually disagree.
"""

from __future__ import annotations

import math

#: Score cut-offs, highest first. Calibrated in evaluation, not guessed.
QUALITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.75, "likely good"),
    (0.55, "fair"),
    (0.35, "likely poor"),
    (0.0, "likely unusable"),
)


def quality_band(score: float) -> str:
    """Turn a 0-1 score into words."""
    for cutoff, name in QUALITY_BANDS:
        if score >= cutoff:
            return name
    return "unknown"


def json_safe(value):
    """Make a value strictly JSON-serialisable.

    A missing measurement is NaN, and ``json.dumps`` writes that as a bare
    ``NaN``, which Python reads back but ``jq``, JavaScript and most other
    parsers refuse. Machine-readable output is there to be consumed by
    something else, so a missing value becomes ``null`` rather than a token
    only Python understands.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, str, type(None))):
        return value
    if isinstance(value, int):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if (math.isnan(number) or math.isinf(number)) else number
