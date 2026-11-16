"""A digest of everything that can change a label.

Two rows may only sit in the same label table if they were measured the same
way. Appending a row that was measured with a different threshold, a different
sampling rate or without DINOv2 would put two rule sets in one training table,
and nothing downstream could tell them apart. So the digest is written beside
the CSVs and checked before anything is appended to them.

Never seeded or keyed from ``hash()``: it is salted per process, so a digest
built that way would differ on every run and every incremental label would look
like a settings change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from capturecheck.config import get_config


def labeling_digest(use_dino: bool = True, cfg=None) -> str:
    """A short, stable digest of the settings the measurements depend on.

    Covers the whole labelling config -- sampling, resize width and every
    threshold -- plus whether DINOv2 ran, because without it a capture has no
    duplicate, diversity or shiny measurement at all.
    """
    cfg = cfg if cfg is not None else get_config()
    payload = {"labeling": asdict(cfg.labeling), "use_dino": bool(use_dino)}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
