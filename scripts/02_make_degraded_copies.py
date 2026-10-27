#!/usr/bin/env python
"""Step 2: make broken copies of the good captures (free, exact labels)."""

from __future__ import annotations

import argparse
from pathlib import Path

from capturecheck.config import get_config
from capturecheck.degrade.make_variants import make_all
from capturecheck.utils.logging import setup_logging


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=cfg.paths.resolve("raw"))
    parser.add_argument("--out", type=Path, default=cfg.paths.resolve("degraded"))
    parser.add_argument("--strength", action="append", default=None)
    parser.add_argument("--op", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging()
    variants = make_all(
        raw_dir=args.raw, out_dir=args.out, ops=args.op,
        strengths=args.strength, overwrite=args.overwrite,
    )
    for variant in variants:
        print(f"{variant.capture_id}: {variant.n_frames} frames  {variant.labels}")
    print(f"done: {len(variants)} copies")


if __name__ == "__main__":
    main()
