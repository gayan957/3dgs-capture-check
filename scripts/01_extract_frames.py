#!/usr/bin/env python
"""Step 1: extract and cache the sampled frames of every capture.

Not required by the rest of the pipeline (frames are sampled on demand), but
useful for looking at what the checks actually see.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from capturecheck.config import get_config
from capturecheck.io import dataset
from capturecheck.io.load import load_capture, save_frames
from capturecheck.utils.logging import setup_logging


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=cfg.paths.resolve("raw"))
    parser.add_argument("--out", type=Path, default=cfg.paths.resolve("frames"))
    args = parser.parse_args()

    setup_logging()
    refs = dataset.discover_all(args.raw)
    for ref in refs:
        capture = load_capture(ref.path, capture_id=ref.capture_id)
        written = save_frames(capture.frames, args.out / ref.capture_id)
        print(f"{ref.capture_id}: {len(written)} frames -> {args.out / ref.capture_id}")
    print(f"done: {len(refs)} captures")


if __name__ == "__main__":
    main()
