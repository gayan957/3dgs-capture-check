#!/usr/bin/env python
"""Step 3: pre-compute and cache the DINOv2 features of every capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from capturecheck.config import get_config
from capturecheck.features.dino import default_embedder
from capturecheck.io import dataset
from capturecheck.io.load import load_capture
from capturecheck.utils.logging import setup_logging


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=cfg.paths.resolve("raw"))
    parser.add_argument("--degraded", type=Path, default=cfg.paths.resolve("degraded"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging()
    refs = dataset.discover_all(args.raw, args.degraded)
    embedder = default_embedder()
    for ref in refs:
        capture = load_capture(ref.path, capture_id=ref.capture_id)
        emb, idx = embedder.embed_capture(
            ref.capture_id, capture.frames, use_cache=not args.overwrite
        )
        print(f"{ref.capture_id}: {emb.shape[0]} embeddings of {capture.n_frames} frames")
    print(f"done: {len(refs)} captures -> {cfg.paths.resolve('features')}")


if __name__ == "__main__":
    main()
