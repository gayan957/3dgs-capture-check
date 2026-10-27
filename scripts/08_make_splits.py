#!/usr/bin/env python
"""Step 8: split scenes (never captures) into train and test."""

from __future__ import annotations

import argparse
from pathlib import Path

from capturecheck.io import dataset
from capturecheck.model.splits import TEST_FILE, TRAIN_FILE, make_splits
from capturecheck.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    report = make_splits(test_share=args.test_share, seed=args.seed, out_dir=args.out)
    print(report.summary())

    train = set(dataset.read_scene_list(TRAIN_FILE, path=args.out))
    test = set(dataset.read_scene_list(TEST_FILE, path=args.out))
    assert not (train & test), f"scenes in both splits: {sorted(train & test)}"
    assert train or test, "both splits are empty"
    print(f"OK: {len(train)} train and {len(test)} test scenes, no overlap")


if __name__ == "__main__":
    main()
