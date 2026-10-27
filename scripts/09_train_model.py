#!/usr/bin/env python
"""Step 9: train the quality model (stage A pretext, then stage B ranking)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capturecheck.model.train import train_run
from capturecheck.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--no-stage-a", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    result = train_run(
        config_path=args.config,
        run_id=args.run_id,
        seeds=args.seed,
        run_stage_a=not args.no_stage_a,
    )
    print(json.dumps(result.metrics, indent=2, default=float))
    print(f"run written to {result.run_dir}")


if __name__ == "__main__":
    main()
