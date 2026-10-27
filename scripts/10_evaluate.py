#!/usr/bin/env python
"""Step 10: evaluate a run against the baselines and write the report."""

from __future__ import annotations

import argparse
from pathlib import Path

from capturecheck.model.evaluate import evaluate_run
from capturecheck.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="Default: the newest run.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--colmap-db", type=Path, default=None, help="A COLMAP database.db.")
    args = parser.parse_args()

    setup_logging()
    result = evaluate_run(run_dir=args.run, report_path=args.report, colmap_db=args.colmap_db)
    print(result.summary())
    print(f"\nreport: {result.report_path}")


if __name__ == "__main__":
    main()
