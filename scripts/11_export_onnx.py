#!/usr/bin/env python
"""Step 11: export a trained run to ONNX and verify it matches PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

from capturecheck.model.export import export_run
from capturecheck.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="Default: the newest run.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    path = export_run(run_dir=args.run, out_path=args.out)
    print(f"exported {path}")


if __name__ == "__main__":
    main()
