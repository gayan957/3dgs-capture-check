#!/usr/bin/env python
"""Step 4: auto-label every capture and write the label CSVs.

A thin wrapper around ``capturecheck label``.
"""

from __future__ import annotations

import sys

from capturecheck.cli import app


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    app(["label", root, *sys.argv[2:]], standalone_mode=True)


if __name__ == "__main__":
    main()
