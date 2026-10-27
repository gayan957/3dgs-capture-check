#!/usr/bin/env python
"""Step 12: does each degradation actually move the measurement it claims to?

A broken copy is only useful as training data if the defect it is labelled with
is really there. This compares the clean captures against each kind of broken
copy on the measurement that defect should move, and fails loudly when one does
not shift in the expected direction.

It is worth running after any change to `capturecheck/degrade/ops.py`,
`configs/degrade.yaml` or a threshold: it is what caught `rotation_only`
labelling walking captures as `no_walking`, and the `noise` op compressing away
the noise it had just added.

    python scripts/12_check_degradations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capturecheck.config import get_config
from capturecheck.io import dataset
from capturecheck.utils.logging import setup_logging

#: (degrade op, the measurement it should move, which way it should move)
EXPECTED: tuple[tuple[str, str, str], ...] = (
    ("blur", "blur_ratio", "up"),
    ("exposure_jitter", "brightness_std", "up"),
    ("noise", "noise_mean", "up"),
    ("subsample", "unique_views", "down"),
    ("rotation_only", "walked_ratio", "down"),
    ("drop_sector", "azimuth_span_deg", "down"),
    ("drop_gap", "overlap_low_ratio", "up"),
)


def mean_of(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=cfg.paths.resolve("labels"))
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero when a degradation does not move its measurement.",
    )
    args = parser.parse_args()
    setup_logging()

    videos = dataset.read_video_labels(out_dir=args.labels)
    if videos.empty:
        raise SystemExit(f"no labels in {args.labels}; run `capturecheck label data` first")

    clean = videos[videos["source"] != "degraded"]
    if clean.empty:
        raise SystemExit("no clean captures to compare against")

    rows = []
    for op, column, direction in EXPECTED:
        broken = videos[videos["degrade_type"] == op]
        if broken.empty:
            rows.append(
                {"op": op, "measurement": column, "clean": float("nan"),
                 "broken": float("nan"), "want": direction, "copies": 0, "verdict": "no copies"}
            )
            continue

        clean_value = mean_of(clean, column)
        broken_value = mean_of(broken, column)
        if not (np.isfinite(clean_value) and np.isfinite(broken_value)):
            verdict = "not measured"
        elif direction == "up":
            verdict = "ok" if broken_value > clean_value else "WRONG WAY"
        else:
            verdict = "ok" if broken_value < clean_value else "WRONG WAY"

        rows.append(
            {"op": op, "measurement": column, "clean": round(clean_value, 3),
             "broken": round(broken_value, 3), "want": direction,
             "copies": len(broken), "verdict": verdict}
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Are the strengths ordered? A "heavy" copy that is no worse than a "light"
    # one gives the ranking loss a pair it cannot learn anything from.
    print("\nStrength ordering")
    for op, column, direction in EXPECTED:
        values = []
        for strength in cfg.degrade.strengths:
            subset = videos[
                (videos["degrade_type"] == op) & (videos["degrade_strength"] == strength)
            ]
            values.append(mean_of(subset, column) if not subset.empty else float("nan"))
        known = [v for v in values if np.isfinite(v)]
        if len(known) < 2:
            continue
        ordered = (
            all(a < b for a, b in zip(known, known[1:], strict=False))
            if direction == "up"
            else all(a > b for a, b in zip(known, known[1:], strict=False))
        )
        shown = ", ".join(
            f"{name}={value:.3g}"
            for name, value in zip(cfg.degrade.strengths, values, strict=False)
        )
        print(f"  {op:<16} {column:<20} {shown}   {'ok' if ordered else 'NOT ORDERED'}")

    failures = table[table["verdict"] == "WRONG WAY"]
    if len(failures):
        print(f"\n{len(failures)} degradation(s) do not move their measurement as expected:")
        for _, row in failures.iterrows():
            print(
                f"  {row['op']}: {row['measurement']} went from "
                f"{row['clean']} to {row['broken']}"
            )
        print(
            "\nEither the op is not creating the defect, or the check is not seeing it. "
            "A copy labelled with a defect it does not have is worse than no copy."
        )
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nEvery degradation moves its measurement in the expected direction.")


if __name__ == "__main__":
    main()
