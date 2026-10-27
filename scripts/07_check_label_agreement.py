#!/usr/bin/env python
"""Step 7: how well do the automatic labels match what people say?

Inputs
  data/labels/video_labels.csv   the automatic labels
  data/labels/human_labels.csv   reviewer decisions (two reviewers on a shared
                                 subset gives the inter-reviewer agreement too)

Output
  reports/label_agreement.md     per label: Cohen's kappa between reviewers, and
                                 precision/recall/F1 of the automatic label
                                 against the agreed human label, plus a short
                                 list of thresholds worth changing.

Without a report like this the thresholds in configs/labeling.yaml are just
guesses, so this step is what makes them defensible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capturecheck.config import get_config
from capturecheck.io import dataset
from capturecheck.labeling import schema
from capturecheck.utils.logging import setup_logging

#: Which threshold to change when a label is systematically wrong.
THRESHOLD_HINTS: dict[str, str] = {
    "motion_blur": "blur.sharp_min_laplacian_var / blur.video_blur_ratio",
    "lighting": "light.video_brightness_std / light.video_bad_ratio",
    "no_walking": "motion.walked_ratio_min",
    "poor_coverage": "coverage.max_empty_bins",
    "too_few_views": "views.min_unique_views / views.duplicate_cosine",
    "low_overlap": "overlap.min_overlap / overlap.low_overlap_ratio",
    "mixed_cameras": "exif.max_cameras / exif.max_sizes",
    "reflective": "shiny.prob / shiny.video_ratio",
    "low_texture": "texture.min_features / texture.video_ratio",
    "noisy": "noise.frame_max / noise.video_ratio",
}


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement between two reviewers, corrected for agreement by chance."""
    if a.size == 0:
        return float("nan")
    observed = float(np.mean(a == b))
    p_a = float(np.mean(a))
    p_b = float(np.mean(b))
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


def precision_recall(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if not np.isfinite(precision) or not np.isfinite(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def agreed_labels(human: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per capture: the agreed human label, and the per-reviewer votes.

    Where two reviewers disagree, the capture is counted as positive only if at
    least half of them said so, and the disagreement is reported.
    """
    votes = []
    for capture_id, group in human.groupby("capture_id"):
        row = {"capture_id": capture_id, "n_reviews": len(group)}
        for name in schema.VIDEO_LABEL_NAMES:
            if name not in group.columns:
                continue
            values = pd.to_numeric(group[name], errors="coerce").dropna().to_numpy()
            row[name] = int(values.mean() >= 0.5) if values.size else np.nan
            row[f"{name}__disagreed"] = int(values.size > 1 and len(set(values.tolist())) > 1)
        votes.append(row)
    agreed = pd.DataFrame(votes)
    pairs = human[human["capture_id"].isin(
        human.groupby("capture_id")["reviewer"].nunique().loc[lambda s: s >= 2].index
    )]
    return agreed, pairs


def reviewer_kappa(pairs: pd.DataFrame, name: str) -> tuple[float, int]:
    """Cohen's kappa between the first two reviewers of each shared capture."""
    first, second = [], []
    for _, group in pairs.groupby("capture_id"):
        unique = group.drop_duplicates(subset=["reviewer"])
        if len(unique) < 2 or name not in unique.columns:
            continue
        values = pd.to_numeric(unique[name], errors="coerce").dropna().to_numpy()
        if values.size < 2:
            continue
        first.append(int(values[0]))
        second.append(int(values[1]))
    if not first:
        return float("nan"), 0
    return cohen_kappa(np.array(first), np.array(second)), len(first)


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=cfg.paths.resolve("labels"))
    parser.add_argument(
        "--out", type=Path, default=cfg.paths.resolve("reports") / "label_agreement.md"
    )
    parser.add_argument(
        "--min-precision", type=float, default=0.7,
        help="Labels below this are listed as needing recalibration.",
    )
    args = parser.parse_args()
    setup_logging()

    auto = dataset.read_video_labels(out_dir=args.labels)
    human = dataset.read_human_labels(out_dir=args.labels)
    if auto.empty:
        raise SystemExit(f"no automatic labels in {args.labels}; run `capturecheck label` first")
    if human.empty:
        raise SystemExit(
            f"no human labels in {args.labels}; review some captures first "
            "(streamlit run capturecheck/review/app.py)"
        )

    agreed, pairs = agreed_labels(human)
    merged = agreed.merge(
        auto, on="capture_id", how="inner", suffixes=("_human", "_auto")
    )
    if merged.empty:
        raise SystemExit("no capture has both an automatic and a human label")

    rows = []
    for name in schema.VIDEO_LABEL_NAMES:
        human_col, auto_col = f"{name}_human", f"{name}_auto"
        if human_col not in merged.columns or auto_col not in merged.columns:
            continue
        truth = pd.to_numeric(merged[human_col], errors="coerce")
        predicted = pd.to_numeric(merged[auto_col], errors="coerce")
        keep = truth.notna() & predicted.notna()
        truth_v = truth[keep].astype(int).to_numpy()
        pred_v = predicted[keep].astype(int).to_numpy()
        if truth_v.size == 0:
            continue
        precision, recall, f1 = precision_recall(truth_v, pred_v)
        kappa, n_pairs = reviewer_kappa(pairs, name)
        rows.append(
            {
                "label": name,
                "captures": int(truth_v.size),
                "human positives": int(truth_v.sum()),
                "auto positives": int(pred_v.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "reviewer kappa": kappa,
                "kappa pairs": n_pairs,
            }
        )

    table = pd.DataFrame(rows)
    needs_work = table[
        (table["precision"] < args.min_precision) | (table["recall"] < args.min_precision)
    ]

    def fmt(value) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a"
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = [
        "# Label agreement",
        "",
        f"Captures with both automatic and human labels: **{len(merged)}**  ",
        f"Captures reviewed by two or more people: **{pairs['capture_id'].nunique()}**",
        "",
        "`precision` and `recall` are of the *automatic* label against the agreed",
        "human label. `reviewer kappa` is Cohen's kappa between the first two",
        "reviewers of each shared capture: it is the ceiling, because the automatic",
        "labels cannot be more reliable than the people they are measured against.",
        "",
        "| label | captures | human + | auto + | precision | recall | F1 "
        "| reviewer kappa | pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| `{row['label']}` | {row['captures']} | {row['human positives']} | "
            f"{row['auto positives']} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
            f"{fmt(row['f1'])} | {fmt(row['reviewer kappa'])} | {row['kappa pairs']} |"
        )

    lines += ["", "## Thresholds to look at", ""]
    if needs_work.empty:
        lines.append(
            f"No label is below {args.min_precision:.2f} precision or recall on this sample."
        )
    else:
        for _, row in needs_work.iterrows():
            direction = []
            if np.isfinite(row["precision"]) and row["precision"] < args.min_precision:
                direction.append("too many false alarms: make the threshold stricter")
            if np.isfinite(row["recall"]) and row["recall"] < args.min_precision:
                direction.append("missing real cases: make the threshold looser")
            lines.append(
                f"- **{row['label']}** (precision {fmt(row['precision'])}, "
                f"recall {fmt(row['recall'])}): {'; '.join(direction)}. "
                f"Tune `{THRESHOLD_HINTS.get(row['label'], 'configs/labeling.yaml')}`."
            )

    disagreement_cols = [c for c in agreed.columns if c.endswith("__disagreed")]
    if disagreement_cols:
        totals = agreed[disagreement_cols].sum().sort_values(ascending=False)
        contested = [(c.replace("__disagreed", ""), int(v)) for c, v in totals.items() if v > 0]
        if contested:
            lines += ["", "## Where reviewers disagreed with each other", ""]
            for name, count in contested:
                lines.append(f"- `{name}`: {count} capture(s)")
            lines.append("")
            lines.append(
                "A label people cannot agree on is a label worth redefining in "
                "`capturecheck/labeling/schema.py`, not just retuning."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
