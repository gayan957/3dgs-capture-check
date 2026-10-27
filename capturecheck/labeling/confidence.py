"""When should a person look at this capture?

Two reasons to ask for review: a measurement sits close enough to its threshold
that a small recalibration would flip the label, or two independent checks
disagree (for example "lots of different views" next to "no camera motion").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capturecheck.config import LabelingConfig, labeling_config

#: (stat column, dotted path to its threshold in the labelling config)
_BORDERLINE_CHECKS: tuple[tuple[str, str], ...] = (
    ("blur_ratio", "blur.video_blur_ratio"),
    ("brightness_std", "light.video_brightness_std"),
    ("walked_ratio", "motion.walked_ratio_min"),
    ("overlap_low_ratio", "overlap.low_overlap_ratio"),
    ("shiny_ratio", "shiny.video_ratio"),
    ("low_texture_ratio", "texture.video_ratio"),
    ("noise_ratio", "noise.video_ratio"),
    ("unique_views", "views.min_unique_views"),
    ("azimuth_span_deg", "coverage.min_azimuth_span_deg"),
)


@dataclass
class ReviewDecision:
    needs_review: bool
    reason: str

    def __iter__(self):  # so callers can write `flag, why = needs_review(row)`
        yield self.needs_review
        yield self.reason


def _threshold(cfg: LabelingConfig, dotted: str) -> float:
    node: object = cfg
    for part in dotted.split("."):
        node = getattr(node, part)
    return float(node)  # type: ignore[arg-type]


def _value(row, name: str) -> float:
    try:
        value = row[name]
    except (KeyError, IndexError, TypeError):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def needs_review(row, cfg: LabelingConfig | None = None) -> ReviewDecision:
    """Decide whether one capture row should go to a human, and say why."""
    cfg = cfg or labeling_config()
    margin = cfg.review.margin
    reasons: list[str] = []

    for stat, dotted in _BORDERLINE_CHECKS:
        value = _value(row, stat)
        if not np.isfinite(value):
            continue
        threshold = _threshold(cfg, dotted)
        if threshold == 0:
            continue
        window = abs(threshold) * margin
        if abs(value - threshold) <= window:
            direction = "just over" if value >= threshold else "just under"
            reasons.append(f"{stat} {value:.3g} is {direction} the {threshold:.3g} threshold")

    # --- checks that disagree with each other ------------------------------ #
    diversity = _value(row, "diversity")
    span = _value(row, "azimuth_span_deg")
    if (
        np.isfinite(diversity)
        and np.isfinite(span)
        and diversity > 0.35
        and span < cfg.coverage.min_azimuth_span_deg
    ):
        reasons.append(
            f"views look varied (diversity {diversity:.2f}) but the camera path only spans "
            f"{span:.0f}°: the camera path may be wrong"
        )

    unique_views = _value(row, "unique_views")
    if (
        np.isfinite(diversity)
        and diversity < cfg.views.min_diversity
        and np.isfinite(unique_views)
        and unique_views >= cfg.views.min_unique_views
    ):
        reasons.append(
            f"{int(unique_views)} views were counted but they all look alike "
            f"(diversity {diversity:.2f}): check for a repetitive or empty scene"
        )

    walked = _value(row, "walked_ratio")
    flow = _value(row, "flow_speed_mean")
    if np.isfinite(walked) and np.isfinite(flow) and walked < 0.2 and flow > 8.0:
        reasons.append(
            f"pairs say the camera only turned but the image moves fast "
            f"(flow {flow:.1f} px): check for a handheld pan"
        )

    overlap_mean = _value(row, "overlap_mean")
    if np.isfinite(overlap_mean) and overlap_mean < 0.1:
        reasons.append(
            f"mean neighbour overlap is only {overlap_mean:.2f}: feature matching may have failed"
        )

    if _value(row, "coverage_empty_bins") < 0 or not np.isfinite(_value(row, "azimuth_span_deg")):
        reasons.append("no camera path could be estimated")

    n_frames = _value(row, "n_frames")
    if np.isfinite(n_frames) and n_frames < 20:
        reasons.append(f"only {int(n_frames)} frames: too few to judge reliably")

    return ReviewDecision(bool(reasons), "; ".join(reasons))


def mark_reviews(rows: pd.DataFrame, cfg: LabelingConfig | None = None) -> pd.DataFrame:
    """Add ``review`` / ``review_reason`` columns to a table of capture rows."""
    cfg = cfg or labeling_config()
    flags, reasons = [], []
    for _, row in rows.iterrows():
        decision = needs_review(row, cfg)
        flags.append(int(decision.needs_review))
        reasons.append(decision.reason)
    out = rows.copy()
    out["review"] = flags
    out["review_reason"] = reasons
    return out
