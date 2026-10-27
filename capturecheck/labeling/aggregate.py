"""Frame labels -> one row of capture labels, with evidence.

Evidence is the part people actually act on, so each problem gets a short string
pointing at *where* it is: a time range and frame numbers for videos, the first
and last photo for bundles, a direction for coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capturecheck.checks.coverage import Coverage, summarize_empty
from capturecheck.checks.motion import TURNED, WALKED
from capturecheck.config import LabelingConfig, labeling_config
from capturecheck.labeling import schema


@dataclass
class VideoLabelResult:
    """The capture-level row, split into its three parts."""

    stats: dict[str, float] = field(default_factory=dict)
    labels: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)

    def as_row(self) -> dict:
        row: dict = {}
        row.update(self.stats)
        row.update(self.labels)
        row["evidence"] = json.dumps(self.evidence)
        return row


def _num(series: pd.Series | None) -> np.ndarray:
    if series is None:
        return np.zeros(0)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    return float(np.count_nonzero(mask) / mask.size)


def longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Start and end index (inclusive) of the longest ``True`` run, or ``None``."""
    if mask.size == 0 or not mask.any():
        return None
    best_len = best_start = 0
    current_start = -1
    for i, flag in enumerate(mask):
        if flag:
            if current_start < 0:
                current_start = i
            length = i - current_start + 1
            if length > best_len:
                best_len, best_start = length, current_start
        else:
            current_start = -1
    return best_start, best_start + best_len - 1


def describe_range(frame_df: pd.DataFrame, start: int, end: int, is_video: bool) -> str:
    """``"14.0-17.5 s (frames 28-35)"`` for videos, ``"IMG_0412 to IMG_0419"`` for photos."""
    names = frame_df["frame_name"].tolist() if "frame_name" in frame_df.columns else []
    idx = (
        frame_df["frame_idx"].tolist()
        if "frame_idx" in frame_df.columns
        else list(range(len(frame_df)))
    )

    if is_video and "time_s" in frame_df.columns:
        times = _num(frame_df["time_s"])
        if times.size > end and np.isfinite(times[start]) and np.isfinite(times[end]):
            if start == end:
                return f"{times[start]:.1f} s (frame {idx[start]})"
            return f"{times[start]:.1f}-{times[end]:.1f} s (frames {idx[start]}-{idx[end]})"

    if names:
        if start == end:
            return f"{names[start]}"
        return f"{names[start]} to {names[end]}"
    return f"frames {idx[start]}-{idx[end]}"


def _exif_blur_note(frame_df: pd.DataFrame, cfg: LabelingConfig) -> str:
    """"shutter 1/30 s" when EXIF shows exposures long enough to smear motion."""
    if "exposure_s" not in frame_df.columns:
        return ""
    exposures = _num(frame_df["exposure_s"])
    exposures = exposures[np.isfinite(exposures)]
    if exposures.size == 0:
        return ""
    slow = exposures[exposures > cfg.exif.slow_shutter_s]
    if slow.size == 0:
        return ""
    worst = float(slow.max())
    share = slow.size / exposures.size
    return f"shutter 1/{max(1, int(round(1.0 / worst)))} s on {share * 100:.0f}% of photos"


def _exif_noise_note(frame_df: pd.DataFrame, cfg: LabelingConfig) -> str:
    if "iso" not in frame_df.columns:
        return ""
    isos = _num(frame_df["iso"])
    isos = isos[np.isfinite(isos)]
    if isos.size == 0:
        return ""
    high = isos[isos > cfg.exif.high_iso]
    if high.size == 0:
        return ""
    return f"ISO up to {int(high.max())}"


def to_video_labels(
    frame_df: pd.DataFrame,
    pair_df: pd.DataFrame | None = None,
    cov: Coverage | None = None,
    info: dict | None = None,
    cfg: LabelingConfig | None = None,
    diversity: float | None = None,
    unique_views: int | None = None,
) -> VideoLabelResult:
    """Turn the per-frame table into one capture row: ratios, 0/1 labels, evidence."""
    cfg = cfg or labeling_config()
    info = info or {}
    is_video = info.get("input_type", "video") == "video"
    pair_df = pair_df if pair_df is not None else pd.DataFrame()

    sharpness = _num(frame_df.get("sharpness"))
    brightness = _num(frame_df.get("brightness"))
    noise = _num(frame_df.get("noise"))
    overlap = _num(frame_df.get("overlap_next"))
    overlap = overlap[np.isfinite(overlap)]

    blur_mask = (frame_df.get("label_blur") == "BLURRY").to_numpy()
    dark_mask = (frame_df.get("label_light") == "DARK").to_numpy()
    bright_mask = (frame_df.get("label_light") == "BRIGHT").to_numpy()
    dup_mask = (frame_df.get("label_view") == "DUPLICATE").to_numpy()
    shiny_mask = (_num(frame_df.get("label_shiny")) > 0.5)
    low_tex_mask = (frame_df.get("label_texture") == "LOW_TEXTURE").to_numpy()
    noise_mask = (_num(frame_df.get("label_noise")) > 0.5)

    kinds = pair_df["kind"].tolist() if "kind" in pair_df.columns else []
    decided = [k for k in kinds if k in (WALKED, TURNED)]
    walked_ratio = (
        float(sum(1 for k in decided if k == WALKED) / len(decided)) if decided else float("nan")
    )
    flow = _num(pair_df.get("flow_speed")) if "flow_speed" in pair_df.columns else np.zeros(0)

    n_unique = int(unique_views) if unique_views is not None else int(np.count_nonzero(~dup_mask))
    div = float(diversity) if diversity is not None else float("nan")

    stats = {
        "overlap_mean": _mean(overlap),
        "overlap_low_ratio": (
            _ratio(overlap < cfg.overlap.min_overlap) if overlap.size else float("nan")
        ),
        "blur_ratio": _ratio(blur_mask),
        "sharpness_mean": _mean(sharpness),
        "brightness_mean": _mean(brightness),
        "brightness_std": float(np.nanstd(brightness)) if brightness.size else float("nan"),
        "walked_ratio": walked_ratio,
        "flow_speed_mean": _mean(flow),
        "diversity": div,
        "coverage_empty_bins": int(cov.empty_bins) if cov is not None else -1,
        "azimuth_span_deg": float(cov.azimuth_span_deg) if cov is not None else float("nan"),
        "unique_views": n_unique,
        "duplicate_ratio": _ratio(dup_mask),
        "shiny_ratio": _ratio(shiny_mask),
        "low_texture_ratio": _ratio(low_tex_mask),
        "noise_mean": _mean(noise),
        "noise_ratio": _ratio(noise_mask),
        "clip_dark_ratio": _ratio(dark_mask),
        "clip_bright_ratio": _ratio(bright_mask),
    }

    labels = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
    evidence: dict[str, str] = {}

    # --- motion blur ------------------------------------------------------- #
    if stats["blur_ratio"] > cfg.blur.video_blur_ratio:
        labels["motion_blur"] = 1
        parts = [f"{stats['blur_ratio'] * 100:.0f}% of frames blurry"]
        run = longest_run(blur_mask)
        if run:
            parts.insert(0, describe_range(frame_df, run[0], run[1], is_video))
        note = _exif_blur_note(frame_df, cfg)
        if note:
            parts.append(note)
        evidence["motion_blur"] = ", ".join(parts)

    # --- lighting ---------------------------------------------------------- #
    bad_light_ratio = stats["clip_dark_ratio"] + stats["clip_bright_ratio"]
    if (
        stats["brightness_std"] > cfg.light.video_brightness_std
        or bad_light_ratio > cfg.light.video_bad_ratio
    ):
        labels["lighting"] = 1
        parts = []
        if stats["brightness_std"] > cfg.light.video_brightness_std:
            parts.append(f"brightness swings by {stats['brightness_std']:.0f} grey levels")
        if stats["clip_dark_ratio"] > 0:
            run = longest_run(dark_mask)
            where = describe_range(frame_df, run[0], run[1], is_video) if run else ""
            parts.append(f"{stats['clip_dark_ratio'] * 100:.0f}% too dark ({where})".strip())
        if stats["clip_bright_ratio"] > 0:
            run = longest_run(bright_mask)
            where = describe_range(frame_df, run[0], run[1], is_video) if run else ""
            parts.append(f"{stats['clip_bright_ratio'] * 100:.0f}% blown out ({where})".strip())
        evidence["lighting"] = "; ".join(parts)

    # --- no walking -------------------------------------------------------- #
    if np.isfinite(walked_ratio) and walked_ratio < cfg.motion.walked_ratio_min:
        labels["no_walking"] = 1
        evidence["no_walking"] = (
            f"only {walked_ratio * 100:.0f}% of frame pairs show real camera movement; "
            "the rest is rotation on the spot"
        )

    # --- poor coverage ----------------------------------------------------- #
    # Judged on how far round the subject the camera got, not on how many bins
    # are empty: the bin count depends on how the path drifted across elevation
    # rows, which punishes the long captures it should reward. The empty bins
    # still name *which* directions are missing.
    if (
        cov is not None
        and cov.reliable
        and cov.azimuth_span_deg < cfg.coverage.min_azimuth_span_deg
    ):
        labels["poor_coverage"] = 1
        parts = [
            f"the camera only got {cov.azimuth_span_deg:.0f}° round the subject "
            f"(want at least {cfg.coverage.min_azimuth_span_deg:.0f}°)"
        ]
        missing = summarize_empty(cov)
        if missing:
            parts.append(f"never filmed from: {missing}")
        evidence["poor_coverage"] = "; ".join(parts)
    elif cov is not None and not cov.reliable and labels["no_walking"]:
        # A pivot covers nothing, and the path estimate agrees it saw no motion.
        labels["poor_coverage"] = 1
        evidence["poor_coverage"] = "camera never moved, so only one side was captured"

    # --- too few views ----------------------------------------------------- #
    # Counted views only. Low DINOv2 diversity is reported as supporting evidence
    # and can send a capture to review, but it is not a label on its own: how
    # varied a scene looks depends on the room as much as on the filming.
    if n_unique < cfg.views.min_unique_views:
        labels["too_few_views"] = 1
        parts = [
            f"only {n_unique} genuinely different views "
            f"(want at least {cfg.views.min_unique_views})"
        ]
        if np.isfinite(div) and div < cfg.views.min_diversity:
            parts.append(f"view diversity {div:.2f}")
        if stats["duplicate_ratio"] > 0:
            parts.append(f"{stats['duplicate_ratio'] * 100:.0f}% near-duplicates")
        evidence["too_few_views"] = ", ".join(parts)

    # --- low overlap ------------------------------------------------------- #
    if (
        np.isfinite(stats["overlap_low_ratio"])
        and stats["overlap_low_ratio"] > cfg.overlap.low_overlap_ratio
    ):
        labels["low_overlap"] = 1
        parts = [f"{stats['overlap_low_ratio'] * 100:.0f}% of neighbouring pairs share too little"]
        gap = _worst_overlap_gap(frame_df, cfg, is_video)
        if gap:
            parts.insert(0, gap)
        evidence["low_overlap"] = ", ".join(parts)

    # --- mixed cameras (image bundles only) -------------------------------- #
    n_cameras = int(info.get("n_cameras") or 1)
    n_sizes = int(info.get("n_sizes") or 1)
    if not is_video and (n_cameras > cfg.exif.max_cameras or n_sizes > cfg.exif.max_sizes):
        labels["mixed_cameras"] = 1
        cameras = info.get("cameras") or []
        sizes = info.get("sizes") or []
        detail = []
        if n_cameras > cfg.exif.max_cameras:
            names = f" ({', '.join(map(str, cameras[:3]))})" if cameras else ""
            detail.append(f"{n_cameras} cameras{names}")
        if n_sizes > cfg.exif.max_sizes:
            shown = f" ({', '.join(map(str, sizes[:3]))})" if sizes else ""
            detail.append(f"{n_sizes} image sizes{shown}")
        evidence["mixed_cameras"] = "; ".join(detail)

    # --- reflective -------------------------------------------------------- #
    if np.isfinite(stats["shiny_ratio"]) and stats["shiny_ratio"] > cfg.shiny.video_ratio:
        labels["reflective"] = 1
        run = longest_run(shiny_mask)
        where = describe_range(frame_df, run[0], run[1], is_video) if run else ""
        evidence["reflective"] = (
            f"{stats['shiny_ratio'] * 100:.0f}% of frames show shiny or glass surfaces"
            + (f" ({where})" if where else "")
        )

    # --- low texture ------------------------------------------------------- #
    if stats["low_texture_ratio"] > cfg.texture.video_ratio:
        labels["low_texture"] = 1
        run = longest_run(low_tex_mask)
        where = describe_range(frame_df, run[0], run[1], is_video) if run else ""
        evidence["low_texture"] = (
            f"{stats['low_texture_ratio'] * 100:.0f}% of frames have under "
            f"{cfg.texture.min_features} features"
            + (f" ({where})" if where else "")
        )

    # --- noise ------------------------------------------------------------- #
    if stats["noise_ratio"] > cfg.noise.video_ratio:
        labels["noisy"] = 1
        parts = [f"{stats['noise_ratio'] * 100:.0f}% of frames noisy"]
        note = _exif_noise_note(frame_df, cfg)
        if note:
            parts.append(note)
        evidence["noisy"] = ", ".join(parts)

    return VideoLabelResult(stats=stats, labels=labels, evidence=evidence)


def _worst_overlap_gap(frame_df: pd.DataFrame, cfg: LabelingConfig, is_video: bool) -> str:
    """Point at the single worst gap between neighbours, e.g. between two photos."""
    if "overlap_next" not in frame_df.columns:
        return ""
    overlap = _num(frame_df["overlap_next"])
    finite = np.where(np.isfinite(overlap))[0]
    if finite.size == 0:
        return ""
    worst = int(finite[np.argmin(overlap[finite])])
    if overlap[worst] >= cfg.overlap.min_overlap:
        return ""
    names = frame_df["frame_name"].tolist() if "frame_name" in frame_df.columns else []
    if not is_video and worst + 1 < len(names):
        return (
            f"gap between {names[worst]} and {names[worst + 1]} "
            f"({overlap[worst] * 100:.0f}% shared)"
        )
    where = describe_range(frame_df, worst, min(worst + 1, len(frame_df) - 1), is_video)
    return f"worst gap at {where} ({overlap[worst] * 100:.0f}% shared)"


def summarize_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """Counts per capture label, for the CLI summary table."""
    data = []
    for name in schema.VIDEO_LABEL_NAMES:
        if name not in rows.columns:
            continue
        flagged = int(pd.to_numeric(rows[name], errors="coerce").fillna(0).sum())
        data.append(
            {
                "label": name,
                "captures": flagged,
                "share": round(flagged / len(rows), 3) if len(rows) else 0.0,
            }
        )
    return pd.DataFrame(data)
