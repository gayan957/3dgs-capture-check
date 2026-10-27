"""Names and meanings of every label. Every other module imports them from here."""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Frame labels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelDef:
    """One label: the column it lives in, its possible values, and what it means."""

    name: str
    meaning: str
    values: tuple[str | int, ...] = ()
    bad_values: tuple[str | int, ...] = ()


FRAME_LABELS: tuple[LabelDef, ...] = (
    LabelDef(
        "label_blur",
        "Is the frame sharp enough to be used for reconstruction?",
        ("SHARP", "BLURRY"),
        ("BLURRY",),
    ),
    LabelDef(
        "label_light",
        "Is the frame correctly exposed?",
        ("OK_LIGHT", "DARK", "BRIGHT"),
        ("DARK", "BRIGHT"),
    ),
    LabelDef(
        "label_view",
        "Does the frame add a new viewpoint, or repeat the previous one?",
        ("UNIQUE_VIEW", "DUPLICATE"),
        ("DUPLICATE",),
    ),
    LabelDef(
        "label_shiny",
        "Does the frame show mostly shiny or glass surfaces?",
        (0, 1),
        (1,),
    ),
    LabelDef(
        "label_texture",
        "Does the frame have enough texture for feature matching?",
        ("OK_TEXTURE", "LOW_TEXTURE"),
        ("LOW_TEXTURE",),
    ),
    LabelDef(
        "label_noise",
        "Is the frame unusually noisy (high ISO, dark scene)?",
        (0, 1),
        (1,),
    ),
)

FRAME_LABEL_NAMES: tuple[str, ...] = tuple(d.name for d in FRAME_LABELS)

# --------------------------------------------------------------------------- #
# Capture labels (the problems the model predicts)
# --------------------------------------------------------------------------- #
VIDEO_LABELS: tuple[LabelDef, ...] = (
    LabelDef("motion_blur", "Too many blurry frames or photos.", (0, 1), (1,)),
    LabelDef(
        "lighting", "Brightness changes a lot, or frames are over/under exposed.", (0, 1), (1,)
    ),
    LabelDef(
        "no_walking", "The camera only turned on the spot instead of moving around.", (0, 1), (1,)
    ),
    LabelDef(
        "poor_coverage", "Whole directions around the subject were never filmed.", (0, 1), (1,)
    ),
    LabelDef("too_few_views", "Not enough genuinely different frames or photos.", (0, 1), (1,)),
    LabelDef("low_overlap", "Neighbouring frames share too little; there are gaps.", (0, 1), (1,)),
    LabelDef("mixed_cameras", "Photos come from several cameras or image sizes.", (0, 1), (1,)),
    LabelDef(
        "reflective", "Many shiny or glass surfaces, which 3DGS reconstructs badly.", (0, 1), (1,)
    ),
    LabelDef("low_texture", "Many plain surfaces, so features cannot be matched.", (0, 1), (1,)),
    LabelDef("noisy", "Heavy sensor noise (high ISO or a very dark scene).", (0, 1), (1,)),
)

VIDEO_LABEL_NAMES: tuple[str, ...] = tuple(d.name for d in VIDEO_LABELS)
#: Alias used by the model: these are the K problem heads, in a fixed order.
PROBLEM_LABELS: tuple[str, ...] = VIDEO_LABEL_NAMES
N_PROBLEMS: int = len(PROBLEM_LABELS)

#: Short names shown in reports.
LABEL_TITLES: dict[str, str] = {
    "motion_blur": "Motion blur",
    "lighting": "Lighting",
    "no_walking": "No walking",
    "poor_coverage": "Poor coverage",
    "too_few_views": "Too few views",
    "low_overlap": "Low overlap",
    "mixed_cameras": "Mixed cameras",
    "reflective": "Reflective",
    "low_texture": "Low texture",
    "noisy": "Noise",
}

#: Problems that only make sense for photo bundles.
IMAGES_ONLY_LABELS: tuple[str, ...] = ("mixed_cameras",)

# --------------------------------------------------------------------------- #
# CSV layouts (section 3 of the build guide)
# --------------------------------------------------------------------------- #
FRAME_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "overlap_next",
    "sharpness",
    "brightness",
    "clip_dark",
    "clip_bright",
    "noise",
    "features",
    "dup_of_prev",
    "shiny_prob",
)

FRAME_CSV_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "frame_idx",
    "frame_name",
    "time_s",
    "exposure_s",
    "iso",
    *FRAME_MEASUREMENT_COLUMNS,
    *FRAME_LABEL_NAMES,
)

VIDEO_STAT_COLUMNS: tuple[str, ...] = (
    "overlap_mean",
    "overlap_low_ratio",
    "blur_ratio",
    "sharpness_mean",
    "brightness_mean",
    "brightness_std",
    "walked_ratio",
    "flow_speed_mean",
    "diversity",
    "coverage_empty_bins",
    "unique_views",
    "duplicate_ratio",
    "shiny_ratio",
    "low_texture_ratio",
    "noise_mean",
    "noise_ratio",
    "clip_dark_ratio",
    "clip_bright_ratio",
)

VIDEO_CSV_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "scene_id",
    "input_type",
    "source",
    "width",
    "height",
    "fps",
    "duration_s",
    "n_frames",
    "n_cameras",
    "n_sizes",
    *VIDEO_STAT_COLUMNS,
    *VIDEO_LABEL_NAMES,
    "evidence",
    "review",
    "review_reason",
    "label_source",
    "psnr",
    "degrade_type",
    "degrade_strength",
)

HUMAN_CSV_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "scene_id",
    "input_type",
    *VIDEO_LABEL_NAMES,
    "note",
    "reviewer",
    "timestamp",
    "label_source",
)

REVIEW_QUEUE_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "scene_id",
    "input_type",
    "review_reason",
    "n_frames",
    *VIDEO_LABEL_NAMES,
)

QUALITY_TARGET_COLUMNS: tuple[str, ...] = ("capture_id", "psnr", "ssim", "notes")

#: Valid values of ``source`` / ``label_source``.
SOURCES: tuple[str, ...] = ("real", "degraded", "company")
LABEL_SOURCES: tuple[str, ...] = ("auto", "degrade_script", "human")


def frame_label_def(name: str) -> LabelDef:
    for d in FRAME_LABELS:
        if d.name == name:
            return d
    raise KeyError(name)


def video_label_def(name: str) -> LabelDef:
    for d in VIDEO_LABELS:
        if d.name == name:
            return d
    raise KeyError(name)


def describe() -> str:
    """A printable summary of every label (used by ``capturecheck labels``)."""
    lines = ["Frame labels:"]
    for d in FRAME_LABELS:
        lines.append(f"  {d.name:<16} {'/'.join(str(v) for v in d.values):<28} {d.meaning}")
    lines.append("")
    lines.append("Capture labels (0/1):")
    for d in VIDEO_LABELS:
        lines.append(f"  {d.name:<16} {d.meaning}")
    return "\n".join(lines)
