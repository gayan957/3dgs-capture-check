"""Run every check on one capture and return its label rows.

This is the single path used by ``capturecheck label``, ``capturecheck check``,
the review app and the model's feature builder, so a capture is always measured
the same way. Each step is timed against the 60 s budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from capturecheck.checks import quality as quality_checks
from capturecheck.checks.coverage import CameraPath, Coverage, coverage_map, rough_path
from capturecheck.checks.motion import (
    OrbFrame,
    approx_intrinsics,
    flow_speed,
    motion_pairs,
    orb_frames,
    overlap_series,
)
from capturecheck.config import Config, get_config
from capturecheck.features import similarity
from capturecheck.features.dino import EMBED_DIM, DinoEmbedder, evenly_spaced
from capturecheck.io.load import Capture, load_capture
from capturecheck.labeling import classifiers
from capturecheck.labeling.aggregate import VideoLabelResult, to_video_labels
from capturecheck.labeling.confidence import needs_review
from capturecheck.labeling.frame_rules import label_frames
from capturecheck.utils.logging import get_logger
from capturecheck.utils.timing import Timings

log = get_logger(__name__)


@dataclass
class CaptureMeasurements:
    """Everything measured for one capture: the inputs to every label and the model."""

    capture: Capture
    frame_df: pd.DataFrame
    pair_df: pd.DataFrame
    coverage: Coverage | None
    path: CameraPath | None
    emb: np.ndarray = field(default_factory=lambda: np.zeros((0, EMBED_DIM), dtype=np.float32))
    emb_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    diversity: float = float("nan")
    unique_views: int = 0
    timings: Timings = field(default_factory=Timings)

    @property
    def info(self) -> dict:
        return self.capture.info


@dataclass
class LabelResult:
    """The output of :func:`label_capture`."""

    capture_id: str
    frame_rows: pd.DataFrame
    video_row: dict
    measurements: CaptureMeasurements | None = None

    @property
    def evidence(self) -> dict:
        import json

        raw = self.video_row.get("evidence") or "{}"
        return json.loads(raw) if isinstance(raw, str) else dict(raw)

    @property
    def problems(self) -> list[str]:
        from capturecheck.labeling import schema

        return [n for n in schema.VIDEO_LABEL_NAMES if int(self.video_row.get(n, 0) or 0) == 1]


def measure_capture(
    capture: Capture,
    cfg: Config | None = None,
    embedder: DinoEmbedder | None = None,
    use_dino: bool = True,
    use_cache: bool = True,
    timings: Timings | None = None,
) -> CaptureMeasurements:
    """Run the OpenCV checks, the DINOv2 embedding and the coverage estimate."""
    cfg = cfg or get_config()
    lab = cfg.labeling
    timings = timings or Timings()
    frames = capture.frames
    n = len(frames)

    # --- per-frame quality ------------------------------------------------- #
    with timings.step("quality"):
        measures = quality_checks.measure_many(frames)

    # --- ORB once, reused by texture, overlap and motion ------------------- #
    with timings.step("orb"):
        features: list[OrbFrame] = orb_frames(frames)
        feature_counts = [f.n for f in features]

    with timings.step("overlap"):
        overlaps = overlap_series(features)

    # --- DINOv2 ------------------------------------------------------------ #
    emb = np.zeros((0, EMBED_DIM), dtype=np.float32)
    emb_idx = np.zeros(0, dtype=int)
    if use_dino and n:
        with timings.step("dino"):
            embedder = embedder or DinoEmbedder()
            try:
                emb, emb_idx = embedder.embed_capture(
                    capture.capture_id, frames, use_cache=use_cache
                )
            except Exception as exc:  # offline, or no weights available
                log.warning("DINOv2 unavailable (%s); continuing without it", exc)
                emb = np.zeros((0, EMBED_DIM), dtype=np.float32)
                emb_idx = np.zeros(0, dtype=int)
    if emb.shape[0] == 0:
        emb_idx = evenly_spaced(n, lab.sampling.dino_max_frames)

    # --- similarity, duplicates, shininess --------------------------------- #
    with timings.step("similarity"):
        dup_of_prev = np.full(n, np.nan, dtype=float)
        shiny_prob = np.full(n, np.nan, dtype=float)
        diversity = float("nan")
        n_unique = 0
        order_pairs: list[tuple[int, int]] | None = None
        if emb.shape[0] > 1:
            sims = similarity.similarity_to_previous(emb)
            for slot, frame_idx in enumerate(emb_idx):
                dup_of_prev[frame_idx] = sims[slot]
            diversity = similarity.diversity(emb)
            n_unique = int(len(similarity.unique_views(emb, lab.views.duplicate_cosine)))
            probs = classifiers.predict_shiny(emb)
            for slot, frame_idx in enumerate(emb_idx):
                shiny_prob[frame_idx] = probs[slot]
            if not capture.has_order:
                # No capture order: pair each photo with the one it looks like most.
                pairs = similarity.nearest_neighbour_pairs(emb)
                order_pairs = [(int(emb_idx[i]), int(emb_idx[j])) for i, j in pairs]

    # --- motion: walked or turned, and flow speed for videos --------------- #
    with timings.step("motion"):
        pairs = motion_pairs(
            frames,
            features=features,
            gap=lab.sampling.pair_gap_frames,
            focal_px=capture.focal_length_px,
            is_video=capture.is_video,
            order=order_pairs,
        )
        pair_rows = []
        for i, j, result in pairs:
            row = {
                "i": i,
                "j": j,
                "kind": result.kind,
                "confidence": result.confidence,
                "n_matches": result.n_matches,
                "inliers_h": result.inliers_h,
                "inliers_e": result.inliers_e,
                "brightness_delta": abs(
                    measures[i].brightness - measures[j].brightness
                ),
                "flow_speed": np.nan,
            }
            pair_rows.append(row)
        pair_df = pd.DataFrame(pair_rows)

    if capture.is_video and len(frames) > 1:
        with timings.step("flow"):
            # Flow is only meaningful when the frames are a fixed time apart.
            speeds = [flow_speed(frames[i], frames[j]) for i, j, _ in pairs]
            if not pair_df.empty:
                pair_df["flow_speed"] = speeds

    # --- coverage ---------------------------------------------------------- #
    with timings.step("coverage"):
        k = approx_intrinsics(features[0].shape, capture.focal_length_px) if features else None
        path = rough_path(pairs=pairs) if pairs else None
        cov = coverage_map(path) if path is not None else None
        if cov is not None and not cov.reliable:
            log.debug("camera path unreliable for %s", capture.capture_id)
        del k

    # --- assemble the frame table ------------------------------------------ #
    exposures = [e.exposure_s for e in capture.exif] if capture.exif else []
    isos = [e.iso for e in capture.exif] if capture.exif else []
    frame_df = pd.DataFrame(
        {
            "capture_id": capture.capture_id,
            "frame_idx": np.arange(n),
            "frame_name": [capture.label_for(i) for i in range(n)],
            "time_s": [
                capture.times[i] if i < len(capture.times) else None for i in range(n)
            ],
            "exposure_s": [exposures[i] if i < len(exposures) else None for i in range(n)],
            "iso": [isos[i] if i < len(isos) else None for i in range(n)],
            "overlap_next": [*overlaps, np.nan] if n else [],
            "sharpness": [m.sharpness for m in measures],
            "brightness": [m.brightness for m in measures],
            "contrast": [m.contrast for m in measures],
            "clip_dark": [m.clip_dark for m in measures],
            "clip_bright": [m.clip_bright for m in measures],
            "noise": [m.noise for m in measures],
            "features": feature_counts,
            "dup_of_prev": dup_of_prev,
            "shiny_prob": shiny_prob,
        }
    )
    frame_df = label_frames(frame_df, lab)

    return CaptureMeasurements(
        capture=capture,
        frame_df=frame_df,
        pair_df=pair_df,
        coverage=cov,
        path=path,
        emb=emb,
        emb_idx=np.asarray(emb_idx, dtype=int),
        diversity=diversity,
        unique_views=n_unique,
        timings=timings,
    )


def build_video_row(
    measurements: CaptureMeasurements,
    scene_id: str | None = None,
    source: str = "real",
    label_source: str = "auto",
    cfg: Config | None = None,
    degrade_labels: dict | None = None,
    degrade_type: str | None = None,
    degrade_strength: str | None = None,
) -> dict:
    """Make the ``video_labels.csv`` row for one measured capture."""
    cfg = cfg or get_config()
    capture = measurements.capture
    info = capture.info

    result: VideoLabelResult = to_video_labels(
        measurements.frame_df,
        measurements.pair_df,
        measurements.coverage,
        info=info,
        cfg=cfg.labeling,
        diversity=measurements.diversity,
        unique_views=measurements.unique_views or None,
    )

    row: dict = {
        "capture_id": capture.capture_id,
        "scene_id": scene_id or capture.capture_id,
        "input_type": capture.input_type,
        "source": source,
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "duration_s": info.get("duration_s"),
        "n_frames": capture.n_frames,
        "n_cameras": info.get("n_cameras", 1),
        "n_sizes": info.get("n_sizes", 1),
    }
    row.update(result.as_row())

    if degrade_labels:
        # Broken copies carry exact labels: trust them over the rules.
        from capturecheck.labeling import schema

        for name in schema.VIDEO_LABEL_NAMES:
            if name in degrade_labels:
                row[name] = int(degrade_labels[name])
        label_source = "degrade_script"
    row["label_source"] = label_source
    row["degrade_type"] = degrade_type
    row["degrade_strength"] = degrade_strength
    row["psnr"] = None

    decision = needs_review(row, cfg.labeling)
    row["review"] = int(decision.needs_review)
    row["review_reason"] = decision.reason
    return row


def label_capture(
    path: str | Path,
    scene_id: str | None = None,
    source: str = "real",
    capture_id: str | None = None,
    cfg: Config | None = None,
    embedder: DinoEmbedder | None = None,
    use_dino: bool = True,
    use_cache: bool = True,
    degrade_labels: dict | None = None,
    degrade_type: str | None = None,
    degrade_strength: str | None = None,
    keep_measurements: bool = False,
) -> LabelResult:
    """Label one capture: a video file, a folder of photos, or a .zip."""
    cfg = cfg or get_config()
    timings = Timings()

    with timings.step("load"):
        capture = load_capture(path, capture_id=capture_id)

    measurements = measure_capture(
        capture, cfg=cfg, embedder=embedder, use_dino=use_dino,
        use_cache=use_cache, timings=timings,
    )
    video_row = build_video_row(
        measurements,
        scene_id=scene_id,
        source=source,
        cfg=cfg,
        degrade_labels=degrade_labels,
        degrade_type=degrade_type,
        degrade_strength=degrade_strength,
    )
    video_row["time_s_total"] = round(timings.total, 2)

    budget = cfg.runtime.time_budget_s
    if timings.total > budget:
        log.warning(
            "%s took %.1f s, over the %.0f s budget:\n%s",
            capture.capture_id, timings.total, budget, timings.report(),
        )
    else:
        log.info("%s measured in %.1f s", capture.capture_id, timings.total)

    frame_rows = measurements.frame_df
    return LabelResult(
        capture_id=capture.capture_id,
        frame_rows=frame_rows,
        video_row=video_row,
        measurements=measurements if keep_measurements else None,
    )
