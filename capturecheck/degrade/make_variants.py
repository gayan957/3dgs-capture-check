"""Apply every op to every good capture and write the broken copies out.

Copies are always written as an image bundle (an ``images/`` folder), whatever the
original was: that keeps one loader path for degraded data and avoids re-encoding
video. Each copy carries a ``scene.txt`` equal to the original's scene (so the
split never separates a capture from its own copies) and a ``labels.json`` with
the exact labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capturecheck.checks.coverage import rough_path
from capturecheck.checks.motion import motion_pairs, orb_frames
from capturecheck.config import Config, get_config
from capturecheck.degrade import ops as degrade_ops
from capturecheck.io.dataset import CaptureRef, list_captures
from capturecheck.io.load import load_capture, save_frames
from capturecheck.utils.logging import get_logger
from capturecheck.utils.seed import stable_seed

log = get_logger(__name__)


@dataclass
class Variant:
    """One broken copy that was written to disk."""

    capture_id: str
    scene_id: str
    source_id: str
    op: str
    strength: str
    path: Path
    n_frames: int
    labels: dict

    def as_dict(self) -> dict:
        return {
            "capture_id": self.capture_id,
            "scene_id": self.scene_id,
            "source_capture": self.source_id,
            "degrade_type": self.op,
            "degrade_strength": self.strength,
            "path": str(self.path),
            "n_frames": self.n_frames,
            **self.labels,
        }


def variant_id(capture_id: str, op: str, strength: str) -> str:
    return f"{capture_id}__{op}_{strength}"


def _needs_path(op: str) -> bool:
    return op == "drop_sector"


def _needs_pairs(op: str) -> bool:
    return op in {"drop_sector", "rotation_only"}


def make_variants_for_capture(
    ref: CaptureRef,
    out_root: str | Path | None = None,
    cfg: Config | None = None,
    ops: list[str] | None = None,
    strengths: list[str] | None = None,
    overwrite: bool = False,
) -> list[Variant]:
    """Write every configured (op, strength) copy of one capture."""
    cfg = cfg or get_config()
    out_root = Path(out_root) if out_root else cfg.paths.resolve("degraded")
    out_root.mkdir(parents=True, exist_ok=True)

    strengths = strengths or cfg.degrade.strengths
    wanted = ops or [name for name in degrade_ops.OPS if name in cfg.degrade.ops]
    if not wanted:
        wanted = list(degrade_ops.OPS)

    capture = load_capture(ref.path, capture_id=ref.capture_id)
    if capture.n_frames < 4:
        log.warning("%s has only %d frames; skipping", ref.capture_id, capture.n_frames)
        return []

    is_video = capture.is_video
    usable = [
        name
        for name in wanted
        if not (is_video and name in degrade_ops.IMAGES_ONLY_OPS)
    ]

    # Pose work is shared by every op that needs it, and only done when needed.
    pairs = None
    path = None
    if any(_needs_pairs(op) for op in usable):
        features = orb_frames(capture.frames)
        pairs = motion_pairs(
            capture.frames,
            features=features,
            gap=cfg.sampling.pair_gap_frames,
            focal_px=capture.focal_length_px,
            is_video=is_video,
        )
        if any(_needs_path(op) for op in usable):
            path = rough_path(pairs=pairs)

    written: list[Variant] = []
    for op in usable:
        for strength in strengths:
            params = cfg.degrade.params(op, strength)
            if not params:
                log.debug("no parameters for %s/%s; skipping", op, strength)
                continue

            vid = variant_id(ref.capture_id, op, strength)
            target = out_root / vid
            if target.exists() and not overwrite:
                log.info("%s already exists; skipping (use --overwrite to redo)", vid)
                continue

            # A fresh, deterministic seed per (capture, op, strength). It has to
            # be a digest, not hash(): hash() of a string is salted per process,
            # so the "same" broken copy came out different on every run.
            seed = stable_seed(ref.capture_id, op, strength) + cfg.degrade.seed
            rng = np.random.default_rng(seed)

            kwargs = dict(params)
            kwargs.update(names=capture.names, times=capture.times, rng=rng)
            if _needs_path(op):
                kwargs["path"] = path
            if op == "rotation_only":
                kwargs["pairs"] = pairs

            function = degrade_ops.OPS[op]
            result = function(capture.frames, **kwargs)
            if result.n < 3:
                log.warning("%s produced only %d frames; skipping", vid, result.n)
                continue
            if not degrade_ops.defect_created(result.labels):
                log.info(
                    "%s: %s could not create its defect here (%s); skipping rather than "
                    "writing a copy with a label that is not true of it",
                    vid, op, result.labels.get("reason", "no reason given"),
                )
                continue

            images_dir = target / "images"
            save_frames(
                result.frames,
                images_dir,
                names=[f"{i:04d}.jpg" for i in range(result.n)],
                jpeg_quality=cfg.degrade.jpeg_quality,
            )
            (target / "scene.txt").write_text(f"{ref.scene_id}\n", encoding="utf-8")
            (target / "source.txt").write_text("degraded\n", encoding="utf-8")

            labels = {
                **result.labels,
                "degrade_type": op,
                "degrade_strength": strength,
                "source_capture": ref.capture_id,
                "source_input_type": capture.input_type,
                "n_frames": result.n,
                "params": params,
            }
            (target / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

            log.info("wrote %s (%d frames)", vid, result.n)
            written.append(
                Variant(
                    capture_id=vid,
                    scene_id=ref.scene_id,
                    source_id=ref.capture_id,
                    op=op,
                    strength=strength,
                    path=target,
                    n_frames=result.n,
                    labels=result.labels,
                )
            )
    return written


def make_all(
    raw_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    cfg: Config | None = None,
    ops: list[str] | None = None,
    strengths: list[str] | None = None,
    overwrite: bool = False,
) -> list[Variant]:
    """Make broken copies of every real capture in ``raw_dir``."""
    cfg = cfg or get_config()
    raw_dir = Path(raw_dir) if raw_dir else cfg.paths.resolve("raw")
    out_dir = Path(out_dir) if out_dir else cfg.paths.resolve("degraded")

    refs = [r for r in list_captures(raw_dir) if r.source != "degraded"]
    if not refs:
        log.warning("no captures found in %s", raw_dir)
        return []

    # Company captures are the exam: never make training copies of them.
    company = [r for r in refs if r.source == "company"]
    if company:
        log.info("skipping %d company capture(s): they are test-only", len(company))
    refs = [r for r in refs if r.source != "company"]

    out: list[Variant] = []
    for ref in refs:
        out.extend(
            make_variants_for_capture(
                ref, out_root=out_dir, cfg=cfg, ops=ops,
                strengths=strengths, overwrite=overwrite,
            )
        )
    log.info("wrote %d broken copies from %d captures", len(out), len(refs))
    return out
