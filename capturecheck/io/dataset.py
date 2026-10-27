"""Finding captures on disk and reading/writing the label CSV files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capturecheck.config import get_config
from capturecheck.io import images as images_io
from capturecheck.io.load import IMAGES, VIDEO, find_capture_file
from capturecheck.labeling import schema
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

FRAME_LABELS_FILE = "frame_labels.csv"
VIDEO_LABELS_FILE = "video_labels.csv"
REVIEW_QUEUE_FILE = "review_queue.csv"
HUMAN_LABELS_FILE = "human_labels.csv"
QUALITY_TARGETS_FILE = "quality_targets.csv"


@dataclass
class CaptureRef:
    """A capture found on disk, before it is loaded."""

    capture_id: str
    path: Path
    input_type: str
    scene_id: str
    source: str = "real"
    degrade_labels: dict | None = None
    degrade_type: str | None = None
    degrade_strength: str | None = None

    @property
    def is_degraded(self) -> bool:
        return self.source == "degraded"


def read_scene_id(folder: Path, default: str) -> str:
    """``scene.txt`` next to the capture, else the capture id."""
    scene_file = folder / "scene.txt"
    if scene_file.is_file():
        value = scene_file.read_text(encoding="utf-8").strip().splitlines()
        if value and value[0].strip():
            return value[0].strip()
    return default


def read_source(folder: Path, default: str = "real") -> str:
    """``source.txt`` lets you mark a capture ``company`` (test only) or ``degraded``."""
    source_file = folder / "source.txt"
    if source_file.is_file():
        value = source_file.read_text(encoding="utf-8").strip().lower()
        if value in schema.SOURCES:
            return value
    return default


def _degrade_meta(folder: Path) -> tuple[dict | None, str | None, str | None]:
    labels_file = folder / "labels.json"
    if not labels_file.is_file():
        return None, None, None
    try:
        payload = json.loads(labels_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("bad labels.json in %s", folder)
        return None, None, None
    return (
        payload,
        payload.get("degrade_type"),
        payload.get("degrade_strength"),
    )


def classify_path(path: Path) -> str | None:
    """``"video"`` / ``"images"`` for a capture folder or file, else ``None``."""
    if path.is_file():
        from capturecheck.io.video import is_video

        if is_video(path):
            return VIDEO
        if path.suffix.lower() == ".zip":
            return IMAGES
        return None
    if not path.is_dir():
        return None
    if find_capture_file(path) is not None:
        return VIDEO
    if images_io.is_image_bundle(path) or any(path.glob("*.zip")):
        return IMAGES
    return None


def list_captures(root: str | Path, source: str | None = None) -> list[CaptureRef]:
    """Every capture directly under ``root``.

    A capture is a folder (holding a video, an ``images/`` folder or a .zip), or a
    bare video / .zip file. ``scene_id`` comes from ``scene.txt`` when present.
    """
    root = Path(root)
    if not root.exists():
        return []

    # A single capture was passed instead of a parent folder.
    direct = classify_path(root)
    if root.is_file() and direct:
        return [
            CaptureRef(
                capture_id=root.stem,
                path=root,
                input_type=direct,
                scene_id=read_scene_id(root.parent, root.stem),
                source=source or "real",
            )
        ]

    out: list[CaptureRef] = []
    entries = sorted(root.iterdir()) if root.is_dir() else []
    for entry in entries:
        if entry.name.startswith(".") or entry.name in {"labels", "frames", "features", "splits"}:
            continue
        kind = classify_path(entry)
        if kind is None:
            continue
        capture_id = entry.name if entry.is_dir() else entry.stem
        folder = entry if entry.is_dir() else entry.parent
        payload, dtype, dstrength = _degrade_meta(folder) if entry.is_dir() else (None, None, None)
        detected_source = read_source(folder, "degraded" if payload else "real")
        out.append(
            CaptureRef(
                capture_id=capture_id,
                path=entry,
                input_type=kind,
                scene_id=read_scene_id(folder, capture_id),
                source=source or detected_source,
                degrade_labels=payload,
                degrade_type=dtype,
                degrade_strength=dstrength,
            )
        )

    if not out and direct and root.is_dir():
        # ``root`` is itself one capture.
        payload, dtype, dstrength = _degrade_meta(root)
        out.append(
            CaptureRef(
                capture_id=root.name,
                path=root,
                input_type=direct,
                scene_id=read_scene_id(root, root.name),
                source=source or read_source(root, "degraded" if payload else "real"),
                degrade_labels=payload,
                degrade_type=dtype,
                degrade_strength=dstrength,
            )
        )
    return out


def discover_all(
    raw_root: str | Path | None = None, degraded_root: str | Path | None = None
) -> list[CaptureRef]:
    """Captures in ``data/raw`` plus the broken copies in ``data/degraded``."""
    cfg = get_config()
    raw_root = Path(raw_root) if raw_root else cfg.paths.resolve("raw")
    degraded_root = Path(degraded_root) if degraded_root else cfg.paths.resolve("degraded")
    refs = list_captures(raw_root)
    refs += list_captures(degraded_root, source="degraded")
    return refs


# --------------------------------------------------------------------------- #
# Label CSV helpers
# --------------------------------------------------------------------------- #
def labels_dir(out_dir: str | Path | None = None) -> Path:
    cfg = get_config()
    path = Path(out_dir) if out_dir else cfg.paths.resolve("labels")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ordered(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Reindex to the canonical column order, keeping extra columns at the end."""
    extra = [c for c in df.columns if c not in columns]
    return df.reindex(columns=[*columns, *extra])


def write_labels(
    df: pd.DataFrame,
    name: str,
    out_dir: str | Path | None = None,
    columns: tuple[str, ...] | None = None,
    append: bool = False,
) -> Path:
    """Write one of the label CSVs, in the canonical column order."""
    target = labels_dir(out_dir) / name
    if columns is not None:
        df = _ordered(df, columns)
    if append and target.is_file():
        existing = pd.read_csv(target)
        df = pd.concat([existing, df], ignore_index=True)
        if columns is not None:
            df = _ordered(df, columns)
    df.to_csv(target, index=False)
    log.info("wrote %s (%d rows)", target, len(df))
    return target


def read_labels(
    name: str, out_dir: str | Path | None = None, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Read one of the label CSVs; an empty frame with the right columns if missing."""
    target = labels_dir(out_dir) / name
    if not target.is_file():
        return pd.DataFrame(columns=list(columns) if columns else [])
    df = pd.read_csv(target)
    return _ordered(df, columns) if columns else df


def write_frame_labels(df: pd.DataFrame, out_dir=None) -> Path:
    return write_labels(df, FRAME_LABELS_FILE, out_dir, schema.FRAME_CSV_COLUMNS)


def read_frame_labels(out_dir=None) -> pd.DataFrame:
    return read_labels(FRAME_LABELS_FILE, out_dir, schema.FRAME_CSV_COLUMNS)


def write_video_labels(df: pd.DataFrame, out_dir=None) -> Path:
    return write_labels(df, VIDEO_LABELS_FILE, out_dir, schema.VIDEO_CSV_COLUMNS)


def read_video_labels(out_dir=None) -> pd.DataFrame:
    return read_labels(VIDEO_LABELS_FILE, out_dir, schema.VIDEO_CSV_COLUMNS)


def write_review_queue(df: pd.DataFrame, out_dir=None) -> Path:
    return write_labels(df, REVIEW_QUEUE_FILE, out_dir, schema.REVIEW_QUEUE_COLUMNS)


def read_review_queue(out_dir=None) -> pd.DataFrame:
    return read_labels(REVIEW_QUEUE_FILE, out_dir, schema.REVIEW_QUEUE_COLUMNS)


def append_human_labels(rows: list[dict] | pd.DataFrame, out_dir=None) -> Path:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return write_labels(df, HUMAN_LABELS_FILE, out_dir, schema.HUMAN_CSV_COLUMNS, append=True)


def read_human_labels(out_dir=None) -> pd.DataFrame:
    return read_labels(HUMAN_LABELS_FILE, out_dir, schema.HUMAN_CSV_COLUMNS)


def read_quality_targets(out_dir=None) -> pd.DataFrame:
    """``capture_id,psnr`` for captures whose real 3DGS quality is known."""
    return read_labels(QUALITY_TARGETS_FILE, out_dir, schema.QUALITY_TARGET_COLUMNS)


def merge_quality_targets(video_df: pd.DataFrame, out_dir=None) -> pd.DataFrame:
    """Fill the ``psnr`` column of ``video_labels`` from ``quality_targets.csv``."""
    targets = read_quality_targets(out_dir)
    if targets.empty or "capture_id" not in targets.columns:
        return video_df
    mapping = dict(zip(targets["capture_id"], targets["psnr"], strict=False))
    video_df = video_df.copy()
    video_df["psnr"] = [
        mapping.get(cid, existing)
        for cid, existing in zip(
            video_df["capture_id"],
            video_df.get("psnr", pd.Series(dtype=float)),
            strict=False,
        )
    ]
    return video_df


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def splits_dir(path: str | Path | None = None) -> Path:
    cfg = get_config()
    out = Path(path) if path else cfg.paths.resolve("splits")
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_scene_list(name: str, path: str | Path | None = None) -> list[str]:
    target = splits_dir(path) / name
    if not target.is_file():
        return []
    lines = target.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def write_scene_list(scenes: list[str], name: str, path: str | Path | None = None) -> Path:
    target = splits_dir(path) / name
    target.write_text("\n".join(sorted(set(scenes))) + "\n", encoding="utf-8")
    log.info("wrote %s (%d scenes)", target, len(set(scenes)))
    return target


def features_path(capture_id: str, path: str | Path | None = None) -> Path:
    cfg = get_config()
    out = Path(path) if path else cfg.paths.resolve("features")
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{capture_id}.npz"
