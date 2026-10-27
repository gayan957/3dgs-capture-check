"""Video input: file information from ffprobe, and 2 fps frame sampling."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from capturecheck.config import get_config
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class VideoInfo:
    """What we can learn about a video file without decoding all of it."""

    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    codec: str | None = None
    n_frames_total: int | None = None
    rotation: int = 0
    probe_source: str = "ffprobe"

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4) if self.fps else self.fps,
            "duration_s": round(self.duration_s, 3) if self.duration_s else self.duration_s,
            "codec": self.codec,
            "n_frames_total": self.n_frames_total,
            "rotation": self.rotation,
            "probe_source": self.probe_source,
        }


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def _parse_fraction(value: str | None) -> float:
    """``"30000/1001"`` -> 29.97. Returns 0.0 when unparsable."""
    if not value:
        return 0.0
    try:
        if "/" in str(value):
            num, den = str(value).split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _probe_with_ffprobe(path: Path) -> VideoInfo | None:
    exe = _ffprobe_path()
    if exe is None:
        return None

    cmd = [
        exe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        payload = json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        log.warning("ffprobe failed on %s (%s); falling back to OpenCV", path.name, exc)
        return None

    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        return None
    stream = streams[0]

    fps = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(
        stream.get("r_frame_rate")
    )
    duration = 0.0
    for candidate in (stream.get("duration"), payload.get("format", {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    n_frames = None
    try:
        n_frames = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        if fps and duration:
            n_frames = int(round(fps * duration))

    rotation = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rotation = int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                rotation = 0
    tag_rotate = (stream.get("tags") or {}).get("rotate")
    if not rotation and tag_rotate:
        try:
            rotation = int(round(float(tag_rotate))) % 360
        except (TypeError, ValueError):
            rotation = 0

    return VideoInfo(
        path=str(path),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=float(fps),
        duration_s=float(duration),
        codec=stream.get("codec_name"),
        n_frames_total=n_frames,
        rotation=rotation,
        probe_source="ffprobe",
    )


def _probe_with_opencv(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    duration = (n_frames / fps) if (fps > 0 and n_frames > 0) else 0.0
    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        duration_s=duration,
        codec=None,
        n_frames_total=n_frames or None,
        rotation=0,
        probe_source="opencv",
    )


def probe(path: str | Path) -> VideoInfo:
    """Width, height, fps, duration and codec. Uses ffprobe, falls back to OpenCV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    info = _probe_with_ffprobe(path)
    if info is None or not info.fps or not info.width:
        fallback = _probe_with_opencv(path)
        if info is not None:
            # Keep whatever ffprobe did tell us.
            info.fps = info.fps or fallback.fps
            info.width = info.width or fallback.width
            info.height = info.height or fallback.height
            info.duration_s = info.duration_s or fallback.duration_s
            info.n_frames_total = info.n_frames_total or fallback.n_frames_total
            return info
        return fallback
    return info


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Shrink to ``width``, keeping the aspect ratio.

    Narrower frames are left alone: upscaling costs time and invents no detail.
    What matters for the thresholds is that every frame of a capture is measured
    at the same width, which this guarantees either way. The width actually used
    is recorded as ``measured_width`` so thresholds can be recalibrated for
    captures that came in below ``resize_width``.
    """
    h, w = frame.shape[:2]
    if width <= 0 or w <= width:
        return frame
    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def _target_times(duration_s: float, fps: float, max_frames: int) -> np.ndarray:
    """Timestamps (seconds) to sample: ``fps`` per second, capped and spread evenly."""
    if duration_s <= 0:
        return np.zeros(0, dtype=np.float64)

    n_wanted = max(1, int(np.floor(duration_s * fps)))
    if n_wanted <= max_frames:
        # Half-step offset keeps us away from the very first/last frame.
        return (np.arange(n_wanted, dtype=np.float64) + 0.5) / fps
    return np.linspace(0.0, duration_s, num=max_frames, endpoint=False) + (
        duration_s / max_frames / 2.0
    )


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation in (90, -270):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation in (180, -180):
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation in (270, -90):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def sample_frames(
    path: str | Path,
    fps: float | None = None,
    max_frames: int | None = None,
    width: int | None = None,
    info: VideoInfo | None = None,
    strategy: str = "auto",
) -> tuple[list[np.ndarray], list[float], VideoInfo]:
    """Take ``fps`` frames per second of video, resized to ``width``.

    Longer videos that would exceed ``max_frames`` get ``max_frames`` spread evenly
    over the whole duration instead.

    Returns ``(frames, timestamps_s, info)``. Frames are BGR uint8 arrays.
    """
    cfg = get_config()
    fps = cfg.sampling.fps if fps is None else float(fps)
    max_frames = cfg.sampling.max_frames if max_frames is None else int(max_frames)
    width = cfg.resize_width if width is None else int(width)

    path = Path(path)
    info = info or probe(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open video: {path}")
    # Let OpenCV apply the phone's rotation tag when it knows how.
    auto_orientation = False
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        auto_orientation = bool(cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1))

    try:
        src_fps = info.fps or float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = info.duration_s
        if duration <= 0 and src_fps > 0 and info.n_frames_total:
            duration = info.n_frames_total / src_fps

        times = _target_times(duration, fps, max_frames)
        if times.size == 0 or src_fps <= 0:
            frames, stamps = _read_all_decimated(cap, fps, max_frames, width)
        else:
            step = max(1, int(round(src_fps / fps))) if n_steps_known(times) else 1
            use_seek = strategy == "seek" or (strategy == "auto" and step > 30)
            if use_seek:
                frames, stamps = _read_by_seek(cap, times, width)
            else:
                frames, stamps = _read_by_grab(cap, times, src_fps, width)
    finally:
        cap.release()

    rotation = 0 if auto_orientation else info.rotation
    if rotation:
        frames = [_apply_rotation(f, rotation) for f in frames]
        frames = [resize_to_width(f, width) for f in frames]

    if frames:
        h, w = frames[0].shape[:2]
        log.info(
            "%s: %d frames at %g fps (source %dx%d @ %.2f fps, %.1f s) -> %dx%d",
            path.name, len(frames), fps, info.width, info.height, info.fps,
            info.duration_s, w, h,
        )
    return frames, stamps, info


def n_steps_known(times: np.ndarray) -> bool:
    """True when the sampled timestamps are evenly spaced (the usual case)."""
    if times.size < 3:
        return True
    diffs = np.diff(times)
    return bool(np.allclose(diffs, diffs[0], rtol=1e-3, atol=1e-6))


def _read_by_grab(
    cap: cv2.VideoCapture, times: np.ndarray, src_fps: float, width: int
) -> tuple[list[np.ndarray], list[float]]:
    """Walk the video once, decoding only the frames we keep (``grab`` skips cheaply)."""
    wanted = np.unique(np.round(times * src_fps).astype(np.int64))
    wanted_set = set(int(i) for i in wanted)
    last_wanted = int(wanted[-1]) if wanted.size else -1

    frames: list[np.ndarray] = []
    stamps: list[float] = []
    idx = 0
    while idx <= last_wanted:
        if not cap.grab():
            break
        if idx in wanted_set:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                frames.append(resize_to_width(frame, width))
                stamps.append(float(idx / src_fps))
        idx += 1
    return frames, stamps


def _read_by_seek(
    cap: cv2.VideoCapture, times: np.ndarray, width: int
) -> tuple[list[np.ndarray], list[float]]:
    """Jump straight to each timestamp. Faster for long videos, needs seekable files."""
    frames: list[np.ndarray] = []
    stamps: list[float] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frames.append(resize_to_width(frame, width))
        stamps.append(float(pos_ms / 1000.0) if pos_ms and pos_ms > 0 else float(t))
    return frames, stamps


def _read_all_decimated(
    cap: cv2.VideoCapture, fps: float, max_frames: int, width: int
) -> tuple[list[np.ndarray], list[float]]:
    """Last resort: the file has no usable fps/duration, so read it and decimate."""
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    step = max(1, int(round(src_fps / fps))) if src_fps > 0 else 1

    raw: list[np.ndarray] = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                raw.append(frame)
        idx += 1

    if len(raw) > max_frames:
        keep = np.linspace(0, len(raw) - 1, num=max_frames).round().astype(int)
        raw = [raw[i] for i in keep]
        stamps = [float(i * step / src_fps) if src_fps > 0 else float(i) for i in keep]
    else:
        stamps = [
            float(i * step / src_fps) if src_fps > 0 else float(i) for i in range(len(raw))
        ]
    return [resize_to_width(f, width) for f in raw], stamps
