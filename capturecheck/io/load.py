"""One entry point for every kind of capture: ``load_capture``.

Everything downstream works on the result of this module, so the rest of the
pipeline never needs to know whether the capture was a video or a pile of photos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from capturecheck.config import get_config
from capturecheck.io import images as images_io
from capturecheck.io import video as video_io
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

VIDEO = "video"
IMAGES = "images"


@dataclass
class Capture:
    """A loaded capture, ready to be measured."""

    capture_id: str
    path: str
    input_type: str  # "video" | "images"
    frames: list[np.ndarray] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    times: list[float | None] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    exif: list[images_io.ExifInfo] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def is_video(self) -> bool:
        return self.input_type == VIDEO

    @property
    def has_order(self) -> bool:
        """False only for photo bundles with no EXIF time and no usable file names."""
        if self.is_video:
            return True
        return self.info.get("order_source", "name") != "unknown"

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the frames as measured, not of the source file."""
        if not self.frames:
            return (0, 0)
        h, w = self.frames[0].shape[:2]
        return (int(h), int(w))

    @property
    def focal_length_px(self) -> float | None:
        """Focal length in pixels at the measured resolution, from EXIF if known."""
        h, w = self.shape
        if not w:
            return None
        f35 = self.info.get("focal_length_35mm")
        f_mm = self.info.get("focal_length_mm")
        sensor_w_mm = 36.0  # 35 mm equivalent
        if f35:
            return float(f35) * w / sensor_w_mm
        if f_mm and self.info.get("focal_length_is_35mm_equiv"):
            return float(f_mm) * w / sensor_w_mm
        return None

    def label_for(self, idx: int) -> str:
        """Human-readable name of one frame: a photo file name, or ``frame_012``."""
        if 0 <= idx < len(self.names):
            return self.names[idx]
        return f"frame_{idx:03d}"


def detect_input_type(path: str | Path) -> str:
    """``"video"`` or ``"images"``; raises when the path is neither."""
    path = Path(path)
    if path.is_file():
        if video_io.is_video(path):
            return VIDEO
        if path.suffix.lower() == ".zip":
            return IMAGES
        if path.suffix.lower() in {e.lower() for e in get_config().labeling.images.extensions}:
            return IMAGES
        raise ValueError(f"unsupported file type: {path}")
    if path.is_dir():
        if find_capture_file(path) is not None:
            return VIDEO
        if images_io.is_image_bundle(path):
            return IMAGES
        zips = sorted(path.glob("*.zip"))
        if zips:
            return IMAGES
        raise ValueError(f"no video and no photos found in {path}")
    raise FileNotFoundError(path)


def find_capture_file(folder: str | Path) -> Path | None:
    """The video inside a ``data/raw/<capture_id>/`` folder, if there is one."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    candidates = [
        p for p in sorted(folder.iterdir()) if p.is_file() and video_io.is_video(p)
    ]
    if not candidates:
        return None
    # "video.mp4" wins if present, otherwise the largest file.
    for p in candidates:
        if p.stem.lower() == "video":
            return p
    return max(candidates, key=lambda p: p.stat().st_size)


def capture_id_for(path: str | Path) -> str:
    """Capture id from a path: the folder name, or the file name without extension."""
    path = Path(path)
    if path.is_dir():
        return path.name
    if video_io.is_video(path) and path.stem.lower() == "video":
        return path.parent.name
    return path.stem


def load_capture(
    path: str | Path,
    fps: float | None = None,
    max_frames: int | None = None,
    width: int | None = None,
    capture_id: str | None = None,
) -> Capture:
    """Load a video, a folder of photos or a .zip of photos into a :class:`Capture`."""
    cfg = get_config()
    path = Path(path)
    input_type = detect_input_type(path)
    cid = capture_id or capture_id_for(path)

    if input_type == VIDEO:
        video_path = path if path.is_file() else find_capture_file(path)
        frames, times, vinfo = video_io.sample_frames(
            video_path, fps=fps, max_frames=max_frames, width=width
        )
        info = vinfo.as_dict()
        measured_w = int(frames[0].shape[1]) if frames else 0
        info.update(
            {
                "input_type": VIDEO,
                "n_frames": len(frames),
                "sample_fps": cfg.sampling.fps if fps is None else float(fps),
                "measured_width": measured_w,
                "n_cameras": 1,
                "n_sizes": 1,
                "order_source": "video",
            }
        )
        names = [f"frame_{i:03d}" for i in range(len(frames))]
        return Capture(
            capture_id=cid,
            path=str(video_path),
            input_type=VIDEO,
            frames=frames,
            names=names,
            times=[float(t) for t in times],
            info=info,
        )

    bundle_path = path
    if path.is_dir() and not images_io.is_image_bundle(path):
        zips = sorted(path.glob("*.zip"))
        if zips:
            bundle_path = zips[0]

    read = images_io.read_bundle(bundle_path, max_frames=max_frames, width=width)
    h, w = read.frames[0].shape[:2]
    src_w, src_h = (read.exif[0].width, read.exif[0].height) if read.exif else (w, h)
    info = read.info.as_dict()
    info.update(
        {
            "input_type": IMAGES,
            "path": str(bundle_path),
            "n_frames": len(read.frames),
            "width": int(src_w or w),
            "height": int(src_h or h),
            "measured_width": int(w),
            "fps": None,
            "duration_s": None,
            "codec": None,
            "focal_length_35mm": (
                read.exif[0].focal_length_35mm if read.exif else None
            ),
        }
    )
    return Capture(
        capture_id=cid,
        path=str(bundle_path),
        input_type=IMAGES,
        frames=read.frames,
        names=read.names,
        times=read.times,
        info=info,
        exif=read.exif,
    )


def save_frames(
    frames: list[np.ndarray],
    out_dir: str | Path,
    names: list[str] | None = None,
    jpeg_quality: int = 92,
) -> list[Path]:
    """Write frames as ``000.jpg``, ``001.jpg`` ... (or under the given names)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, frame in enumerate(frames):
        if names is not None and i < len(names):
            name = Path(names[i]).name
            if not Path(name).suffix:
                name = f"{name}.jpg"
        else:
            name = f"{i:03d}.jpg"
        target = out_dir / name
        ok = cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:  # pragma: no cover - only on a full or read-only disk
            raise OSError(f"could not write {target}")
        written.append(target)
    return written
