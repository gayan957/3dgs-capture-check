"""Shared fixtures: a tiny synthetic video and a tiny image bundle.

Everything is generated with OpenCV/Pillow so the tests need no sample data and
run offline.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import ExifTags, Image


# --------------------------------------------------------------------------- #
# Image generators
# --------------------------------------------------------------------------- #
def checkerboard(width: int = 320, height: int = 240, square: int = 20) -> np.ndarray:
    """A sharp, high-contrast board: plenty of corners for ORB."""
    ys, xs = np.mgrid[0:height, 0:width]
    board = (((xs // square) + (ys // square)) % 2) * 255
    img = np.dstack([board, board, board]).astype(np.uint8)
    return img


def textured_scene(width: int = 320, height: int = 240, seed: int = 0) -> np.ndarray:
    """A checkerboard with coloured blobs and noise: realistic feature density."""
    rng = np.random.default_rng(seed)
    img = checkerboard(width, height, square=24)
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    for _ in range(40):
        cx, cy = rng.integers(0, width), rng.integers(0, height)
        r = int(rng.integers(4, 16))
        colour = tuple(int(c) for c in rng.integers(30, 226, size=3))
        cv2.circle(img, (int(cx), int(cy)), r, colour, -1)
    speckle = rng.normal(0, 8, size=img.shape)
    return np.clip(img.astype(np.float64) + speckle, 0, 255).astype(np.uint8)


def pan_frames(n: int = 60, width: int = 320, height: int = 240, step: int = 3) -> list[np.ndarray]:
    """A wide textured strip, cropped at a moving offset: a smooth camera pan."""
    strip = textured_scene(width + n * step + 8, height, seed=1)
    return [strip[:, i * step : i * step + width].copy() for i in range(n)]


def rotation_frames(
    n: int = 12, width: int = 320, height: int = 240, deg_per_frame: float = 2.0
) -> list[np.ndarray]:
    """Pure camera rotation about its own centre: a homography explains every pair."""
    base = textured_scene(width * 2, height * 2, seed=2)
    f = float(width)  # focal length in pixels
    k = np.array([[f, 0, width], [0, f, height], [0, 0, 1]], dtype=np.float64)
    k_inv = np.linalg.inv(k)
    out = []
    for i in range(n):
        theta = np.deg2rad(deg_per_frame * i)
        r = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        h = k @ r @ k_inv
        warped = cv2.warpPerspective(base, h, (base.shape[1], base.shape[0]))
        crop = warped[
            height // 2 : height // 2 + height, width // 2 : width // 2 + width
        ]
        out.append(np.ascontiguousarray(crop))
    return out


def translation_frames(
    n: int = 12, width: int = 320, height: int = 240, step: int = 6
) -> list[np.ndarray]:
    """Two planes at different depths sliding at different speeds: real parallax."""
    far = textured_scene(width + n * step * 2 + 16, height, seed=3)
    near = textured_scene(width + n * step * 4 + 16, height, seed=4)
    out = []
    for i in range(n):
        frame = far[:, i * step : i * step + width].copy()
        patch = near[:, i * step * 3 : i * step * 3 + width // 2]
        y0 = height // 4
        frame[y0 : y0 + height // 2, width // 4 : width // 4 + width // 2] = patch[
            : height // 2, : width // 2
        ]
        out.append(frame)
    return out


def write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> Path:
    """Write frames to an .mp4 (falls back to MJPG/.avi if mp4v is unavailable)."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        writer.release()
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    if not path.is_file() or path.stat().st_size == 0:  # pragma: no cover
        raise RuntimeError("OpenCV could not write a test video")
    return path


def write_photo(
    path: Path,
    frame: np.ndarray,
    camera: str | None = "TestCam X1",
    when: str | None = None,
    exposure: tuple[int, int] | None = (1, 125),
    iso: int | None = 200,
    focal_35mm: int | None = 26,
) -> Path:
    """Save a BGR frame as a JPEG with the EXIF fields CaptureCheck reads."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    exif = Image.Exif()
    if camera:
        make, _, model = camera.partition(" ")
        exif[ExifTags.Base.Make.value] = make
        exif[ExifTags.Base.Model.value] = model or make
    ifd = exif.get_ifd(ExifTags.IFD.Exif)
    if when:
        ifd[ExifTags.Base.DateTimeOriginal.value] = when
    if exposure:
        ifd[ExifTags.Base.ExposureTime.value] = exposure
    if iso:
        ifd[ExifTags.Base.ISOSpeedRatings.value] = iso
    if focal_35mm:
        ifd[ExifTags.Base.FocalLength.value] = (focal_35mm, 1)
        ifd[ExifTags.Base.FocalLengthIn35mmFilm.value] = focal_35mm
    im.save(path, quality=95, exif=exif)
    return path


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def narrow_width() -> int:
    """Measure test frames at their native width to keep the tests fast."""
    return 320


@pytest.fixture(scope="session")
def video_frames() -> list[np.ndarray]:
    return pan_frames(60)


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory, video_frames) -> Path:
    """60 frames of a moving checkerboard at 30 fps (2 seconds)."""
    out = tmp_path_factory.mktemp("video")
    return write_video(out / "tiny.mp4", video_frames, fps=30.0)


@pytest.fixture(scope="session")
def tiny_capture_dir(tmp_path_factory, video_frames) -> Path:
    """A ``data/raw/<capture_id>/`` folder holding ``video.mp4`` and ``scene.txt``."""
    root = tmp_path_factory.mktemp("raw_video")
    folder = root / "kitchen_video"
    folder.mkdir()
    write_video(folder / "video.mp4", video_frames, fps=30.0)
    (folder / "scene.txt").write_text("kitchen\n", encoding="utf-8")
    return folder


@pytest.fixture(scope="session")
def bundle_frames() -> list[np.ndarray]:
    return pan_frames(20, step=9)


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory, bundle_frames) -> Path:
    """20 JPEGs of the same scene, with EXIF capture times two seconds apart."""
    folder = tmp_path_factory.mktemp("bundle") / "kitchen_photos"
    folder.mkdir()
    for i, frame in enumerate(bundle_frames):
        write_photo(
            folder / f"IMG_{i:04d}.jpg",
            frame,
            when=f"2026:01:02 10:{i * 2 // 60:02d}:{i * 2 % 60:02d}",
        )
    return folder


@pytest.fixture(scope="session")
def tiny_bundle_zip(tmp_path_factory, tiny_bundle) -> Path:
    """The same 20 photos packed into a .zip."""
    out = tmp_path_factory.mktemp("bundle_zip") / "kitchen_photos.zip"
    with zipfile.ZipFile(out, "w") as zf:
        for photo in sorted(tiny_bundle.glob("*.jpg")):
            zf.write(photo, arcname=f"kitchen_photos/{photo.name}")
    return out


@pytest.fixture(scope="session")
def no_exif_bundle(tmp_path_factory, bundle_frames) -> Path:
    """Photos with neither EXIF time nor sortable names."""
    folder = tmp_path_factory.mktemp("bundle_noexif") / "random_photos"
    folder.mkdir()
    for i, frame in enumerate(bundle_frames[:8]):
        write_photo(
            folder / f"{'abcdefgh'[i]}_{i * 7 % 8}.jpg",
            frame,
            camera=None,
            when=None,
            exposure=None,
            iso=None,
            focal_35mm=None,
        )
    return folder


@pytest.fixture(scope="session")
def rot_frames() -> list[np.ndarray]:
    return rotation_frames(12)


@pytest.fixture(scope="session")
def trans_frames() -> list[np.ndarray]:
    return translation_frames(12)


@pytest.fixture()
def sharp_image() -> np.ndarray:
    return textured_scene(320, 240, seed=5)


@pytest.fixture()
def tmp_labels_dir(tmp_path) -> Path:
    out = tmp_path / "labels"
    out.mkdir()
    return out
