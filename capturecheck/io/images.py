"""Image-bundle input: a folder or .zip of photos, plus their EXIF data."""

from __future__ import annotations

import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from capturecheck.config import get_config
from capturecheck.io.video import resize_to_width
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

_EXIF_TAGS = {name: tag for tag, name in ExifTags.TAGS.items()}
_SKIP_DIR_NAMES = {"__MACOSX", ".git"}


@dataclass
class ExifInfo:
    """The EXIF fields that matter for capture quality (``None`` when missing)."""

    path: str
    width: int | None = None
    height: int | None = None
    camera: str | None = None
    exposure_s: float | None = None
    iso: float | None = None
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    datetime_original: datetime | None = None

    @property
    def size(self) -> tuple[int | None, int | None]:
        return (self.width, self.height)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["datetime_original"] = (
            self.datetime_original.isoformat() if self.datetime_original else None
        )
        return out


@dataclass
class BundleInfo:
    """Summary of a photo bundle: how many photos, cameras and image sizes."""

    n_photos: int = 0
    n_cameras: int = 0
    n_sizes: int = 0
    cameras: list[str] = field(default_factory=list)
    sizes: list[tuple[int, int]] = field(default_factory=list)
    has_exif_time: bool = False
    order_source: str = "name"  # exif_time | name | unknown
    exposure_s_median: float | None = None
    iso_median: float | None = None
    focal_length_mm: float | None = None

    def as_dict(self) -> dict:
        out = asdict(self)
        out["sizes"] = [f"{w}x{h}" for w, h in self.sizes]
        return out


@dataclass
class BundleRead:
    """What :func:`read_bundle` gives back."""

    frames: list[np.ndarray]
    names: list[str]
    times: list[float | None]
    exif: list[ExifInfo]
    info: BundleInfo


def _rational(value) -> float | None:
    """Pillow returns IFDRational / tuples / strings; make a float of it."""
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if float(den) else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_exif(img_path: str | Path) -> ExifInfo:
    """Camera, size, exposure time, ISO, focal length and capture time of one photo."""
    img_path = Path(img_path)
    out = ExifInfo(path=str(img_path))
    try:
        with Image.open(img_path) as im:
            out.width, out.height = im.size
            exif = im.getexif()
            if not exif:
                return out

            make = exif.get(_EXIF_TAGS.get("Make"))
            model = exif.get(_EXIF_TAGS.get("Model"))
            parts = [str(p).strip() for p in (make, model) if p]
            out.camera = " ".join(parts) if parts else None

            # Exposure, ISO and focal length live in the Exif IFD.
            try:
                ifd = exif.get_ifd(ExifTags.IFD.Exif)
            except Exception:  # pragma: no cover - old Pillow
                ifd = {}

            def pick(name: str):
                tag = _EXIF_TAGS.get(name)
                if tag is None:
                    return None
                return ifd.get(tag, exif.get(tag))

            out.exposure_s = _rational(pick("ExposureTime"))
            if out.exposure_s is None:
                shutter = _rational(pick("ShutterSpeedValue"))
                if shutter is not None:
                    out.exposure_s = float(2.0 ** (-shutter))

            iso = pick("ISOSpeedRatings") or pick("PhotographicSensitivity")
            if isinstance(iso, (list, tuple)):
                iso = iso[0] if iso else None
            out.iso = _rational(iso)

            out.focal_length_mm = _rational(pick("FocalLength"))
            out.focal_length_35mm = _rational(pick("FocalLengthIn35mmFilm"))

            raw_dt = pick("DateTimeOriginal") or exif.get(_EXIF_TAGS.get("DateTime"))
            if raw_dt:
                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        out.datetime_original = datetime.strptime(str(raw_dt).strip(), fmt)
                        break
                    except ValueError:
                        continue
    except Exception as exc:  # unreadable or non-image file
        log.debug("no EXIF for %s (%s)", img_path.name, exc)
    return out


def bundle_info(photos: list[ExifInfo], order_source: str = "name") -> BundleInfo:
    """Count photos, distinct cameras and distinct image sizes."""
    cameras = sorted({p.camera for p in photos if p.camera})
    sizes = sorted(
        {(p.width, p.height) for p in photos if p.width and p.height},
        key=lambda s: (-(s[0] * s[1]), s),
    )
    exposures = [p.exposure_s for p in photos if p.exposure_s]
    isos = [p.iso for p in photos if p.iso]
    focals = [p.focal_length_mm for p in photos if p.focal_length_mm]

    return BundleInfo(
        n_photos=len(photos),
        # An unlabelled camera still counts as one camera.
        n_cameras=max(len(cameras), 1 if photos else 0),
        n_sizes=max(len(sizes), 1 if photos else 0),
        cameras=cameras,
        sizes=[(int(w), int(h)) for w, h in sizes],
        has_exif_time=any(p.datetime_original for p in photos),
        order_source=order_source,
        exposure_s_median=float(np.median(exposures)) if exposures else None,
        iso_median=float(np.median(isos)) if isos else None,
        focal_length_mm=float(np.median(focals)) if focals else None,
    )


def is_image_bundle(path: str | Path, extensions: list[str] | None = None) -> bool:
    """True for a .zip, or a folder that contains photos (directly or in ``images/``)."""
    path = Path(path)
    if path.is_file():
        return path.suffix.lower() == ".zip"
    if not path.is_dir():
        return False
    return bool(find_photos(path, extensions))


def find_photos(folder: str | Path, extensions: list[str] | None = None) -> list[Path]:
    """Photo files in a folder: directly inside, else in ``images/``, else anywhere below."""
    cfg = get_config()
    exts = {e.lower() for e in (extensions or cfg.labeling.images.extensions)}
    folder = Path(folder)
    if not folder.is_dir():
        return []

    def keep(p: Path) -> bool:
        if p.suffix.lower() not in exts or p.name.startswith("."):
            return False
        return not any(part in _SKIP_DIR_NAMES or part.startswith(".") for part in p.parts)

    direct = sorted(p for p in folder.iterdir() if p.is_file() and keep(p))
    if direct:
        return direct

    images_dir = folder / "images"
    if images_dir.is_dir():
        nested = sorted(p for p in images_dir.rglob("*") if p.is_file() and keep(p))
        if nested:
            return nested

    return sorted(
        p for p in folder.rglob("*") if p.is_file() and keep(p.relative_to(folder))
    )


def _extract_zip(zip_path: Path, dest: Path, extensions: set[str]) -> list[Path]:
    """Unzip only the photo entries, flat, ignoring any path tricks in the archive."""
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename)
            if name.suffix.lower() not in extensions or name.name.startswith("."):
                continue
            if any(part in _SKIP_DIR_NAMES for part in name.parts):
                continue
            target = dest / name.name
            if target.exists():  # same file name in two folders of the archive
                target = dest / f"{name.parent.name}__{name.name}"
            with zf.open(member) as src, target.open("wb") as fh:
                fh.write(src.read())
            out.append(target)
    return sorted(out)


def _order_photos(exifs: list[ExifInfo], mode: str) -> tuple[list[int], str]:
    """Indices in capture order, and which signal was used."""
    have_time = [e.datetime_original is not None for e in exifs]
    if mode.startswith("exif_time") and all(have_time) and len(exifs) > 1:
        order = sorted(range(len(exifs)), key=lambda i: (exifs[i].datetime_original, exifs[i].path))
        return order, "exif_time"
    if mode.startswith("exif_time") and any(have_time):
        # Partial EXIF: time first where known, file name otherwise.
        order = sorted(
            range(len(exifs)),
            key=lambda i: (
                0 if have_time[i] else 1,
                exifs[i].datetime_original or datetime.min,
                Path(exifs[i].path).name.lower(),
            ),
        )
        return order, "exif_time_partial"
    order = sorted(range(len(exifs)), key=lambda i: Path(exifs[i].path).name.lower())
    return order, "name"


def _time_offsets(exifs: list[ExifInfo]) -> list[float | None]:
    """Seconds since the first photo, when EXIF times are available."""
    stamps = [e.datetime_original for e in exifs]
    known = [s for s in stamps if s is not None]
    if not known:
        return [None] * len(exifs)
    t0 = min(known)
    return [None if s is None else (s - t0).total_seconds() for s in stamps]


def read_bundle(
    path: str | Path,
    max_frames: int | None = None,
    width: int | None = None,
) -> BundleRead:
    """Read a folder or .zip of photos in capture order, resized to ``width``.

    More than ``max_frames`` photos are thinned out evenly across the bundle.
    """
    cfg = get_config()
    max_frames = cfg.sampling.max_frames if max_frames is None else int(max_frames)
    width = cfg.resize_width if width is None else int(width)
    exts = {e.lower() for e in cfg.labeling.images.extensions}

    path = Path(path)
    tmpdir: tempfile.TemporaryDirectory | None = None
    try:
        if path.is_file() and path.suffix.lower() == ".zip":
            tmpdir = tempfile.TemporaryDirectory(prefix="capturecheck_zip_")
            photo_paths = _extract_zip(path, Path(tmpdir.name), exts)
        elif path.is_dir():
            photo_paths = find_photos(path, list(exts))
        elif path.is_file() and path.suffix.lower() in exts:
            photo_paths = [path]
        else:
            raise FileNotFoundError(f"not an image bundle: {path}")

        if not photo_paths:
            raise ValueError(f"no photos ({sorted(exts)}) found in {path}")

        exifs = [read_exif(p) for p in photo_paths]
        order, order_source = _order_photos(exifs, cfg.labeling.images.order)
        exifs = [exifs[i] for i in order]
        photo_paths = [photo_paths[i] for i in order]

        times = _time_offsets(exifs)

        if len(photo_paths) > max_frames:
            keep = np.linspace(0, len(photo_paths) - 1, num=max_frames).round().astype(int)
            keep = np.unique(keep)
            log.info("bundle has %d photos; keeping %d spread evenly", len(photo_paths), keep.size)
            photo_paths = [photo_paths[i] for i in keep]
            exifs = [exifs[i] for i in keep]
            times = [times[i] for i in keep]

        frames = [load_photo(p, width) for p in photo_paths]
        names = [Path(p).name for p in photo_paths]
        keep_mask = [f is not None for f in frames]
        if not all(keep_mask):
            bad = [n for n, ok in zip(names, keep_mask, strict=False) if not ok]
            log.warning("skipping %d unreadable photo(s): %s", len(bad), ", ".join(bad[:5]))
            frames = [f for f, ok in zip(frames, keep_mask, strict=False) if ok]
            names = [n for n, ok in zip(names, keep_mask, strict=False) if ok]
            times = [t for t, ok in zip(times, keep_mask, strict=False) if ok]
            exifs = [e for e, ok in zip(exifs, keep_mask, strict=False) if ok]
        if not frames:
            raise ValueError(f"no readable photos in {path}")

        info = bundle_info(exifs, order_source=order_source)
        if order_source == "name" and not info.has_exif_time:
            info.order_source = (
                "unknown" if cfg.labeling.images.unknown_order == "dino_nearest" else "name"
            )
        log.info(
            "%s: %d photos (order: %s, cameras: %d, sizes: %d)",
            path.name, len(frames), info.order_source, info.n_cameras, info.n_sizes,
        )
        return BundleRead(frames=frames, names=names, times=times, exif=exifs, info=info)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def load_photo(img_path: str | Path, width: int | None = None) -> np.ndarray | None:
    """Read one photo as BGR uint8, honouring the EXIF orientation flag."""
    cfg = get_config()
    width = cfg.resize_width if width is None else int(width)
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            rgb = np.asarray(im, dtype=np.uint8)
    except Exception as exc:
        log.debug("cannot read %s (%s)", img_path, exc)
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return resize_to_width(bgr, width)
