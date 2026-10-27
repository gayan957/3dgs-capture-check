#!/usr/bin/env python
"""Step 0 (optional): generate a small synthetic dataset in ``data/raw``.

Real captures are what CaptureCheck is for, but a synthetic set lets you run the
whole pipeline -- label, degrade, split, train, evaluate -- before any filming is
done, and it is what the smoke run uses.

The scene is a textured box on a textured floor, rendered by projecting each
planar face through a homography. That matters: a surface with continuous texture
gives stable ORB features between neighbouring views, which is exactly what the
overlap, walked-vs-turned and coverage checks measure. The camera orbits the box
over a different arc per scene, so the set contains a real quality gradient from
"filmed from every side" down to "filmed from one corner".

A ``psnr`` target is written for every capture. It is a *stand-in* computed from
the capture geometry, not a 3DGS measurement; replace
``data/labels/quality_targets.csv`` with real PSNR values for real captures.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from capturecheck.config import get_config
from capturecheck.io.load import save_frames
from capturecheck.utils.logging import get_logger, setup_logging

log = get_logger(__name__)

WORLD_UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# Textures
# --------------------------------------------------------------------------- #
def make_texture(width: int, height: int, seed: int) -> np.ndarray:
    """A richly textured, non-repetitive surface patch."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(55, 205, (max(2, height // 24), max(2, width // 24), 3), dtype=np.uint8)
    img = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    mid = rng.integers(55, 205, (max(2, height // 6), max(2, width // 6), 3), dtype=np.uint8)
    img = cv2.addWeighted(
        img, 0.6, cv2.resize(mid, (width, height), interpolation=cv2.INTER_LINEAR), 0.4, 0
    )
    for _ in range(90):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        colour = tuple(int(c) for c in rng.integers(45, 215, size=3))
        if rng.random() < 0.5:
            cv2.circle(img, centre, int(rng.integers(4, 22)), colour, -1)
        else:
            size = (int(rng.integers(8, 40)), int(rng.integers(8, 40)))
            top_left = (centre[0] - size[0] // 2, centre[1] - size[1] // 2)
            bottom_right = (top_left[0] + size[0], top_left[1] + size[1])
            cv2.rectangle(img, top_left, bottom_right, colour, -1)
    grain = rng.normal(0, 7, size=img.shape)
    img = np.clip(img.astype(np.float64) + grain, 25, 235).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 0.6)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class Face:
    """One planar, textured quad. Corners are counter-clockwise seen from outside."""

    corners: np.ndarray  # (4, 3) world coordinates
    texture: np.ndarray
    shade: float = 1.0

    @property
    def centroid(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def normal(self) -> np.ndarray:
        a = self.corners[1] - self.corners[0]
        b = self.corners[3] - self.corners[0]
        n = np.cross(a, b)
        return n / max(np.linalg.norm(n), 1e-9)


def box_faces(
    centre: np.ndarray, size: np.ndarray, seed: int, texture_px: int = 256
) -> list[Face]:
    """The six faces of an axis-aligned box, each with its own texture."""
    cx, cy, cz = centre
    sx, sy, sz = size / 2.0
    corners = {
        "x+": [(cx + sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz + sz),
               (cx + sx, cy + sy, cz + sz), (cx + sx, cy + sy, cz - sz)],
        "x-": [(cx - sx, cy - sy, cz + sz), (cx - sx, cy - sy, cz - sz),
               (cx - sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz + sz)],
        "z+": [(cx + sx, cy - sy, cz + sz), (cx - sx, cy - sy, cz + sz),
               (cx - sx, cy + sy, cz + sz), (cx + sx, cy + sy, cz + sz)],
        "z-": [(cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
               (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz)],
        "y+": [(cx - sx, cy + sy, cz - sz), (cx + sx, cy + sy, cz - sz),
               (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz)],
        "y-": [(cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
               (cx + sx, cy - sy, cz - sz), (cx - sx, cy - sy, cz - sz)],
    }
    shades = {"x+": 0.95, "x-": 0.78, "z+": 1.0, "z-": 0.72, "y+": 1.05, "y-": 0.6}
    out = []
    for i, (name, pts) in enumerate(corners.items()):
        out.append(
            Face(
                corners=np.array(pts, dtype=np.float64),
                texture=make_texture(texture_px, texture_px, seed + i),
                shade=shades[name],
            )
        )
    return out


def floor_faces(
    extent: float, tiles: int, seed: int, texture_px: int = 192, y: float = 0.0
) -> list[Face]:
    """The ground, cut into tiles so faces behind the camera can simply be dropped."""
    edges = np.linspace(-extent, extent, tiles + 1)
    out = []
    for i in range(tiles):
        for j in range(tiles):
            x0, x1 = edges[i], edges[i + 1]
            z0, z1 = edges[j], edges[j + 1]
            out.append(
                Face(
                    corners=np.array(
                        [(x0, y, z1), (x1, y, z1), (x1, y, z0), (x0, y, z0)], dtype=np.float64
                    ),
                    texture=make_texture(texture_px, texture_px, seed + 97 * i + j),
                    shade=0.9,
                )
            )
    return out


def build_scene(seed: int) -> list[Face]:
    """A main box, two smaller ones for occlusion and parallax, and the floor."""
    rng = np.random.default_rng(seed)
    faces: list[Face] = []
    faces += floor_faces(extent=6.0, tiles=6, seed=seed + 1000)
    faces += box_faces(np.array([0.0, 0.6, 0.0]), np.array([1.5, 1.2, 1.1]), seed=seed)
    faces += box_faces(
        np.array([float(rng.uniform(-1.6, -1.0)), 0.3, float(rng.uniform(0.6, 1.4))]),
        np.array([0.6, 0.6, 0.6]),
        seed=seed + 61,
    )
    faces += box_faces(
        np.array([float(rng.uniform(1.0, 1.7)), 0.45, float(rng.uniform(-1.4, -0.7))]),
        np.array([0.5, 0.9, 0.5]),
        seed=seed + 131,
    )
    return faces


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def camera_rotation(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-from-world rotation with axes x right, y down, z forward."""
    forward = target - position
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    right = np.cross(WORLD_UP, forward)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(right, forward)
    return np.stack([right, down, forward])


def render(
    faces: list[Face],
    position: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    focal: float,
    near: float = 0.3,
) -> np.ndarray:
    """Project every visible face through a homography, far to near."""
    rotation = camera_rotation(position, target)
    sky = np.linspace(150, 108, height, dtype=np.float32).reshape(height, 1)
    image = np.repeat(sky[:, :, None], 3, axis=2).repeat(width, axis=1).astype(np.uint8)

    drawable = []
    for face in faces:
        # Backface culling: only faces turned towards the camera.
        if np.dot(face.normal, position - face.centroid) <= 0:
            continue
        local = (face.corners - position) @ rotation.T
        if np.any(local[:, 2] < near):
            continue  # crosses the image plane; dropping it is good enough here
        projected = np.stack(
            [
                local[:, 0] / local[:, 2] * focal + width / 2.0,
                local[:, 1] / local[:, 2] * focal + height / 2.0,
            ],
            axis=1,
        )
        if projected.max() < -width or projected.min() > 2 * width:
            continue
        drawable.append((float(local[:, 2].mean()), face, projected))

    for _, face, projected in sorted(drawable, key=lambda item: -item[0]):
        th, tw = face.texture.shape[:2]
        source = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
        shaded = np.clip(face.texture.astype(np.float32) * face.shade, 0, 255).astype(np.uint8)
        warped = cv2.warpPerspective(
            shaded, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        mask = cv2.warpPerspective(
            np.full((th, tw), 255, np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        )
        image[mask > 127] = warped[mask > 127]

    return cv2.GaussianBlur(image, (0, 0), 0.5)


def orbit_poses(n: int, arc_deg: float, radius: float, height: float) -> list[np.ndarray]:
    """Camera positions walking round the subject: the capture you want."""
    angles = np.deg2rad(np.linspace(0.0, arc_deg, n, endpoint=False))
    return [
        np.array([radius * np.sin(a), height, radius * np.cos(a)], dtype=np.float64)
        for a in angles
    ]


def pivot_poses(
    n: int, arc_deg: float, radius: float, height: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Standing still and turning on the spot: the capture that cannot be solved.

    The camera never moves, so there is no parallax anywhere in the sequence and
    3DGS has nothing to recover depth from, however sharp every frame is. This
    is the failure mode `no_walking` exists to catch, and without an example of
    it in the set nothing can learn it.
    """
    position = np.array([0.0, height, radius], dtype=np.float64)
    angles = np.deg2rad(np.linspace(0.0, arc_deg, n, endpoint=False))
    targets = [
        position + np.array([np.sin(a), -0.1, -np.cos(a)], dtype=np.float64) * 4.0
        for a in angles
    ]
    return [position.copy() for _ in angles], targets


def geometry_psnr(
    arc_deg: float, n_frames: int, radius: float, motion: str = "orbit"
) -> float:
    """A stand-in quality target: more coverage and more views reconstruct better.

    Deliberately simple, and *not* a 3DGS measurement. It exists so the ranking
    stage has a defensible ordering to learn on a synthetic set. A capture that
    only pivoted gets the floor: with no baseline there is nothing to
    triangulate, so sweeping a wide arc from one spot buys nothing.
    """
    if motion == "pivot":
        return 17.0
    coverage = min(arc_deg, 360.0) / 360.0
    views = min(n_frames, 240) / 240.0
    baseline = min(radius, 4.0) / 4.0
    return float(18.0 + 9.0 * coverage + 4.0 * views + 1.5 * baseline)


#: Degrees between neighbouring frames. Held constant across scenes, so that the
#: quality gradient comes from *how far round* the camera got, not from a
#: different capture rate: it is the same hand filming at the same speed.
STEP_DEG = 2.5


def frame_count(arc_deg: float) -> int:
    return max(8, int(round(arc_deg / STEP_DEG)))


#: Some scenes are written as videos so the demo exercises the video path --
#: ffprobe/OpenCV, 2 fps sampling, optical flow -- and not only photo bundles.
SCENES: tuple[dict, ...] = (
    {"scene": "studio", "arc": 360.0, "radius": 4.2, "height": 1.6, "format": "images"},
    {"scene": "kitchen", "arc": 300.0, "radius": 4.0, "height": 1.4, "format": "video"},
    {"scene": "office", "arc": 240.0, "radius": 4.4, "height": 1.8, "format": "images"},
    {"scene": "hallway", "arc": 180.0, "radius": 3.8, "height": 1.3, "format": "video"},
    {"scene": "bedroom", "arc": 120.0, "radius": 4.0, "height": 1.5, "format": "images"},
    {"scene": "garage", "arc": 90.0, "radius": 3.6, "height": 1.2, "format": "video"},
    # Filmed by standing still and turning: no parallax anywhere, so the splat
    # cannot be solved however sharp the frames are.
    {
        "scene": "lobby", "arc": 160.0, "radius": 4.0, "height": 1.5,
        "format": "video", "motion": "pivot",
    },
)

COMPANY_SCENE = {
    "scene": "company_room", "arc": 330.0, "radius": 4.1, "height": 1.5, "format": "images",
}

#: Frames per second of the written videos. Each rendered pose is held for
#: ``VIDEO_FPS / sampling.fps`` frames, so sampling at 2 fps recovers exactly the
#: poses that were rendered.
VIDEO_FPS = 30.0
SAMPLE_FPS = 2.0


def write_video(path: Path, frames: list[np.ndarray], fps: float = VIDEO_FPS) -> Path:
    """Write the rendered poses as a video, holding each one long enough that
    sampling at 2 fps recovers them all."""
    hold = max(1, int(round(fps / SAMPLE_FPS)))
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        writer.release()
        raise RuntimeError(f"OpenCV cannot write {path}; is this build missing mp4v?")
    for frame in frames:
        for _ in range(hold):
            writer.write(frame)
    writer.release()
    return path


def render_capture(spec: dict, width: int, height: int, focal: float) -> list[np.ndarray]:
    faces = build_scene(seed=abs(hash(spec["scene"])) % 10_000)
    n = frame_count(spec["arc"])
    if spec.get("motion") == "pivot":
        poses, targets = pivot_poses(n, spec["arc"], spec["radius"], spec["height"])
    else:
        poses = orbit_poses(n, spec["arc"], spec["radius"], spec["height"])
        targets = [np.array([0.0, 0.6, 0.0])] * n
    return [
        render(faces, pose, target, width, height, focal)
        for pose, target in zip(poses, targets, strict=True)
    ]


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=cfg.paths.resolve("raw"))
    parser.add_argument("--labels", type=Path, default=cfg.paths.resolve("labels"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--clean", action="store_true", help="Delete the folder first.")
    args = parser.parse_args()

    setup_logging()
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    focal = 0.9 * args.width
    targets: list[dict] = []

    for spec in (*SCENES, COMPANY_SCENE):
        is_company = spec is COMPANY_SCENE
        capture_id = "company_room_scan" if is_company else f"{spec['scene']}_orbit"
        frames = render_capture(spec, args.width, args.height, focal)

        folder = args.out / capture_id
        folder.mkdir(parents=True, exist_ok=True)
        if spec.get("format") == "video":
            write_video(folder / "video.mp4", frames)
        else:
            save_frames(
                frames, folder / "images", names=[f"{i:04d}.jpg" for i in range(len(frames))]
            )
        (folder / "scene.txt").write_text(f"{spec['scene']}\n", encoding="utf-8")
        if is_company:
            (folder / "source.txt").write_text("company\n", encoding="utf-8")

        psnr = geometry_psnr(
            spec["arc"], len(frames), spec["radius"], spec.get("motion", "orbit")
        )
        targets.append(
            {
                "capture_id": capture_id,
                "psnr": round(psnr, 2),
                "ssim": "",
                "notes": "synthetic stand-in, not a 3DGS measurement",
            }
        )
        log.info(
            "%s: %d frames as %s, arc %.0f deg, stand-in psnr %.1f%s",
            capture_id, len(frames), spec.get("format", "images"), spec["arc"], psnr,
            " (company: test only)" if is_company else "",
        )

    args.labels.mkdir(parents=True, exist_ok=True)
    targets_path = args.labels / "quality_targets.csv"
    with targets_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["capture_id", "psnr", "ssim", "notes"])
        writer.writeheader()
        writer.writerows(targets)

    print(f"wrote {len(targets)} captures to {args.out}")
    print(f"wrote stand-in quality targets to {targets_path}")


if __name__ == "__main__":
    main()
