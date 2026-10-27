"""How the camera moved: neighbour overlap, flow speed, and walked vs turned.

A capture that only *turns on the spot* gives 3DGS no parallax, so it cannot
recover depth however sharp the frames are. Telling "walked" from "turned" is
therefore one of the most useful things we can measure without running SfM.

The decision uses GRIC-style model selection: fit a homography (which explains
pure rotation and flat scenes) and an essential matrix (which needs real
translation) to the same ORB matches, and keep the model that explains the data
with fewer degrees of freedom.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from capturecheck.checks.quality import to_gray
from capturecheck.checks.texture import MAX_FEATURES, detect

WALKED = "walked"
TURNED = "turned"
UNKNOWN = "unknown"

#: Below this many ORB matches, no model can be trusted.
MIN_MATCHES = 20
#: Lowe ratio for descriptor matching.
RATIO_TEST = 0.75
#: RANSAC inlier threshold in pixels, at the measured width.
RANSAC_PX = 3.0
#: Width the optical flow runs at (Farneback is quadratic in pixels).
FLOW_WIDTH = 320
#: Scales the GRIC gap per correspondence into a 0-1 confidence.
GRIC_CONF_SCALE = 0.5


@dataclass
class OrbFrame:
    """ORB keypoints and descriptors for one frame, computed once and reused."""

    pts: np.ndarray  # (N, 2) float32 pixel coordinates
    desc: np.ndarray | None  # (N, 32) uint8 descriptors
    shape: tuple[int, int]  # (height, width)

    @property
    def n(self) -> int:
        return int(self.pts.shape[0])


@dataclass
class PairMotion:
    """What one pair of frames says about the camera motion between them."""

    kind: str  # walked | turned | unknown
    confidence: float
    n_matches: int = 0
    inliers_h: int = 0
    inliers_e: int = 0
    gric_h: float = float("nan")
    gric_e: float = float("nan")
    R: np.ndarray | None = None  # rotation, frame i -> frame i+1
    t: np.ndarray | None = None  # unit translation (scale is unknowable)

    def as_dict(self) -> dict:
        out = asdict(self)
        out.pop("R", None)
        out.pop("t", None)
        return out


def orb_frames(frames: list[np.ndarray], max_features: int = MAX_FEATURES) -> list[OrbFrame]:
    """Detect ORB once per frame; every later check reuses the result."""
    out: list[OrbFrame] = []
    for frame in frames:
        keypoints, desc = detect(frame, max_features)
        pts = (
            np.array([kp.pt for kp in keypoints], dtype=np.float32)
            if keypoints
            else np.zeros((0, 2), dtype=np.float32)
        )
        h, w = frame.shape[:2]
        out.append(OrbFrame(pts=pts, desc=desc, shape=(int(h), int(w))))
    return out


def _as_orb(frame_or_features) -> OrbFrame:
    if isinstance(frame_or_features, OrbFrame):
        return frame_or_features
    return orb_frames([frame_or_features])[0]


def _matcher() -> cv2.BFMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def match_features(a: OrbFrame, b: OrbFrame) -> tuple[np.ndarray, np.ndarray]:
    """Matched point pairs between two frames, after Lowe's ratio test."""
    if a.desc is None or b.desc is None or a.n < 2 or b.n < 2:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty

    knn = _matcher().knnMatch(a.desc, b.desc, k=2)
    idx_a: list[int] = []
    idx_b: list[int] = []
    for pair in knn:
        if len(pair) < 2:
            if pair:
                idx_a.append(pair[0].queryIdx)
                idx_b.append(pair[0].trainIdx)
            continue
        best, second = pair[0], pair[1]
        if best.distance < RATIO_TEST * second.distance:
            idx_a.append(best.queryIdx)
            idx_b.append(best.trainIdx)

    if not idx_a:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty
    return a.pts[idx_a], b.pts[idx_b]


def overlap(a, b) -> float:
    """Share of the features in ``a`` that are found again in ``b`` (0-1).

    Works for video frames and for neighbouring photos alike. Accepts images or
    pre-computed :class:`OrbFrame` objects.
    """
    fa, fb = _as_orb(a), _as_orb(b)
    if fa.n == 0:
        return 0.0
    pts_a, _ = match_features(fa, fb)
    return float(min(1.0, pts_a.shape[0] / float(fa.n)))


def overlap_series(features: list[OrbFrame]) -> list[float]:
    """Overlap for every neighbouring pair; the last frame has no successor."""
    out = [overlap(features[i], features[i + 1]) for i in range(len(features) - 1)]
    return out


def flow_speed(a: np.ndarray, b: np.ndarray, width: int = FLOW_WIDTH) -> float:
    """Median optical-flow magnitude between two frames, in pixels at ``width``.

    Videos only: at 2 fps the frames are 0.5 s apart, which is still close enough
    for dense flow. Photo bundles have no meaningful time step, so the labeller
    skips this measurement for them.
    """
    ga, gb = to_gray(a), to_gray(b)
    h, w = ga.shape[:2]
    if w > width:
        scale = width / float(w)
        size = (width, max(1, int(round(h * scale))))
        ga = cv2.resize(ga, size, interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, size, interpolation=cv2.INTER_AREA)

    flow = cv2.calcOpticalFlowFarneback(
        ga, gb, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.linalg.norm(flow, axis=2)
    return float(np.median(magnitude))


def approx_intrinsics(shape: tuple[int, int], focal_px: float | None = None) -> np.ndarray:
    """Camera matrix K. Uses the EXIF focal length when known, else a 0.9*width guess."""
    h, w = shape
    f = float(focal_px) if focal_px else 0.9 * float(w)
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _gric(
    errors_sq: np.ndarray, sigma: float, dim_manifold: int, n_params: int, dim_data: int = 4
) -> float:
    """GRIC score of one model: lower explains the matches better per parameter.

    ``errors_sq`` are squared residuals, ``dim_manifold`` is 2 for a homography and
    3 for an essential matrix, ``n_params`` is 8 and 5 respectively.
    """
    n = int(errors_sq.size)
    if n == 0:
        return float("inf")
    lambda_1 = np.log(float(dim_data))
    lambda_2 = np.log(float(dim_data * n))
    lambda_3 = 2.0
    limit = lambda_3 * (dim_data - dim_manifold)
    rho = np.minimum(errors_sq / (sigma**2), limit)
    return float(rho.sum() + lambda_1 * dim_manifold * n + lambda_2 * n_params)


def _homography_errors(
    pts_a: np.ndarray, pts_b: np.ndarray, h: np.ndarray
) -> np.ndarray:
    """Symmetric transfer error (squared pixels) of a homography."""
    def transfer(src, mat):
        pts = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64), mat)
        return pts.reshape(-1, 2)

    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return np.full(pts_a.shape[0], np.inf)
    fwd = np.sum((transfer(pts_a, h) - pts_b) ** 2, axis=1)
    bwd = np.sum((transfer(pts_b, h_inv) - pts_a) ** 2, axis=1)
    return (fwd + bwd) / 2.0


def _sampson_errors(pts_a: np.ndarray, pts_b: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Sampson distance (squared pixels) of a fundamental/essential matrix."""
    ones = np.ones((pts_a.shape[0], 1))
    x1 = np.hstack([pts_a.astype(np.float64), ones])
    x2 = np.hstack([pts_b.astype(np.float64), ones])
    fx1 = x1 @ f.T
    ftx2 = x2 @ f
    num = np.einsum("ij,ij->i", x2, fx1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 1e-12, num / den, np.inf)
    return out


def walked_or_turned(
    a,
    b,
    K: np.ndarray | None = None,
    focal_px: float | None = None,
) -> PairMotion:
    """Decide whether the camera moved through the scene or only pivoted.

    Returns a :class:`PairMotion` with ``kind`` in ``walked`` / ``turned`` /
    ``unknown`` and a 0-1 confidence. Pure rotation or a flat scene favours the
    homography and therefore ``turned``.
    """
    fa, fb = _as_orb(a), _as_orb(b)
    pts_a, pts_b = match_features(fa, fb)
    n = int(pts_a.shape[0])
    if n < MIN_MATCHES:
        return PairMotion(kind=UNKNOWN, confidence=0.0, n_matches=n)

    k = K if K is not None else approx_intrinsics(fa.shape, focal_px)

    h_mat, h_mask = cv2.findHomography(
        pts_a, pts_b, cv2.RANSAC, RANSAC_PX, maxIters=2000, confidence=0.999
    )
    e_mat, e_mask = cv2.findEssentialMat(
        pts_a, pts_b, k, method=cv2.RANSAC, prob=0.999, threshold=RANSAC_PX
    )
    if h_mat is None and e_mat is None:
        return PairMotion(kind=UNKNOWN, confidence=0.0, n_matches=n)
    if e_mat is not None and e_mat.shape[0] > 3:
        e_mat = e_mat[:3, :3]  # OpenCV can return several solutions stacked

    inliers_h = int(h_mask.sum()) if h_mask is not None else 0
    inliers_e = int(e_mask.sum()) if e_mask is not None else 0

    gric_h = float("inf")
    if h_mat is not None:
        gric_h = _gric(_homography_errors(pts_a, pts_b, h_mat), RANSAC_PX, 2, 8)
    gric_e = float("inf")
    if e_mat is not None:
        f_mat = np.linalg.inv(k).T @ e_mat @ np.linalg.inv(k)
        gric_e = _gric(_sampson_errors(pts_a, pts_b, f_mat), RANSAC_PX, 3, 5)

    if not np.isfinite(gric_h) and not np.isfinite(gric_e):
        return PairMotion(kind=UNKNOWN, confidence=0.0, n_matches=n)

    kind = TURNED if gric_h <= gric_e else WALKED
    gap_per_point = abs(gric_h - gric_e) / float(n)
    confidence = float(np.clip(gap_per_point / GRIC_CONF_SCALE, 0.0, 1.0))

    rotation = translation = None
    if kind == WALKED and e_mat is not None:
        try:
            _, rotation, translation, _ = cv2.recoverPose(
                e_mat, pts_a, pts_b, k, mask=e_mask.copy()
            )
            norm = np.linalg.norm(translation)
            translation = (translation / norm) if norm > 1e-9 else translation
        except cv2.error:  # pragma: no cover - degenerate configurations
            rotation = translation = None
    elif h_mat is not None:
        # A turn still has a rotation we can use for the coverage path.
        try:
            n_sol, rots, _, _ = cv2.decomposeHomographyMat(h_mat, k)
            if n_sol:
                rotation = rots[0]
                translation = np.zeros((3, 1), dtype=np.float64)
        except cv2.error:  # pragma: no cover
            rotation = translation = None

    return PairMotion(
        kind=kind,
        confidence=confidence,
        n_matches=n,
        inliers_h=inliers_h,
        inliers_e=inliers_e,
        gric_h=gric_h,
        gric_e=gric_e,
        R=rotation,
        t=translation,
    )


def motion_pairs(
    frames: list[np.ndarray],
    features: list[OrbFrame] | None = None,
    gap: int = 2,
    focal_px: float | None = None,
    is_video: bool = True,
    order: list[int] | None = None,
) -> list[tuple[int, int, PairMotion]]:
    """Walked-vs-turned for every sampled pair.

    Videos use pairs ``gap`` frames apart (~1 s at 2 fps); photo bundles use each
    photo and the next. ``order`` lets an unordered bundle supply its own pairing
    (see :func:`capturecheck.features.similarity.nearest_neighbour_pairs`).
    """
    features = features if features is not None else orb_frames(frames)
    step = max(1, int(gap)) if is_video else 1
    k = approx_intrinsics(features[0].shape, focal_px) if features else None

    pairs: list[tuple[int, int]]
    if order is not None:
        pairs = list(order)  # type: ignore[arg-type]
    else:
        pairs = [(i, i + step) for i in range(0, len(features) - step, step)]

    out = []
    for i, j in pairs:
        if j >= len(features):
            continue
        out.append((i, j, walked_or_turned(features[i], features[j], K=k)))
    return out


def walked_ratio(pairs: list[tuple[int, int, PairMotion]]) -> float:
    """Share of decided pairs that show real camera movement."""
    decided = [p for _, _, p in pairs if p.kind in (WALKED, TURNED)]
    if not decided:
        return float("nan")
    return float(sum(1 for p in decided if p.kind == WALKED) / len(decided))
