"""DINOv2 embeddings: one 384-d vector per frame, cached on disk.

DINOv2 is used **pretrained and frozen** -- nothing here is trained. The
embeddings answer questions pixels cannot: is this view a duplicate of the last
one, how varied is the capture, and (via a small logistic regression) is the
surface shiny. Blur is deliberately *not* measured here; OpenCV does that better
and faster.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from capturecheck.config import get_config
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

MODEL_NAME = "dinov2_vits14"
HUB_REPO = "facebookresearch/dinov2"
EMBED_DIM = 384
INPUT_SIZE = 224  # multiple of the patch size 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(preference: str | None = None) -> str:
    """``cpu`` unless a GPU is available and the config allows it."""
    want = (preference or get_config().runtime.device or "auto").lower()
    if want in {"cpu", "cuda", "mps"}:
        return want
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of [model]
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Apple's MPS backend is available but slower than CPU for this batch size.
    return "cpu"


def evenly_spaced(n_items: int, max_items: int) -> np.ndarray:
    """Indices of at most ``max_items`` items spread evenly over ``n_items``."""
    if n_items <= max_items:
        return np.arange(n_items)
    return np.unique(np.linspace(0, n_items - 1, num=max_items).round().astype(int))


def frames_hash(frames: list[np.ndarray]) -> str:
    """Short content hash of a frame list, used as the cache key."""
    digest = hashlib.sha1()
    digest.update(str(len(frames)).encode())
    for frame in frames:
        digest.update(np.ascontiguousarray(frame.shape).tobytes())
        # Hash a thumbnail: the full pixels would dominate the runtime.
        small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        digest.update(small.tobytes())
    return digest.hexdigest()[:16]


def preprocess(frames: list[np.ndarray], size: int = INPUT_SIZE) -> np.ndarray:
    """BGR uint8 frames -> (N, 3, size, size) float32, ImageNet-normalised."""
    batch = np.empty((len(frames), 3, size, size), dtype=np.float32)
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        batch[i] = arr.transpose(2, 0, 1)
    return batch


def l2_normalise(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


class DinoEmbedder:
    """Loads DINOv2 once and embeds frames, with an on-disk cache per capture.

    The model is only fetched from torch.hub when :meth:`embed` is first called,
    so importing this module stays offline-safe.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        cfg = get_config()
        self.model_name = model_name
        self.device = pick_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else cfg.paths.resolve("features")
        self._model = None

    # ---------------------------------------------------------------- model
    @property
    def model(self):
        """The frozen DINOv2 backbone (loaded on first use)."""
        if self._model is None:
            import torch

            log.info("loading %s on %s (first run downloads weights)", self.model_name, self.device)
            model = torch.hub.load(HUB_REPO, self.model_name, verbose=False)
            model.eval()
            model.to(self.device)
            for param in model.parameters():
                param.requires_grad_(False)
            self._model = model
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    # ---------------------------------------------------------------- embed
    def embed(
        self,
        frames: list[np.ndarray],
        batch_size: int = 8,
        max_frames: int | None = None,
    ) -> np.ndarray:
        """Embed frames into ``(N, 384)`` L2-normalised CLS features.

        At most ``sampling.dino_max_frames`` frames are used; the caller should
        pass the frames it wants embedded (see :func:`evenly_spaced`).
        """
        cfg = get_config()
        limit = cfg.sampling.dino_max_frames if max_frames is None else int(max_frames)
        if not frames:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        if len(frames) > limit:
            keep = evenly_spaced(len(frames), limit)
            frames = [frames[i] for i in keep]

        import torch

        out = np.empty((len(frames), EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                chunk = frames[start : start + batch_size]
                tensor = torch.from_numpy(preprocess(chunk)).to(self.device)
                feats = self.model(tensor)  # CLS token for dinov2_vit*
                if isinstance(feats, dict):  # pragma: no cover - other hub variants
                    feats = feats.get("x_norm_clstoken", next(iter(feats.values())))
                out[start : start + len(chunk)] = feats.float().cpu().numpy()
        return l2_normalise(out)

    # ---------------------------------------------------------------- cache
    def cache_path(self, capture_id: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{capture_id}.npz"

    def embed_capture(
        self,
        capture_id: str,
        frames: list[np.ndarray],
        max_frames: int | None = None,
        use_cache: bool = True,
        extra: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embeddings for one capture, cached in ``data/features/<capture_id>.npz``.

        Returns ``(embeddings, frame_indices)`` where the indices say which frames
        of the capture were embedded.
        """
        cfg = get_config()
        limit = cfg.sampling.dino_max_frames if max_frames is None else int(max_frames)
        keep = evenly_spaced(len(frames), limit)
        subset = [frames[i] for i in keep]
        key = frames_hash(subset)
        path = self.cache_path(capture_id)

        if use_cache and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    if str(data.get("hash", np.array(""))) == key and "emb" in data:
                        log.info("using cached DINOv2 features for %s", capture_id)
                        return data["emb"], data["frame_idx"]
            except (OSError, ValueError, KeyError):
                log.warning("cache for %s is unreadable; recomputing", capture_id)

        emb = self.embed(subset, max_frames=limit)
        payload = {
            "emb": emb,
            "frame_idx": keep,
            "hash": np.array(key),
            "model": np.array(self.model_name),
        }
        if extra:
            payload.update(extra)
        if use_cache:
            np.savez_compressed(path, **payload)
            log.info("cached DINOv2 features for %s -> %s", capture_id, path.name)
        return emb, keep


def load_cached(capture_id: str, cache_dir: str | Path | None = None) -> dict | None:
    """Everything stored in ``data/features/<capture_id>.npz``, or ``None``."""
    cfg = get_config()
    folder = Path(cache_dir) if cache_dir else cfg.paths.resolve("features")
    path = folder / f"{capture_id}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_cached(capture_id: str, payload: dict, cache_dir: str | Path | None = None) -> Path:
    """Add arrays to a capture's feature cache, keeping what is already there."""
    cfg = get_config()
    folder = Path(cache_dir) if cache_dir else cfg.paths.resolve("features")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{capture_id}.npz"
    existing = load_cached(capture_id, folder) or {}
    existing.update(payload)
    np.savez_compressed(path, **existing)
    return path
