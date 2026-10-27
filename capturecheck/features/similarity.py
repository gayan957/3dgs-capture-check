"""What the DINOv2 embeddings say: duplicates, unique views and diversity."""

from __future__ import annotations

import numpy as np

from capturecheck.config import get_config


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """Full cosine-similarity matrix of already L2-normalised embeddings."""
    if emb.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.maximum(norms, 1e-12)
    return np.clip(unit @ unit.T, -1.0, 1.0).astype(np.float32)


def similarity_to_previous(emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of each frame to the one before it (first frame: 0)."""
    n = emb.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if n < 2:
        return out
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.maximum(norms, 1e-12)
    out[1:] = np.clip(np.einsum("ij,ij->i", unit[1:], unit[:-1]), -1.0, 1.0)
    return out


def duplicates(emb: np.ndarray, thr: float | None = None) -> np.ndarray:
    """Boolean mask of frames too similar to the last frame we kept.

    Comparing against the last *kept* frame (rather than the immediate
    predecessor) stops a slow pan from being thrown away one frame at a time.
    """
    thr = get_config().labeling.views.duplicate_cosine if thr is None else float(thr)
    n = emb.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.maximum(norms, 1e-12)
    last_kept = 0
    for i in range(1, n):
        if float(unit[i] @ unit[last_kept]) > thr:
            mask[i] = True
        else:
            last_kept = i
    return mask


def unique_views(emb: np.ndarray, thr: float | None = None) -> np.ndarray:
    """Indices of a greedy subset of frames that all differ from each other."""
    thr = get_config().labeling.views.duplicate_cosine if thr is None else float(thr)
    n = emb.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.maximum(norms, 1e-12)
    kept: list[int] = [0]
    for i in range(1, n):
        sims = unit[kept] @ unit[i]
        if float(sims.max()) <= thr:
            kept.append(i)
    return np.array(kept, dtype=int)


def diversity(emb: np.ndarray) -> float:
    """Mean pairwise cosine *distance*, 0 (all the same) to 1 (all different)."""
    n = emb.shape[0]
    if n < 2:
        return 0.0
    sim = cosine_matrix(emb)
    iu = np.triu_indices(n, k=1)
    mean_sim = float(sim[iu].mean())
    return float(np.clip(1.0 - mean_sim, 0.0, 1.0))


def nearest_neighbour_pairs(emb: np.ndarray) -> list[tuple[int, int]]:
    """Pair each frame with its most similar *other* frame.

    Used for photo bundles with no EXIF time and no usable file names: there is no
    "next photo", so the closest-looking photo is the best stand-in for one.
    """
    n = emb.shape[0]
    if n < 2:
        return []
    sim = cosine_matrix(emb).astype(np.float64)
    np.fill_diagonal(sim, -np.inf)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        j = int(np.argmax(sim[i]))
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, j))
    return pairs


def coverage_proxy(emb: np.ndarray, n_clusters: int = 8) -> float:
    """How evenly the views spread out, as a fallback when the path is unreliable.

    Clusters the embeddings and returns the share of clusters that hold a
    reasonable number of frames; a capture that circled a room fills them all.
    """
    n = emb.shape[0]
    if n < n_clusters:
        return float(n) / float(n_clusters) if n_clusters else 0.0
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=n_clusters, n_init=4, random_state=0).fit_predict(emb)
    counts = np.bincount(labels, minlength=n_clusters)
    expected = n / n_clusters
    return float((counts >= 0.3 * expected).mean())
