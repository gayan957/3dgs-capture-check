"""Small classifiers on top of frozen DINOv2 features.

Only ~200 hand-labelled frames are needed, because the embeddings already
separate shiny and matte surfaces; all that is left to learn is a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capturecheck.config import get_config
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

SHINY_MODEL_NAME = "shiny_classifier.joblib"


def shiny_model_path(models_dir: str | Path | None = None) -> Path:
    cfg = get_config()
    folder = Path(models_dir) if models_dir else cfg.paths.resolve("models")
    return folder / SHINY_MODEL_NAME


@dataclass
class ShinyReport:
    """Cross-validated quality of the shiny classifier."""

    n_samples: int
    n_positive: int
    precision: float
    recall: float
    f1: float
    path: str

    def summary(self) -> str:
        return (
            f"shiny classifier: {self.n_samples} frames ({self.n_positive} shiny), "
            f"5-fold CV precision {self.precision:.2f}, recall {self.recall:.2f}, "
            f"f1 {self.f1:.2f} -> {self.path}"
        )


def train_shiny(
    csv_path: str | Path,
    models_dir: str | Path | None = None,
    embedder=None,
    folds: int = 5,
) -> ShinyReport:
    """Train on ``data/classifier_train/labels.csv`` (``image_path,shiny``).

    Reports cross-validated precision and recall, then saves a logistic
    regression to ``models/shiny_classifier.joblib``.
    """
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from capturecheck.features.dino import DinoEmbedder
    from capturecheck.io.images import load_photo

    csv_path = Path(csv_path)
    table = pd.read_csv(csv_path)
    missing = {"image_path", "shiny"} - set(table.columns)
    if missing:
        raise ValueError(f"{csv_path} must have columns image_path,shiny (missing {missing})")

    root = csv_path.parent
    paths, labels = [], []
    for _, row in table.iterrows():
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            log.warning("skipping missing image %s", path)
            continue
        paths.append(path)
        labels.append(int(row["shiny"]))

    if len(set(labels)) < 2:
        raise ValueError("need both shiny and non-shiny examples")

    embedder = embedder or DinoEmbedder()
    frames = [load_photo(p) for p in paths]
    keep = [i for i, f in enumerate(frames) if f is not None]
    frames = [frames[i] for i in keep]
    y = np.array([labels[i] for i in keep], dtype=int)
    log.info("embedding %d labelled frames", len(frames))
    x = embedder.embed(frames, max_frames=len(frames))

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    n_folds = int(min(folds, np.bincount(y).min()))
    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        predicted = cross_val_predict(model, x, y, cv=cv)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predicted, average="binary", zero_division=0
        )
    else:  # pragma: no cover - only with a handful of examples
        log.warning("not enough examples per class for cross-validation")
        precision = recall = f1 = float("nan")

    model.fit(x, y)
    target = shiny_model_path(models_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "embed_dim": x.shape[1]}, target)

    report = ShinyReport(
        n_samples=int(len(y)),
        n_positive=int(y.sum()),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        path=str(target),
    )
    log.info("%s", report.summary())
    return report


_CACHE: dict[str, object] = {}


def load_shiny(models_dir: str | Path | None = None):
    """The saved classifier, or ``None`` when it has not been trained yet."""
    path = shiny_model_path(models_dir)
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    if not path.is_file():
        _CACHE[key] = None
        return None
    try:
        import joblib

        payload = joblib.load(path)
        _CACHE[key] = payload.get("model") if isinstance(payload, dict) else payload
    except Exception as exc:  # pragma: no cover - corrupted file
        log.warning("cannot load %s (%s)", path, exc)
        _CACHE[key] = None
    return _CACHE[key]


def predict_shiny(emb: np.ndarray, models_dir: str | Path | None = None) -> np.ndarray:
    """Shiny probability per embedding. All-NaN when no classifier is saved yet."""
    if emb.size == 0:
        return np.zeros(0, dtype=float)
    model = load_shiny(models_dir)
    if model is None:
        return np.full(emb.shape[0], np.nan, dtype=float)
    try:
        return model.predict_proba(emb)[:, 1].astype(float)
    except Exception as exc:  # pragma: no cover - dimension mismatch
        log.warning("shiny classifier failed (%s); skipping", exc)
        return np.full(emb.shape[0], np.nan, dtype=float)


def clear_cache() -> None:
    _CACHE.clear()
