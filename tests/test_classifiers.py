"""The shiny-surface classifier, and the pipeline's behaviour without one."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from capturecheck.features.dino import EMBED_DIM, l2_normalise
from capturecheck.labeling import classifiers
from capturecheck.labeling.frame_rules import label_frames


@pytest.fixture(autouse=True)
def clear_cache():
    classifiers.clear_cache()
    yield
    classifiers.clear_cache()


def test_predict_shiny_is_nan_without_a_trained_model(tmp_path):
    emb = np.zeros((4, EMBED_DIM), dtype=np.float32)
    probabilities = classifiers.predict_shiny(emb, models_dir=tmp_path)
    assert probabilities.shape == (4,)
    assert np.isnan(probabilities).all()


def test_an_untrained_classifier_never_flags_a_frame(tmp_path):
    """No classifier must mean "not flagged", not "flagged by default"."""
    frames = pd.DataFrame({"shiny_prob": [np.nan] * 5, "sharpness": [500.0] * 5})
    labelled = label_frames(frames)
    assert (labelled["label_shiny"] == 0).all()


class FakeEmbedder:
    """Separable embeddings: shiny images get a different direction."""

    def embed(self, frames, batch_size=8, max_frames=None):
        vectors = np.zeros((len(frames), EMBED_DIM), dtype=np.float32)
        for i, frame in enumerate(frames):
            # The fixture writes a bright patch into the shiny examples.
            shiny = float(frame.mean()) > 128
            vectors[i, 0] = 1.0 if shiny else 0.0
            vectors[i, 1] = 0.0 if shiny else 1.0
            vectors[i, 2:] = np.random.default_rng(i).normal(0, 0.01, EMBED_DIM - 2)
        return l2_normalise(vectors)


@pytest.fixture()
def labelled_frames(tmp_path):
    import cv2

    folder = tmp_path / "classifier_train"
    images = folder / "images"
    images.mkdir(parents=True)
    rows = []
    for i in range(20):
        shiny = i % 2 == 0
        value = 220 if shiny else 60
        image = np.full((64, 64, 3), value, dtype=np.uint8)
        name = f"frame_{i:03d}.jpg"
        cv2.imwrite(str(images / name), image)
        rows.append({"image_path": f"images/{name}", "shiny": int(shiny)})

    csv_path = folder / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "shiny"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_train_shiny_reports_cross_validated_scores(labelled_frames, tmp_path):
    models = tmp_path / "models"
    report = classifiers.train_shiny(
        labelled_frames, models_dir=models, embedder=FakeEmbedder(), folds=5
    )
    assert report.n_samples == 20
    assert report.n_positive == 10
    assert report.precision == pytest.approx(1.0)
    assert report.recall == pytest.approx(1.0)
    assert (models / classifiers.SHINY_MODEL_NAME).is_file()
    assert "5-fold CV precision" in report.summary()


def test_a_trained_classifier_is_used(labelled_frames, tmp_path):
    models = tmp_path / "models"
    classifiers.train_shiny(
        labelled_frames, models_dir=models, embedder=FakeEmbedder(), folds=5
    )
    classifiers.clear_cache()

    shiny_vector = l2_normalise(np.eye(EMBED_DIM, dtype=np.float32)[:1])
    matte_vector = l2_normalise(np.eye(EMBED_DIM, dtype=np.float32)[1:2])
    probabilities = classifiers.predict_shiny(
        np.vstack([shiny_vector, matte_vector]), models_dir=models
    )
    assert probabilities[0] > 0.5 > probabilities[1]


def test_training_needs_both_classes(tmp_path):
    folder = tmp_path / "one_class"
    folder.mkdir()
    csv_path = folder / "labels.csv"
    csv_path.write_text("image_path,shiny\na.jpg,1\nb.jpg,1\n")
    with pytest.raises(ValueError, match="both shiny and non-shiny"):
        classifiers.train_shiny(csv_path, models_dir=tmp_path, embedder=FakeEmbedder())


def test_bad_csv_is_rejected(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("path,label\na.jpg,1\n")
    with pytest.raises(ValueError, match="image_path,shiny"):
        classifiers.train_shiny(csv_path, models_dir=tmp_path, embedder=FakeEmbedder())
