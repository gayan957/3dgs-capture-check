"""DINOv2 helpers and the similarity measures built on them.

The model itself is mocked: the tests must run offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from capturecheck.features import dino, similarity


def test_evenly_spaced_caps_and_spreads():
    assert dino.evenly_spaced(10, 80).tolist() == list(range(10))
    picked = dino.evenly_spaced(600, 80)
    assert len(picked) == 80
    assert picked[0] == 0 and picked[-1] == 599


def test_preprocess_shape_and_normalisation(sharp_image):
    batch = dino.preprocess([sharp_image, sharp_image], size=224)
    assert batch.shape == (2, 3, 224, 224)
    assert batch.dtype == np.float32
    # ImageNet normalisation puts values roughly in [-2.2, 2.7].
    assert -3.0 < batch.min() < batch.max() < 3.0


def test_frames_hash_is_stable_and_content_sensitive(sharp_image):
    other = np.roll(sharp_image, 50, axis=1)
    assert dino.frames_hash([sharp_image]) == dino.frames_hash([sharp_image.copy()])
    assert dino.frames_hash([sharp_image]) != dino.frames_hash([other])


def test_identical_frames_are_duplicates():
    emb = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), 5, axis=0)
    mask = similarity.duplicates(emb, thr=0.97)
    assert mask.tolist() == [False, True, True, True, True]
    assert similarity.diversity(emb) == pytest.approx(0.0, abs=1e-6)


def test_random_frames_are_unique():
    rng = np.random.default_rng(0)
    emb = dino.l2_normalise(rng.normal(size=(20, 64)).astype(np.float32))
    assert similarity.duplicates(emb, thr=0.97).sum() == 0
    assert len(similarity.unique_views(emb, thr=0.97)) == 20
    assert similarity.diversity(emb) > 0.5


def test_slow_pan_keeps_some_unique_views():
    """A gradual drift must not count as 80 duplicates in a row."""
    base = np.zeros((40, 8), dtype=np.float32)
    for i in range(40):
        base[i, 0] = 1.0
        base[i, 1] = i * 0.05
    emb = dino.l2_normalise(base)
    mask = similarity.duplicates(emb, thr=0.995)
    assert 0 < mask.sum() < 40
    assert len(similarity.unique_views(emb, thr=0.995)) > 1


def test_similarity_to_previous():
    emb = dino.l2_normalise(np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
    sims = similarity.similarity_to_previous(emb)
    assert sims[0] == 0.0
    assert sims[1] == pytest.approx(1.0, abs=1e-6)
    assert sims[2] == pytest.approx(0.0, abs=1e-6)


def test_nearest_neighbour_pairs_links_similar_photos():
    emb = dino.l2_normalise(
        np.array([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0.99, 0.1]], dtype=np.float32)
    )
    pairs = similarity.nearest_neighbour_pairs(emb)
    assert (0, 1) in pairs and (2, 3) in pairs


def test_embed_capture_uses_the_cache(monkeypatch, tmp_path, sharp_image):
    """The second call must not touch the model."""
    frames = [sharp_image, np.roll(sharp_image, 30, axis=1)]
    calls = {"n": 0}

    embedder = dino.DinoEmbedder(cache_dir=tmp_path)

    def fake_embed(self, frames, batch_size=8, max_frames=None):
        calls["n"] += 1
        return dino.l2_normalise(
            np.random.default_rng(calls["n"]).normal(size=(len(frames), dino.EMBED_DIM)).astype(
                np.float32
            )
        )

    monkeypatch.setattr(dino.DinoEmbedder, "embed", fake_embed, raising=True)
    monkeypatch.setattr(dino.DinoEmbedder, "model", property(lambda self: object()))

    emb1, idx1 = embedder.embed_capture("cap", frames)
    emb2, idx2 = embedder.embed_capture("cap", frames)
    assert calls["n"] == 1  # second call came from the cache
    assert np.allclose(emb1, emb2)
    assert idx1.tolist() == idx2.tolist() == [0, 1]
    assert (tmp_path / "cap.npz").is_file()


def test_cache_is_invalidated_when_frames_change(monkeypatch, tmp_path, sharp_image):
    calls = {"n": 0}

    def fake_embed(self, frames, batch_size=8, max_frames=None):
        calls["n"] += 1
        return np.zeros((len(frames), dino.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr(dino.DinoEmbedder, "embed", fake_embed, raising=True)
    embedder = dino.DinoEmbedder(cache_dir=tmp_path)
    embedder.embed_capture("cap", [sharp_image])
    embedder.embed_capture("cap", [np.roll(sharp_image, 70, axis=1)])
    assert calls["n"] == 2


def test_pick_device_respects_explicit_choice():
    assert dino.pick_device("cpu") == "cpu"
