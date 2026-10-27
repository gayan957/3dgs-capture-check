"""The small trained model: shapes, losses, batching and a real training step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from capturecheck.labeling import schema
from capturecheck.model import losses as loss_fns
from capturecheck.model.batching import collate, scene_batches
from capturecheck.model.data import (
    FRAME_DIM,
    GLOBAL_DIM,
    PAIR_DIM,
    PRETEXT_TYPES,
    CaptureSample,
    FeatureDegrader,
    FeatureStats,
    build_sample,
)
from capturecheck.model.net import CaptureQualityNet, count_parameters


def make_sample(
    capture_id: str = "cap",
    scene_id: str = "scene",
    n_frames: int = 12,
    n_pairs: int = 6,
    psnr: float = float("nan"),
    degrade_type: str = "none",
    degrade_strength: str = "",
    source: str = "real",
    seed: int = 0,
) -> CaptureSample:
    rng = np.random.default_rng(seed)
    return CaptureSample(
        capture_id=capture_id,
        scene_id=scene_id,
        frames=rng.normal(size=(n_frames, FRAME_DIM)).astype(np.float32),
        pairs=rng.normal(size=(n_pairs, PAIR_DIM)).astype(np.float32),
        global_vec=rng.normal(size=GLOBAL_DIM).astype(np.float32),
        problems=np.zeros(schema.N_PROBLEMS, dtype=np.float32),
        psnr=psnr,
        source=source,
        degrade_type=degrade_type,
        degrade_strength=degrade_strength,
    )


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def test_forward_pass_shapes():
    net = CaptureQualityNet()
    frames = torch.randn(3, 20, FRAME_DIM)
    pairs = torch.randn(3, 8, PAIR_DIM)
    global_vec = torch.randn(3, GLOBAL_DIM)
    out = net(frames, pairs, global_vec, return_attention=True)

    assert out.score.shape == (3,)
    assert torch.all((out.score >= 0) & (out.score <= 1))
    assert out.problems.shape == (3, schema.N_PROBLEMS)
    assert out.pretext_type.shape == (3, len(PRETEXT_TYPES))
    assert out.pretext_strength.shape == (3,)
    assert out.attention.shape == (3, 1 + 20 + 8)
    assert torch.allclose(out.attention.sum(dim=1), torch.ones(3), atol=1e-5)


def test_padding_is_ignored():
    """A padded capture must score the same as the unpadded one."""
    net = CaptureQualityNet(dropout=0.0).eval()
    short = make_sample(n_frames=5, n_pairs=2)
    long = make_sample("other", n_frames=20, n_pairs=9, seed=1)

    with torch.no_grad():
        alone = collate([short])
        score_alone = net(
            alone.frames, alone.pairs, alone.global_vec,
            frame_mask=alone.frame_mask, pair_mask=alone.pair_mask,
        ).score[0]

        together = collate([short, long])
        score_padded = net(
            together.frames, together.pairs, together.global_vec,
            frame_mask=together.frame_mask, pair_mask=together.pair_mask,
        ).score[0]

    assert float(abs(score_alone - score_padded)) < 1e-4


def test_model_is_small_enough_for_a_laptop():
    assert count_parameters(CaptureQualityNet()) < 2_000_000


def test_a_training_step_lowers_the_loss():
    torch.manual_seed(0)
    net = CaptureQualityNet(dropout=0.0)
    batch = collate([make_sample(seed=i) for i in range(4)])
    targets = torch.zeros(4, schema.N_PROBLEMS)
    targets[:2, 0] = 1.0

    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3)
    first = None
    for step in range(12):
        out = net(
            batch.frames, batch.pairs, batch.global_vec,
            frame_mask=batch.frame_mask, pair_mask=batch.pair_mask,
        )
        loss = loss_fns.problem_bce_loss(out.problems, targets)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = float(loss.detach())
    assert float(loss.detach()) < first


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def test_ranking_loss_is_zero_when_the_order_is_right():
    scores = torch.tensor([0.9, 0.2])
    better = torch.tensor([0])
    worse = torch.tensor([1])
    assert float(loss_fns.pairwise_ranking_loss(scores, better, worse, margin=0.1)) == 0.0
    # The wrong way round costs something.
    assert float(loss_fns.pairwise_ranking_loss(scores, worse, better, margin=0.1)) > 0.5


def test_build_pairs_uses_psnr_across_scenes():
    psnr = torch.tensor([30.0, 24.0])
    damage = torch.tensor([0.0, 0.0])
    better, worse = loss_fns.build_pairs(["a", "b"], psnr, damage, ["real", "real"])
    assert better.tolist() == [0] and worse.tolist() == [1]


def test_build_pairs_only_ranks_damage_within_a_scene():
    psnr = torch.tensor([float("nan")] * 3)
    damage = torch.tensor([0.0, 2.0, 1.0])
    better, worse = loss_fns.build_pairs(
        ["kitchen", "kitchen", "office"], psnr, damage, ["real", "degraded", "degraded"]
    )
    pairs = set(zip(better.tolist(), worse.tolist(), strict=True))
    assert (0, 1) in pairs  # clean beats its own broken copy
    assert not {(0, 2), (2, 0), (1, 2), (2, 1)} & pairs  # never across scenes


def test_build_pairs_ranks_damage_strength():
    psnr = torch.tensor([float("nan")] * 2)
    damage = torch.tensor([1.0, 3.0])  # light vs heavy
    better, worse = loss_fns.build_pairs(["s", "s"], psnr, damage, ["degraded"] * 2)
    assert better.tolist() == [0] and worse.tolist() == [1]


def test_positive_weights_favour_rare_labels():
    targets = torch.zeros(20, 3)
    targets[:10, 0] = 1.0  # balanced
    targets[:1, 1] = 1.0  # rare
    weights = loss_fns.positive_weights(targets)
    assert weights[1] > weights[0]
    assert float(weights.max()) <= 10.0


def test_pretext_loss_runs():
    logits = torch.randn(4, len(PRETEXT_TYPES))
    types = torch.tensor([0, 1, 2, 3])
    loss = loss_fns.pretext_loss(logits, types, torch.rand(4), torch.rand(4))
    assert float(loss) > 0


# --------------------------------------------------------------------------- #
# Batching and sample building
# --------------------------------------------------------------------------- #
def test_collate_pads_and_masks():
    batch = collate([make_sample(n_frames=5, n_pairs=2), make_sample(n_frames=9, n_pairs=4)])
    assert batch.frames.shape == (2, 9, FRAME_DIM)
    assert batch.frame_mask[0, 5:].all() and not batch.frame_mask[0, :5].any()
    assert not batch.frame_mask[1].any()
    assert batch.pairs.shape == (2, 4, PAIR_DIM)


def test_scene_batches_always_contain_a_rankable_pair():
    samples = [
        make_sample("clean", "kitchen"),
        make_sample("blur_light", "kitchen", degrade_type="blur",
                    degrade_strength="light", source="degraded"),
        make_sample("blur_heavy", "kitchen", degrade_type="blur",
                    degrade_strength="heavy", source="degraded"),
        make_sample("other", "office"),
    ]
    batches = scene_batches(samples, batch_size=4, rng=np.random.default_rng(0))
    assert batches
    for group in batches:
        batch = collate(group)
        better, worse = loss_fns.build_pairs(
            batch.scene_ids, batch.psnr, batch.damage, batch.sources
        )
        assert better.numel() > 0


def test_damage_rank_orders_strengths():
    clean = make_sample("c")
    light = make_sample("l", degrade_type="blur", degrade_strength="light", source="degraded")
    heavy = make_sample("h", degrade_type="blur", degrade_strength="heavy", source="degraded")
    assert clean.damage_rank == 0.0 < light.damage_rank < heavy.damage_rank


def test_build_sample_from_measurements(tiny_bundle):
    from capturecheck.io.load import load_capture
    from capturecheck.labeling.pipeline import measure_capture

    capture = load_capture(tiny_bundle)
    measurements = measure_capture(capture, use_dino=False)
    sample = build_sample(
        capture_id="kitchen_photos",
        scene_id="kitchen",
        frame_df=measurements.frame_df,
        pair_df=measurements.pair_df,
        video_row={"input_type": "images", "n_frames": capture.n_frames, "motion_blur": 1},
        emb=measurements.emb,
        emb_idx=measurements.emb_idx,
        coverage_grid=measurements.coverage.grid if measurements.coverage else None,
    )
    assert sample.frames.shape[1] == FRAME_DIM
    assert sample.pairs.shape[1] == PAIR_DIM
    assert sample.global_vec.shape == (GLOBAL_DIM,)
    assert sample.problems[schema.PROBLEM_LABELS.index("motion_blur")] == 1.0
    assert sample.frames.shape[0] <= 80


# --------------------------------------------------------------------------- #
# Normalisation and feature-level damage
# --------------------------------------------------------------------------- #
def test_feature_stats_normalise_and_roundtrip(tmp_path):
    samples = [make_sample(f"c{i}", seed=i) for i in range(5)]
    stats = FeatureStats.fit(samples)
    normalised = stats.normalise(samples[0])
    assert np.isfinite(normalised.frames).all()
    assert np.isfinite(normalised.global_vec).all()

    path = stats.save(tmp_path / "feature_stats.json")
    back = FeatureStats.load(path)
    assert np.allclose(back.frame_mean, stats.frame_mean)
    assert np.allclose(back.global_std, stats.global_std)


def test_feature_stats_never_divide_by_zero():
    identical = [make_sample("a", seed=0), make_sample("b", seed=0)]
    stats = FeatureStats.fit(identical)
    assert (stats.frame_std > 0).all()
    assert np.isfinite(stats.normalise(identical[0]).frames).all()


@pytest.mark.parametrize("kind", [k for k in PRETEXT_TYPES if k != "none"])
def test_every_feature_degradation_sets_its_label(kind):
    from capturecheck.model.data import PRETEXT_TO_LABEL

    degrader = FeatureDegrader(seed=0)
    clean = make_sample(n_frames=30, n_pairs=12)
    clean.input_type = "images"  # so drop_gap is allowed
    damaged = degrader.apply(clean, kind=kind)

    assert damaged.pretext_type == PRETEXT_TYPES.index(kind)
    assert 0.0 < damaged.strength <= 1.0
    label = PRETEXT_TO_LABEL[kind]
    assert damaged.sample.problems[schema.PROBLEM_LABELS.index(label)] == 1.0
    assert damaged.sample.frames.shape[1] == FRAME_DIM
    assert damaged.sample.frames.shape[0] >= 2
    assert damaged.sample.source == "degraded"


def test_feature_blur_lowers_the_sharpness_numbers():
    from capturecheck.features.dino import EMBED_DIM
    from capturecheck.model.data import FRAME_FEATURES

    degrader = FeatureDegrader(seed=3)
    clean = make_sample(n_frames=20)
    clean.frames[:, EMBED_DIM + FRAME_FEATURES.index("sharpness")] = 500.0
    damaged = degrader.apply(clean, kind="blur")
    column = damaged.sample.frames[:, EMBED_DIM + FRAME_FEATURES.index("sharpness")]
    assert column.min() < 500.0


def test_feature_subsample_removes_tokens():
    degrader = FeatureDegrader(seed=1)
    clean = make_sample(n_frames=40)
    damaged = degrader.apply(clean, kind="subsample")
    assert damaged.sample.frames.shape[0] < clean.frames.shape[0]


def test_video_captures_never_get_the_photo_only_damage():
    degrader = FeatureDegrader(seed=2)
    clean = make_sample(n_frames=30)
    clean.input_type = "video"
    kinds = {degrader.apply(clean).sample.degrade_type for _ in range(60)}
    assert "drop_gap" not in kinds
