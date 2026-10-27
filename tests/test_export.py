"""ONNX export: it must agree with PyTorch at a sequence length it never saw."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from capturecheck.model.data import FRAME_DIM, GLOBAL_DIM, PAIR_DIM, FeatureStats
from capturecheck.model.export import ExportWrapper, _dummy_sample, _inputs, verify_export
from capturecheck.model.net import CaptureQualityNet

onnxruntime = pytest.importorskip("onnxruntime")


class FakePredictor:
    """Just enough of RunPredictor for the export helpers."""

    def __init__(self):
        self.stats = FeatureStats.fit([_dummy_sample(12, 6, seed=i) for i in range(4)])


def test_export_wrapper_returns_plain_tensors():
    net = CaptureQualityNet(d_model=32, layers=1, heads=2, dropout=0.0).eval()
    wrapper = ExportWrapper(net).eval()
    with torch.no_grad():
        score, problems, attention = wrapper(
            torch.randn(1, 6, FRAME_DIM),
            torch.randn(1, 3, PAIR_DIM),
            torch.randn(1, GLOBAL_DIM),
            torch.zeros(1, 6, dtype=torch.bool),
            torch.zeros(1, 3, dtype=torch.bool),
        )
    assert score.shape == (1,)
    assert problems.shape[0] == 1
    assert attention.shape == (1, 1 + 6 + 3)


@pytest.mark.slow
def test_exported_model_matches_at_an_unseen_length(tmp_path):
    """The bug this guards: tracing froze the attention reshape at 8 frames."""
    pytest.importorskip("onnxscript")

    net = CaptureQualityNet(d_model=32, layers=1, heads=2, dropout=0.0).eval()
    wrapper = ExportWrapper(net).eval()
    predictor = FakePredictor()

    small = _inputs(predictor, n_frames=8, n_pairs=4)
    big = _inputs(predictor, n_frames=60, n_pairs=30, seed=5)

    target = tmp_path / "model.onnx"
    batch = torch.export.Dim("batch", min=1, max=8)
    n_frames = torch.export.Dim("n_frames", min=2, max=256)
    n_pairs = torch.export.Dim("n_pairs", min=2, max=256)
    torch.onnx.export(
        wrapper,
        small,
        str(target),
        input_names=["frames", "pairs", "global_vec", "frame_mask", "pair_mask"],
        output_names=["score", "problems", "attention"],
        dynamic_shapes=(
            {0: batch, 1: n_frames},
            {0: batch, 1: n_pairs},
            {0: batch},
            {0: batch, 1: n_frames},
            {0: batch, 1: n_pairs},
        ),
        opset_version=17,
        dynamo=True,
    )

    # Exported at 8 frames, verified at 8 and at 60.
    verify_export(target, wrapper, small, label="8 frames")
    verify_export(target, wrapper, big, label="60 frames")


def test_verify_export_rejects_a_mismatch(tmp_path, monkeypatch):
    """A silent disagreement must fail loudly, not pass."""
    pytest.importorskip("onnxscript")

    net = CaptureQualityNet(d_model=32, layers=1, heads=2, dropout=0.0).eval()
    wrapper = ExportWrapper(net).eval()
    predictor = FakePredictor()
    inputs = _inputs(predictor, n_frames=8, n_pairs=4)

    target = tmp_path / "model.onnx"
    torch.onnx.export(
        wrapper, inputs, str(target),
        input_names=["frames", "pairs", "global_vec", "frame_mask", "pair_mask"],
        output_names=["score", "problems", "attention"],
        opset_version=17, dynamo=True,
    )

    class Wrong(torch.nn.Module):
        def forward(self, *args):
            score, problems, attention = wrapper(*args)
            return score + 1.0, problems, attention

    with pytest.raises(AssertionError, match="differs from PyTorch"):
        verify_export(target, Wrong().eval(), inputs)


def test_tolerance_is_tight():
    from capturecheck.model.export import TOLERANCE

    assert TOLERANCE <= 1e-4


def test_dummy_sample_shapes():
    sample = _dummy_sample(n_frames=5, n_pairs=3)
    assert sample.frames.shape == (5, FRAME_DIM)
    assert sample.pairs.shape == (3, PAIR_DIM)
    assert sample.global_vec.shape == (GLOBAL_DIM,)
    assert np.isfinite(sample.frames).all()
