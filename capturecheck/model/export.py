"""Export a trained run to ONNX, and check the exported model still agrees.

ONNX Runtime on CPU is several times faster than PyTorch for a model this small,
which matters when the whole check has a 60-second budget. Only the first member
of the ensemble is exported, so the export is checked against *that* member, not
against the averaged ensemble.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from capturecheck.model.batching import collate
from capturecheck.model.data import FRAME_DIM, GLOBAL_DIM, PAIR_DIM, CaptureSample
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

TOLERANCE = 1e-4


class ExportWrapper(torch.nn.Module):
    """Plain tensors in, plain tensors out: ONNX cannot export a dataclass."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames, pairs, global_vec, frame_mask, pair_mask):
        out = self.model(
            frames, pairs, global_vec,
            frame_mask=frame_mask, pair_mask=pair_mask, return_attention=True,
        )
        attention = out.attention
        if attention is None:  # pragma: no cover - return_attention is always True here
            attention = torch.zeros_like(frames[:, :, 0])
        return out.score, out.problems, attention


def _dummy_sample(n_frames: int = 8, n_pairs: int = 4) -> CaptureSample:
    rng = np.random.default_rng(0)
    return CaptureSample(
        capture_id="dummy",
        scene_id="dummy",
        frames=rng.normal(size=(n_frames, FRAME_DIM)).astype(np.float32),
        pairs=rng.normal(size=(n_pairs, PAIR_DIM)).astype(np.float32),
        global_vec=rng.normal(size=GLOBAL_DIM).astype(np.float32),
        problems=np.zeros(1, dtype=np.float32),
    )


def export_run(
    run_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    opset: int = 17,
) -> Path:
    """Write ``model.onnx`` next to ``model.pt`` and verify the outputs match."""
    from capturecheck.model.predict import RunPredictor
    from capturecheck.model.train import latest_run

    path = Path(run_dir) if run_dir else latest_run()
    if path is None:
        raise RuntimeError("no trained run found; run `capturecheck train` first")

    predictor = RunPredictor(path, prefer_onnx=False)
    model = predictor.models[0]
    model.eval()
    wrapper = ExportWrapper(model)
    wrapper.eval()

    sample = _dummy_sample()
    batch = collate([predictor.stats.normalise(sample)])
    inputs = (
        batch.frames, batch.pairs, batch.global_vec, batch.frame_mask, batch.pair_mask,
    )

    target = Path(out_path) if out_path else path / "model.onnx"
    torch.onnx.export(
        wrapper,
        inputs,
        str(target),
        input_names=["frames", "pairs", "global_vec", "frame_mask", "pair_mask"],
        output_names=["score", "problems", "attention"],
        dynamic_axes={
            "frames": {0: "batch", 1: "n_frames"},
            "pairs": {0: "batch", 1: "n_pairs"},
            "global_vec": {0: "batch"},
            "frame_mask": {0: "batch", 1: "n_frames"},
            "pair_mask": {0: "batch", 1: "n_pairs"},
            "score": {0: "batch"},
            "problems": {0: "batch"},
            "attention": {0: "batch", 1: "n_tokens"},
        },
        opset_version=opset,
        dynamo=False,
    )
    log.info("exported %s", target)

    verify_export(target, wrapper, inputs)
    return target


def verify_export(onnx_path: str | Path, wrapper: torch.nn.Module, inputs: tuple) -> None:
    """Fail loudly if ONNX and PyTorch disagree by more than ``TOLERANCE``."""
    import onnxruntime

    with torch.no_grad():
        expected = wrapper(*inputs)

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    feed = {
        "frames": inputs[0].numpy(),
        "pairs": inputs[1].numpy(),
        "global_vec": inputs[2].numpy(),
        "frame_mask": inputs[3].numpy(),
        "pair_mask": inputs[4].numpy(),
    }
    actual = session.run(None, feed)

    for name, want, got in zip(("score", "problems", "attention"), expected, actual, strict=True):
        difference = float(np.max(np.abs(want.numpy() - np.asarray(got))))
        if difference > TOLERANCE:
            raise AssertionError(
                f"ONNX output `{name}` differs from PyTorch by {difference:.2e} "
                f"(tolerance {TOLERANCE:.0e})"
            )
        log.info("ONNX `%s` matches PyTorch (max difference %.2e)", name, difference)
