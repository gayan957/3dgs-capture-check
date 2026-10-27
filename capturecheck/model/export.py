"""Export a trained run to ONNX, and check the exported model still agrees.

ONNX Runtime on CPU is several times faster than PyTorch for a model this small,
which matters when the whole check has a 60-second budget.

The *whole* ensemble goes into the graph, averaged inside it. Exporting only the
first member used to look like a fair trade for speed, but it meant
`capturecheck check`, which prefers ONNX, scored with one model of five while
everything else described a 5-seed ensemble: on the demo test set the two
disagreed by up to 0.18 on a 0-1 score and ranked the captures differently
(Spearman 0.67 against 0.69). No export check could catch that, because it
compared ONNX against the one member it had exported. Five models cost about
5 ms against a 60-second budget, so there is nothing to trade.
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
    """Plain tensors in, plain tensors out: ONNX cannot export a dataclass.

    Holds the whole ensemble and averages it, the same reduction
    :meth:`RunPredictor.predict` applies on the PyTorch path, so the two
    backends answer with the same number.
    """

    def __init__(self, models: torch.nn.Module | list[torch.nn.Module]) -> None:
        super().__init__()
        if isinstance(models, torch.nn.Module):
            models = [models]
        self.models = torch.nn.ModuleList(models)

    def forward(self, frames, pairs, global_vec, frame_mask, pair_mask):
        scores, problems, attentions = [], [], []
        for model in self.models:
            out = model(
                frames, pairs, global_vec,
                frame_mask=frame_mask, pair_mask=pair_mask, return_attention=True,
            )
            scores.append(out.score)
            problems.append(out.problems)
            attentions.append(
                out.attention
                if out.attention is not None
                else torch.zeros_like(frames[:, :, 0])
            )
        score = torch.stack(scores).mean(dim=0)
        problem = torch.stack(problems).mean(dim=0)
        attention = torch.stack(attentions).mean(dim=0)
        return score, problem, attention


def _dummy_sample(n_frames: int = 8, n_pairs: int = 4, seed: int = 0) -> CaptureSample:
    rng = np.random.default_rng(seed)
    return CaptureSample(
        capture_id="dummy",
        scene_id="dummy",
        frames=rng.normal(size=(n_frames, FRAME_DIM)).astype(np.float32),
        pairs=rng.normal(size=(n_pairs, PAIR_DIM)).astype(np.float32),
        global_vec=rng.normal(size=GLOBAL_DIM).astype(np.float32),
        problems=np.zeros(1, dtype=np.float32),
    )


def _inputs(predictor, n_frames: int, n_pairs: int, seed: int = 0) -> tuple:
    sample = _dummy_sample(n_frames, n_pairs, seed)
    batch = collate([predictor.stats.normalise(sample)])
    return (batch.frames, batch.pairs, batch.global_vec, batch.frame_mask, batch.pair_mask)


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
    models = predictor.models
    for model in models:
        model.eval()
    wrapper = ExportWrapper(models)
    wrapper.eval()
    log.info("exporting all %d ensemble members, averaged in the graph", len(models))

    inputs = _inputs(predictor, n_frames=8, n_pairs=4)
    target = Path(out_path) if out_path else path / "model.onnx"

    # The tracing exporter freezes the attention reshapes at whatever sequence
    # length it saw, so an export traced on 8 frames fails on a real capture's
    # 80. The dynamo exporter keeps the length symbolic; it needs `onnxscript`.
    batch = torch.export.Dim("batch", min=1, max=64)
    n_frames = torch.export.Dim("n_frames", min=2, max=4096)
    n_pairs = torch.export.Dim("n_pairs", min=2, max=4096)
    try:
        torch.onnx.export(
            wrapper,
            inputs,
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
            opset_version=opset,
            dynamo=True,
        )
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "ONNX export needs the `onnxscript` package "
            "(pip install onnxscript). PyTorch scoring works without it."
        ) from exc
    log.info("exported %s", target)

    # Verify at the traced size *and* at a realistic one: exporting at one
    # sequence length and running at another is exactly how this breaks.
    verify_export(target, wrapper, inputs, label="8 frames")
    verify_export(
        target, wrapper, _inputs(predictor, n_frames=80, n_pairs=40, seed=1), label="80 frames"
    )
    return target


def verify_export(
    onnx_path: str | Path, wrapper: torch.nn.Module, inputs: tuple, label: str = ""
) -> None:
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

    where = f" at {label}" if label else ""
    for name, want, got in zip(("score", "problems", "attention"), expected, actual, strict=True):
        difference = float(np.max(np.abs(want.numpy() - np.asarray(got))))
        if difference > TOLERANCE:
            raise AssertionError(
                f"ONNX output `{name}`{where} differs from PyTorch by {difference:.2e} "
                f"(tolerance {TOLERANCE:.0e})"
            )
        log.info("ONNX `%s`%s matches PyTorch (max difference %.2e)", name, where, difference)
