"""Score one capture with a saved run.

The ensemble's members are averaged. Evidence comes from two places: the rules
(which point at a time range or a photo name) and the model's attention over the
frame tokens (which points at the frames it actually used).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from capturecheck.config import Config, get_config
from capturecheck.labeling import schema
from capturecheck.model.batching import collate
from capturecheck.model.data import CaptureSample, FeatureStats, _text, build_sample
from capturecheck.model.net import CaptureQualityNet
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Prediction:
    """What the model says about one capture."""

    capture_id: str
    score: float
    problems: dict[str, float] = field(default_factory=dict)
    attention: np.ndarray | None = None
    top_frames: list[int] = field(default_factory=list)
    run_id: str | None = None
    backend: str = "torch"
    score_per_seed: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "capture_id": self.capture_id,
            "score": round(float(self.score), 4),
            "problems": {k: round(float(v), 4) for k, v in self.problems.items()},
            "top_frames": list(self.top_frames),
            "run_id": self.run_id,
            "backend": self.backend,
        }


class RunPredictor:
    """A loaded run: the ensemble, its normalisation, and how to score with it."""

    def __init__(self, run_dir: str | Path, prefer_onnx: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.stats = FeatureStats.load(self.run_dir / "feature_stats.json")
        self._session = None
        self._models: list[CaptureQualityNet] | None = None
        self.backend = "torch"

        onnx_path = self.run_dir / "model.onnx"
        if prefer_onnx and onnx_path.is_file():
            try:
                import onnxruntime

                self._session = onnxruntime.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
                self.backend = "onnx"
                log.info("scoring with %s (ONNX)", self.run_id)
            except Exception as exc:  # pragma: no cover - falls back to torch
                log.warning("cannot use %s (%s); falling back to PyTorch", onnx_path.name, exc)

    @property
    def models(self) -> list[CaptureQualityNet]:
        if self._models is None:
            payload = torch.load(self.run_dir / "model.pt", map_location="cpu", weights_only=False)
            settings = payload.get("model", {})
            models = []
            for state in payload["states"]:
                model = CaptureQualityNet(
                    d_model=settings.get("d_model", 128),
                    layers=settings.get("layers", 2),
                    heads=settings.get("heads", 4),
                    dropout=settings.get("dropout", 0.3),
                )
                model.load_state_dict(state)
                model.eval()
                models.append(model)
            self._models = models
            log.info("scoring with %s (%d-model ensemble)", self.run_id, len(models))
        return self._models

    @torch.no_grad()
    def predict(self, sample: CaptureSample) -> Prediction:
        batch = collate([self.stats.normalise(sample)])

        if self._session is not None:
            inputs = {
                "frames": batch.frames.numpy(),
                "pairs": batch.pairs.numpy(),
                "global_vec": batch.global_vec.numpy(),
                "frame_mask": batch.frame_mask.numpy(),
                "pair_mask": batch.pair_mask.numpy(),
            }
            names = {o.name for o in self._session.get_outputs()}
            outputs = self._session.run(None, inputs)
            lookup = dict(zip([o.name for o in self._session.get_outputs()], outputs, strict=True))
            score = float(lookup["score"][0])
            problem_logits = np.asarray(lookup["problems"][0], dtype=np.float64)
            attention = (
                np.asarray(lookup["attention"][0], dtype=np.float64)
                if "attention" in names
                else None
            )
            scores = [score]
        else:
            scores, problem_logit_list, attentions = [], [], []
            for model in self.models:
                out = model(
                    batch.frames, batch.pairs, batch.global_vec,
                    frame_mask=batch.frame_mask, pair_mask=batch.pair_mask,
                    return_attention=True,
                )
                scores.append(float(out.score[0]))
                problem_logit_list.append(out.problems[0].numpy())
                if out.attention is not None:
                    attentions.append(out.attention[0].numpy())
            score = float(np.mean(scores))
            problem_logits = np.mean(problem_logit_list, axis=0)
            attention = np.mean(attentions, axis=0) if attentions else None

        probabilities = 1.0 / (1.0 + np.exp(-problem_logits))
        problems = {
            name: float(probabilities[i]) for i, name in enumerate(schema.PROBLEM_LABELS)
        }

        top_frames: list[int] = []
        if attention is not None and sample.frames.shape[0]:
            # Token 0 is the global summary; the frame tokens follow it.
            frame_attention = attention[1 : 1 + sample.frames.shape[0]]
            if frame_attention.size:
                order = np.argsort(-frame_attention)[:5]
                top_frames = [int(i) for i in order]

        return Prediction(
            capture_id=sample.capture_id,
            score=score,
            problems=problems,
            attention=attention,
            top_frames=top_frames,
            run_id=self.run_id,
            backend=self.backend,
            score_per_seed=[float(s) for s in scores],
        )


_CACHE: dict[str, RunPredictor] = {}


def load_predictor(
    run_dir: str | Path | None = None, prefer_onnx: bool = True
) -> RunPredictor | None:
    """Load a run (the newest by default), or ``None`` when none has been trained."""
    from capturecheck.model.train import latest_run

    path = Path(run_dir) if run_dir else latest_run()
    if path is None or not (Path(path) / "model.pt").is_file():
        return None
    key = f"{path}:{prefer_onnx}"
    if key not in _CACHE:
        _CACHE[key] = RunPredictor(path, prefer_onnx=prefer_onnx)
    return _CACHE[key]


def sample_from_result(result, cfg: Config | None = None) -> CaptureSample:
    """Build a model sample straight from a :class:`LabelResult`.

    Uses the in-memory measurements when the pipeline kept them, so a single
    ``capturecheck check`` never measures a capture twice.
    """
    cfg = cfg or get_config()
    measurements = result.measurements
    row = dict(result.video_row)

    if measurements is not None:
        grid = measurements.coverage.grid if measurements.coverage is not None else None
        return build_sample(
            capture_id=result.capture_id,
            scene_id=_text(row.get("scene_id"), result.capture_id),
            frame_df=result.frame_rows,
            pair_df=measurements.pair_df,
            video_row=row,
            emb=measurements.emb,
            emb_idx=measurements.emb_idx,
            coverage_grid=grid,
            cfg=cfg,
        )

    import pandas as pd

    from capturecheck.features.dino import EMBED_DIM

    return build_sample(
        capture_id=result.capture_id,
        scene_id=_text(row.get("scene_id"), result.capture_id),
        frame_df=result.frame_rows,
        pair_df=pd.DataFrame(),
        video_row=row,
        emb=np.zeros((0, EMBED_DIM), dtype=np.float32),
        emb_idx=np.zeros(0, dtype=int),
        coverage_grid=None,
        cfg=cfg,
    )


def predict_capture(result, run_dir: str | Path | None = None) -> Prediction | None:
    """Score an already-labelled capture; ``None`` when no run has been trained."""
    predictor = load_predictor(run_dir)
    if predictor is None:
        log.info("no trained run found: reporting rules and evidence only")
        return None
    return predictor.predict(sample_from_result(result))


def predict_path(path: str | Path, run_dir: str | Path | None = None, use_dino: bool = True):
    """Measure and score one capture from scratch. Returns ``(result, prediction)``."""
    from capturecheck.labeling.pipeline import label_capture

    result = label_capture(path, use_dino=use_dino, keep_measurements=True)
    return result, predict_capture(result, run_dir=run_dir)
