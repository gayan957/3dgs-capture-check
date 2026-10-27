"""Evaluate a run: Spearman against PSNR, precision per problem, and timing.

Everything is reported with a bootstrap confidence interval, because a test set
of a few dozen captures cannot support a bare correlation figure. Where a
baseline matches the model, the report says so.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from capturecheck.config import Config, get_config
from capturecheck.labeling import schema
from capturecheck.model import baselines as baseline_mod
from capturecheck.model.data import CaptureSample, samples_from_labels
from capturecheck.model.predict import load_predictor
from capturecheck.model.splits import read_splits
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ignoring pairs where either value is missing."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    from scipy.stats import spearmanr

    value = spearmanr(a[keep], b[keep]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the Spearman correlation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 4:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.size, a.size)
        if np.unique(a[idx]).size < 2 or np.unique(b[idx]).size < 2:
            continue
        value = spearman(a[idx], b[idx])
        if np.isfinite(value):
            values.append(value)
    if len(values) < 20:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def precision_recall_f1(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if not (np.isfinite(precision) and np.isfinite(recall)) or precision + recall == 0:
        return precision, recall, float("nan")
    return precision, recall, 2 * precision * recall / (precision + recall)


@dataclass
class MethodScore:
    name: str
    description: str
    spearman_all: float
    ci_all: tuple[float, float]
    spearman_video: float = float("nan")
    spearman_images: float = float("nan")
    n: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "spearman": self.spearman_all,
            "ci95": list(self.ci_all),
            "spearman_video": self.spearman_video,
            "spearman_images": self.spearman_images,
            "n": self.n,
            "note": self.note,
        }


@dataclass
class EvaluationResult:
    run_id: str
    run_dir: Path
    methods: list[MethodScore] = field(default_factory=list)
    problems: pd.DataFrame = field(default_factory=pd.DataFrame)
    timing: dict = field(default_factory=dict)
    report_path: Path | None = None
    n_test: int = 0
    n_with_psnr: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def model_score(self) -> MethodScore | None:
        for method in self.methods:
            if method.name == "capturecheck":
                return method
        return None

    def summary(self) -> str:
        lines = [f"run {self.run_id}: {self.n_test} test captures, {self.n_with_psnr} with PSNR"]
        for method in sorted(self.methods, key=lambda m: -(m.spearman_all or -9)):
            lo, hi = method.ci_all
            interval = f" [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else ""
            lines.append(f"  {method.name:<20} Spearman {method.spearman_all:>6.3f}{interval}")
        if not self.problems.empty:
            best = self.problems.sort_values("precision", ascending=False).head(4)
            lines.append("  problem precision: " + ", ".join(
                f"{row['label']} {row['precision']:.2f}" for _, row in best.iterrows()
                if np.isfinite(row["precision"])
            ))
        if self.timing:
            lines.append(
                f"  time per capture: {self.timing.get('seconds_per_capture', float('nan')):.1f} s"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _subset_spearman(
    samples: list[CaptureSample], scores: np.ndarray, psnr: np.ndarray, input_type: str
) -> float:
    mask = np.array([s.input_type == input_type for s in samples])
    if mask.sum() < 3:
        return float("nan")
    return spearman(scores[mask], psnr[mask])


def evaluate_run(
    run_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    cfg: Config | None = None,
    colmap_db: str | Path | None = None,
    time_captures: int = 3,
) -> EvaluationResult:
    """Score the test scenes with the model and every baseline, and write the report."""
    from capturecheck.model.train import latest_run

    cfg = cfg or get_config()
    path = Path(run_dir) if run_dir else latest_run()
    if path is None:
        raise RuntimeError("no trained run found; run `capturecheck train` first")

    predictor = load_predictor(path)
    if predictor is None:
        raise RuntimeError(f"{path} does not hold a model")

    train_scenes, test_scenes = read_splits()
    samples = samples_from_labels(cfg=cfg)
    train_samples = [s for s in samples if s.scene_id in set(train_scenes)]
    test_samples = [s for s in samples if s.scene_id in set(test_scenes)]
    if len(test_samples) < 3:
        raise RuntimeError(
            f"only {len(test_samples)} test captures; need at least 3 to evaluate"
        )

    notes: list[str] = []
    psnr = np.array([s.psnr for s in test_samples], dtype=float)
    n_with_psnr = int(np.isfinite(psnr).sum())
    if n_with_psnr < 3:
        # Without PSNR the only order we know is clean-versus-damaged.
        psnr = np.array([-s.damage_rank for s in test_samples], dtype=float)
        notes.append(
            "Fewer than 3 test captures have a PSNR target, so the correlation is "
            "against damage severity (clean > light > medium > heavy), not PSNR."
        )

    predictions = [predictor.predict(s) for s in test_samples]
    model_scores = np.array([p.score for p in predictions], dtype=float)

    methods = [
        MethodScore(
            name="capturecheck",
            description=f"CaptureCheck model ({predictor.run_id}, {predictor.backend})",
            spearman_all=spearman(model_scores, psnr),
            ci_all=bootstrap_ci(model_scores, psnr, n_boot=cfg.train.eval.bootstrap),
            spearman_video=_subset_spearman(test_samples, model_scores, psnr, "video"),
            spearman_images=_subset_spearman(test_samples, model_scores, psnr, "images"),
            n=len(test_samples),
        )
    ]

    for baseline in baseline_mod.all_baselines(train_samples, test_samples, colmap_db):
        methods.append(
            MethodScore(
                name=baseline.name,
                description=baseline.description,
                spearman_all=spearman(baseline.scores, psnr),
                ci_all=bootstrap_ci(baseline.scores, psnr, n_boot=cfg.train.eval.bootstrap),
                spearman_video=_subset_spearman(test_samples, baseline.scores, psnr, "video"),
                spearman_images=_subset_spearman(test_samples, baseline.scores, psnr, "images"),
                n=len(test_samples),
                note=baseline.note,
            )
        )

    # --- per-problem precision and recall ---------------------------------- #
    rows = []
    truth_matrix = np.stack([s.problems for s in test_samples])
    for i, name in enumerate(schema.PROBLEM_LABELS):
        truth = truth_matrix[:, i].astype(int)
        predicted = np.array(
            [1 if p.problems.get(name, 0.0) >= 0.5 else 0 for p in predictions], dtype=int
        )
        precision, recall, f1 = precision_recall_f1(truth, predicted)
        rows.append(
            {
                "label": name,
                "test captures": len(truth),
                "positives": int(truth.sum()),
                "predicted": int(predicted.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    problems = pd.DataFrame(rows)

    timing = measure_timing(test_samples, predictor, cfg, n=time_captures)

    # --- honesty check: did anything simple do just as well? ---------------- #
    model = methods[0]
    for other in methods[1:]:
        if np.isfinite(other.spearman_all) and np.isfinite(model.spearman_all):
            if other.spearman_all >= model.spearman_all - 0.02:
                notes.append(
                    f"`{other.name}` ranks the test captures as well as the model "
                    f"({other.spearman_all:.2f} vs {model.spearman_all:.2f}). On this test set "
                    "the trained model is not yet earning its place."
                )

    result = EvaluationResult(
        run_id=predictor.run_id,
        run_dir=Path(path),
        methods=methods,
        problems=problems,
        timing=timing,
        n_test=len(test_samples),
        n_with_psnr=n_with_psnr,
        notes=notes,
    )

    _write_outputs(result, test_samples, model_scores, psnr, report_path, cfg)
    return result


def measure_timing(
    samples: list[CaptureSample], predictor, cfg: Config, n: int = 3
) -> dict:
    """How long scoring takes, and how long a full check took when labelling ran."""
    if not samples:
        return {}
    chosen = samples[: max(1, n)]
    start = time.perf_counter()
    for sample in chosen:
        predictor.predict(sample)
    model_seconds = (time.perf_counter() - start) / len(chosen)

    # The measured pipeline time is recorded by `capturecheck label`.
    from capturecheck.io import dataset

    video_df = dataset.read_video_labels()
    measured = float("nan")
    if "time_s_total" in video_df.columns:
        values = pd.to_numeric(video_df["time_s_total"], errors="coerce").dropna()
        measured = float(values.mean()) if len(values) else float("nan")

    return {
        "model_seconds_per_capture": round(model_seconds, 4),
        "pipeline_seconds_per_capture": round(measured, 2) if np.isfinite(measured) else None,
        "seconds_per_capture": round(
            (measured if np.isfinite(measured) else 0.0) + model_seconds, 2
        ),
        "budget_s": cfg.runtime.time_budget_s,
        "backend": predictor.backend,
    }


def _write_outputs(
    result: EvaluationResult,
    samples: list[CaptureSample],
    model_scores: np.ndarray,
    psnr: np.ndarray,
    report_path: str | Path | None,
    cfg: Config,
) -> None:
    run_dir = result.run_dir
    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    model = result.model_score
    metrics.update(
        {
            "test": {
                "n_captures": result.n_test,
                "n_with_psnr": result.n_with_psnr,
                "spearman": model.spearman_all if model else None,
                "spearman_ci95": list(model.ci_all) if model else None,
                "spearman_video": model.spearman_video if model else None,
                "spearman_images": model.spearman_images if model else None,
                "problems": result.problems.replace({np.nan: None}).to_dict("records"),
                "timing": result.timing,
                "notes": result.notes,
            }
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    baselines_payload = {
        method.name: method.as_dict() for method in result.methods if method.name != "capturecheck"
    }
    (run_dir / "baselines.json").write_text(
        json.dumps(baselines_payload, indent=2, default=float), encoding="utf-8"
    )

    figure_path = _scatter_plot(model_scores, psnr, samples, cfg, result)
    result.report_path = _write_report(result, report_path, cfg, figure_path)


def _scatter_plot(
    scores: np.ndarray,
    psnr: np.ndarray,
    samples: list[CaptureSample],
    cfg: Config,
    result: EvaluationResult,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    figures = cfg.paths.resolve("reports") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    target = figures / f"predicted_vs_psnr_{result.run_id}.png"

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=140)
    for input_type, marker in (("video", "o"), ("images", "s")):
        mask = np.array([s.input_type == input_type for s in samples])
        if mask.any():
            ax.scatter(
                psnr[mask], scores[mask], marker=marker, alpha=0.75,
                label=f"{input_type} (n={int(mask.sum())})",
            )
    ax.set_xlabel("PSNR (or damage order where PSNR is unknown)")
    ax.set_ylabel("predicted score")
    model = result.model_score
    title = "Predicted score vs quality target"
    if model and np.isfinite(model.spearman_all):
        title += f"  ·  Spearman {model.spearman_all:.2f}"
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def _fmt(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.2f}"


def _write_report(
    result: EvaluationResult,
    report_path: str | Path | None,
    cfg: Config,
    figure_path: Path | None,
) -> Path:
    target = (
        Path(report_path)
        if report_path
        else cfg.paths.resolve("reports") / "evaluation_report.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    splits_file = result.run_dir / "splits.json"
    splits = json.loads(splits_file.read_text()) if splits_file.is_file() else {}

    lines = [
        "# Evaluation report",
        "",
        f"Run `{result.run_id}` · {result.n_test} unseen test captures "
        f"({result.n_with_psnr} with a PSNR target)",
        "",
        "Test scenes were never trained on, and scenes marked `company` are test-only.",
        "",
        "## Ranking quality",
        "",
        "Spearman correlation between each method's score and the quality target, with a "
        "percentile bootstrap 95% interval.",
        "",
        "| method | Spearman (95% CI) | videos | image bundles |",
        "|---|---|---|---|",
    ]
    for method in result.methods:
        lo, hi = method.ci_all
        interval = f" ({_fmt(lo)} – {_fmt(hi)})" if np.isfinite(lo) else ""
        name = f"**{method.name}**" if method.name == "capturecheck" else f"`{method.name}`"
        lines.append(
            f"| {name} | {_fmt(method.spearman_all)}{interval} | "
            f"{_fmt(method.spearman_video)} | {_fmt(method.spearman_images)} |"
        )

    lines += ["", "## Problems found", "",
              "Precision and recall of the model's problem heads against the labels.",
              "", "| problem | test captures | positives | predicted | precision | recall | F1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in result.problems.iterrows():
        lines.append(
            f"| `{row['label']}` | {row['test captures']} | {row['positives']} | "
            f"{row['predicted']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['f1'])} |"
        )

    timing = result.timing
    lines += ["", "## Runtime", ""]
    if timing:
        lines.append(
            f"- Measuring a capture: **{timing.get('pipeline_seconds_per_capture', 'n/a')} s** "
            "(OpenCV checks + DINOv2, mean over everything labelled)"
        )
        lines.append(
            f"- Scoring it with the model: **{timing.get('model_seconds_per_capture', 'n/a')} s** "
            f"({timing.get('backend', 'torch')})"
        )
        lines.append(f"- Budget: {timing.get('budget_s')} s per capture on a laptop CPU")

    if figure_path is not None:
        lines += ["", "## Predicted score vs target", "",
                  f"![predicted vs psnr](figures/{figure_path.name})"]

    if splits:
        lines += [
            "", "## Split", "",
            f"- train scenes ({len(splits.get('train_scenes', []))}): "
            f"{', '.join(f'`{s}`' for s in splits.get('train_scenes', []))}",
            f"- test scenes ({len(splits.get('test_scenes', []))}): "
            f"{', '.join(f'`{s}`' for s in splits.get('test_scenes', []))}",
        ]

    lines += ["", "## Honest notes", ""]
    if result.notes:
        for note in result.notes:
            lines.append(f"- {note}")
    else:
        lines.append("- The model ranks the test captures better than every baseline tried.")
    lines.append(
        "- A test set of this size cannot separate methods whose intervals overlap; "
        "read the CI, not the point estimate."
    )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", target)
    return target
