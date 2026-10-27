"""Two-stage training: learn to spot damage, then learn to rank.

**Stage A (pretext).** Damaged copies of the clean captures are generated at the
feature level, by the thousand, and the model is asked what was done to them and
how strongly. No PSNR and no human labels are needed, so this stage can use far
more data than stage B and is what stops a 0.5M-parameter model from
memorising a few hundred captures.

**Stage B (score).** The score head is trained by ranking pairs whose true order
is known, with the pretext and problem heads kept on as auxiliary tasks.

Five seeds are trained and kept as an ensemble; each validates on a different
held-out scene, which is leave-one-scene-out across the ensemble.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from capturecheck.config import Config, get_config, load_config
from capturecheck.labeling import schema
from capturecheck.model import losses as loss_fns
from capturecheck.model.batching import Batch, collate, scene_batches
from capturecheck.model.data import (
    PRETEXT_TYPES,
    CaptureSample,
    FeatureDegrader,
    FeatureStats,
    clean_samples,
    samples_from_labels,
)
from capturecheck.model.net import CaptureQualityNet, count_parameters
from capturecheck.model.splits import read_splits
from capturecheck.utils.logging import get_logger
from capturecheck.utils.seed import set_seed

log = get_logger(__name__)


@dataclass
class SeedResult:
    seed: int
    val_scene: str
    val_ranking_accuracy: float
    best_epoch: int
    state: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


@dataclass
class TrainResult:
    run_dir: Path
    metrics: dict
    seeds: list[SeedResult] = field(default_factory=list)


def pick_device(cfg: Config) -> str:
    want = (cfg.runtime.device or "auto").lower()
    if want in {"cpu", "cuda", "mps"}:
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


def next_run_id(models_dir: Path) -> str:
    today = date.today().isoformat()
    existing = [p.name for p in models_dir.glob(f"{today}_run*")] if models_dir.exists() else []
    return f"{today}_run{len(existing) + 1:02d}"


def _forward(model: CaptureQualityNet, batch: Batch):
    return model(
        batch.frames, batch.pairs, batch.global_vec,
        frame_mask=batch.frame_mask, pair_mask=batch.pair_mask,
    )


def ranking_accuracy(scores: torch.Tensor, better: torch.Tensor, worse: torch.Tensor) -> float:
    """Share of known-order pairs the model puts the right way round."""
    if better.numel() == 0:
        return float("nan")
    return float((scores[better] > scores[worse]).float().mean())


# --------------------------------------------------------------------------- #
# Stage A
# --------------------------------------------------------------------------- #
def train_stage_a(
    model: CaptureQualityNet,
    clean: list[CaptureSample],
    stats: FeatureStats,
    cfg: Config,
    device: str,
    seed: int,
    history: list[dict],
) -> None:
    """Teach the model to recognise damage, from generated variants."""
    if not clean:
        log.warning("no clean captures to damage; skipping stage A")
        return

    settings = cfg.train.stage_a
    degrader = FeatureDegrader(seed=seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay
    )
    rng = np.random.default_rng(seed)
    model.train()

    for epoch in range(settings.epochs):
        total, n_batches, correct, seen = 0.0, 0, 0, 0
        for _ in range(max(1, settings.variants_per_epoch // settings.batch_size)):
            samples: list[CaptureSample] = []
            types: list[int] = []
            strengths: list[float] = []
            for _ in range(settings.batch_size):
                base = clean[int(rng.integers(0, len(clean)))]
                if rng.random() < 0.2:  # keep some clean examples in the mix
                    samples.append(stats.normalise(base))
                    types.append(PRETEXT_TYPES.index("none"))
                    strengths.append(0.0)
                else:
                    damaged = degrader.apply(base)
                    samples.append(stats.normalise(damaged.sample))
                    types.append(damaged.pretext_type)
                    strengths.append(damaged.strength)

            batch = collate(samples, types, strengths).to(device)
            outputs = _forward(model, batch)
            loss = loss_fns.pretext_loss(
                outputs.pretext_type, batch.pretext_type,
                outputs.pretext_strength, batch.pretext_strength,
            )
            loss = loss + loss_fns.problem_bce_loss(outputs.problems, batch.problems)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total += float(loss.detach())
            n_batches += 1
            correct += int((outputs.pretext_type.argmax(dim=1) == batch.pretext_type).sum())
            seen += len(batch)

        accuracy = correct / max(1, seen)
        history.append(
            {
                "stage": "A",
                "epoch": epoch,
                "loss": round(total / max(1, n_batches), 4),
                "pretext_accuracy": round(accuracy, 4),
            }
        )
        if epoch % 5 == 0 or epoch == settings.epochs - 1:
            log.info(
                "stage A epoch %d/%d: loss %.4f, damage recognised %.1f%%",
                epoch + 1, settings.epochs, total / max(1, n_batches), 100 * accuracy,
            )


# --------------------------------------------------------------------------- #
# Stage B
# --------------------------------------------------------------------------- #
def train_stage_b(
    model: CaptureQualityNet,
    train_samples: list[CaptureSample],
    val_samples: list[CaptureSample],
    stats: FeatureStats,
    cfg: Config,
    device: str,
    seed: int,
    history: list[dict],
) -> tuple[dict, int, float]:
    """Train the score by ranking. Returns the best state, epoch and val accuracy."""
    settings = cfg.train.stage_b
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.optimizer.lr, weight_decay=cfg.train.optimizer.weight_decay
    )
    rng = np.random.default_rng(seed)
    degrader = FeatureDegrader(seed=seed + 1000)

    all_problems = torch.from_numpy(np.stack([s.problems for s in train_samples]))
    class_weights = loss_fns.positive_weights(all_problems).to(device)

    normalised = [stats.normalise(s) for s in train_samples]
    val_normalised = [stats.normalise(s) for s in val_samples]

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_accuracy, best_epoch, patience = -1.0, 0, 0

    for epoch in range(settings.epochs):
        model.train()
        batches = scene_batches(normalised, settings.batch_size, rng)
        if not batches:
            log.warning("no batch contains a usable ranking pair; stage B cannot train")
            break

        total, n_batches, accuracies = 0.0, 0, []
        for group in batches:
            batch = collate(group).to(device)
            outputs = _forward(model, batch)

            better, worse = loss_fns.build_pairs(
                batch.scene_ids, batch.psnr, batch.damage, batch.sources
            )
            loss = loss_fns.pairwise_ranking_loss(
                outputs.score, better, worse, margin=settings.margin
            )
            loss = loss + settings.problem_weight * loss_fns.problem_bce_loss(
                outputs.problems, batch.problems, class_weights
            )
            if settings.loss == "ranking+mse":
                loss = loss + loss_fns.score_mse_loss(outputs.score, batch.psnr)

            # Auxiliary pretext, so stage A is not forgotten while stage B runs.
            if settings.aux_pretext_weight > 0:
                aux_samples, types, strengths = [], [], []
                for sample in group[: min(len(group), settings.batch_size)]:
                    damaged = degrader.apply(sample)
                    aux_samples.append(damaged.sample)
                    types.append(damaged.pretext_type)
                    strengths.append(damaged.strength)
                aux_batch = collate(aux_samples, types, strengths).to(device)
                aux_out = _forward(model, aux_batch)
                loss = loss + settings.aux_pretext_weight * loss_fns.pretext_loss(
                    aux_out.pretext_type, aux_batch.pretext_type,
                    aux_out.pretext_strength, aux_batch.pretext_strength,
                )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total += float(loss.detach())
            n_batches += 1
            accuracy = ranking_accuracy(outputs.score.detach(), better, worse)
            if np.isfinite(accuracy):
                accuracies.append(accuracy)

        val_accuracy = evaluate_ranking(model, val_normalised, cfg, device)
        history.append(
            {
                "stage": "B",
                "epoch": epoch,
                "loss": round(total / max(1, n_batches), 4),
                "train_ranking_accuracy": (
                    round(float(np.mean(accuracies)), 4) if accuracies else None
                ),
                "val_ranking_accuracy": (
                    round(val_accuracy, 4) if np.isfinite(val_accuracy) else None
                ),
            }
        )

        improved = np.isfinite(val_accuracy) and val_accuracy > best_accuracy
        if improved or not np.isfinite(val_accuracy) and best_accuracy < 0:
            best_accuracy = val_accuracy if np.isfinite(val_accuracy) else 0.0
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= settings.patience:
                log.info("early stop at epoch %d (best %d)", epoch + 1, best_epoch + 1)
                break

        if epoch % 5 == 0:
            log.info(
                "stage B epoch %d/%d: loss %.4f, train rank acc %.2f, val rank acc %.2f",
                epoch + 1, settings.epochs, total / max(1, n_batches),
                float(np.mean(accuracies)) if accuracies else float("nan"), val_accuracy,
            )

    return best_state, best_epoch, best_accuracy


@torch.no_grad()
def evaluate_ranking(
    model: CaptureQualityNet,
    samples: list[CaptureSample],
    cfg: Config,
    device: str,
) -> float:
    """Ranking accuracy over every known-order pair in a set of samples."""
    if len(samples) < 2:
        return float("nan")
    model.eval()
    batch = collate(samples).to(device)
    outputs = _forward(model, batch)
    better, worse = loss_fns.build_pairs(
        batch.scene_ids, batch.psnr, batch.damage, batch.sources
    )
    return ranking_accuracy(outputs.score, better, worse)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def train_run(
    config_path: str | Path | None = None,
    run_id: str | None = None,
    seeds: list[int] | None = None,
    run_stage_a: bool = True,
    labels_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> TrainResult:
    """Train the ensemble and write everything a run needs to ``models/<run_id>/``."""
    cfg = load_config(Path(config_path).parent) if config_path else get_config()
    device = pick_device(cfg)
    models_root = Path(models_dir) if models_dir else cfg.paths.resolve("models")
    models_root.mkdir(parents=True, exist_ok=True)
    run_dir = models_root / (run_id or next_run_id(models_root))
    run_dir.mkdir(parents=True, exist_ok=True)

    train_scenes, test_scenes = read_splits()
    if not train_scenes:
        raise RuntimeError("no train scenes; run `capturecheck splits` first")

    samples = samples_from_labels(labels_dir=labels_dir, cfg=cfg)
    if not samples:
        raise RuntimeError("no samples; run `capturecheck label data` first")

    train_set = [s for s in samples if s.scene_id in set(train_scenes)]
    test_set = [s for s in samples if s.scene_id in set(test_scenes)]
    if not train_set:
        raise RuntimeError("the train scenes have no labelled captures")
    log.info(
        "%d captures: %d train (%d scenes), %d test (%d scenes)",
        len(samples), len(train_set), len(train_scenes), len(test_set), len(test_scenes),
    )

    # Statistics come from the train side only: the test scenes must stay unseen.
    stats = FeatureStats.fit(train_set)
    clean = clean_samples(train_set)
    log.info("%d clean captures available for stage A", len(clean))

    seed_list = seeds or cfg.train.seed
    scenes_in_order = sorted({s.scene_id for s in train_set})
    results: list[SeedResult] = []
    history_rows: list[dict] = []

    for position, seed in enumerate(seed_list):
        set_seed(int(seed))
        # Leave one scene out, a different one per seed.
        val_scene = scenes_in_order[position % len(scenes_in_order)]
        fold_train = [s for s in train_set if s.scene_id != val_scene]
        fold_val = [s for s in train_set if s.scene_id == val_scene]
        if not fold_train:  # only one scene available
            fold_train, fold_val = train_set, train_set

        model = CaptureQualityNet(
            d_model=cfg.train.model.d_model,
            layers=cfg.train.model.layers,
            heads=cfg.train.model.heads,
            dropout=cfg.train.model.dropout,
        ).to(device)
        if position == 0:
            log.info("model has %d parameters", count_parameters(model))

        history: list[dict] = []
        if run_stage_a:
            fold_clean = [s for s in clean if s.scene_id != val_scene] or clean
            train_stage_a(model, fold_clean, stats, cfg, device, int(seed), history)

        state, best_epoch, val_accuracy = train_stage_b(
            model, fold_train, fold_val, stats, cfg, device, int(seed), history
        )
        model.load_state_dict(state)

        for row in history:
            row["seed"] = int(seed)
            row["val_scene"] = val_scene
        history_rows.extend(history)

        results.append(
            SeedResult(
                seed=int(seed),
                val_scene=val_scene,
                val_ranking_accuracy=float(val_accuracy),
                best_epoch=int(best_epoch),
                state=state,
                history=history,
            )
        )
        log.info(
            "seed %d done: held out `%s`, val ranking accuracy %.2f",
            seed, val_scene, val_accuracy,
        )

    # --- save the run ------------------------------------------------------ #
    torch.save(
        {
            "states": [r.state for r in results],
            "seeds": [r.seed for r in results],
            "model": {
                "d_model": cfg.train.model.d_model,
                "layers": cfg.train.model.layers,
                "heads": cfg.train.model.heads,
                "dropout": cfg.train.model.dropout,
            },
        },
        run_dir / "model.pt",
    )
    stats.save(run_dir / "feature_stats.json")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.raw["train"], sort_keys=False), encoding="utf-8"
    )
    (run_dir / "thresholds.yaml").write_text(
        yaml.safe_dump(cfg.raw["labeling"], sort_keys=False), encoding="utf-8"
    )
    (run_dir / "splits.json").write_text(
        json.dumps(
            {
                "train_scenes": sorted(train_scenes),
                "test_scenes": sorted(test_scenes),
                "val_scene_per_seed": {str(r.seed): r.val_scene for r in results},
                "n_train_captures": len(train_set),
                "n_test_captures": len(test_set),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if history_rows:
        import pandas as pd

        pd.DataFrame(history_rows).to_csv(run_dir / "train_log.csv", index=False)

    shiny = cfg.paths.resolve("models") / "shiny_classifier.joblib"
    if shiny.is_file():
        shutil.copy2(shiny, run_dir / "shiny_classifier.joblib")

    validation = [r.val_ranking_accuracy for r in results if np.isfinite(r.val_ranking_accuracy)]
    metrics = {
        "run_id": run_dir.name,
        "n_seeds": len(results),
        "n_train_captures": len(train_set),
        "n_test_captures": len(test_set),
        "n_problems": schema.N_PROBLEMS,
        "val_ranking_accuracy_mean": float(np.mean(validation)) if validation else None,
        "val_ranking_accuracy_per_seed": {
            str(r.seed): (
                None if not np.isfinite(r.val_ranking_accuracy) else r.val_ranking_accuracy
            )
            for r in results
        },
        "val_scene_per_seed": {str(r.seed): r.val_scene for r in results},
        "device": device,
        "stage_a": run_stage_a,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("run written to %s", run_dir)

    return TrainResult(run_dir=run_dir, metrics=metrics, seeds=results)


def latest_run(models_dir: str | Path | None = None) -> Path | None:
    """The most recent ``models/<run_id>/`` that holds a model."""
    cfg = get_config()
    root = Path(models_dir) if models_dir else cfg.paths.resolve("models")
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "model.pt").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "model.pt").stat().st_mtime)
