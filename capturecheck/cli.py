"""The ``capturecheck`` command line tool.

Commands: ``check`` (one capture), ``label`` (a whole folder), ``degrade``,
``features``, ``splits``, ``train``, ``evaluate``, ``export``, ``review`` and
``labels`` (print the label definitions).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from capturecheck import __version__
from capturecheck.config import get_config
from capturecheck.labeling import schema
from capturecheck.utils.logging import get_logger, setup_logging

app = typer.Typer(
    add_completion=False,
    help="Predict 3D Gaussian Splatting quality before you train, and diagnose the capture.",
    no_args_is_help=True,
)
console = Console()
log = get_logger(__name__)

QUALITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.75, "likely good"),
    (0.55, "fair"),
    (0.35, "likely poor"),
    (0.0, "likely unusable"),
)


def quality_band(score: float) -> str:
    """Turn a 0-1 score into words. The cut-offs are calibrated in evaluation."""
    for cutoff, name in QUALITY_BANDS:
        if score >= cutoff:
            return name
    return "unknown"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"capturecheck {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Print the version."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only."),
) -> None:
    setup_logging("DEBUG" if verbose else ("WARNING" if quiet else "INFO"))


# --------------------------------------------------------------------------- #
# check: one capture
# --------------------------------------------------------------------------- #
@app.command()
def check(
    path: Path = typer.Argument(..., help="A video file, a folder of photos, or a .zip."),
    run: Path | None = typer.Option(
        None, "--run", help="Model run to score with (default: the newest in models/)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    no_dino: bool = typer.Option(False, "--no-dino", help="Skip DINOv2 (faster, less accurate)."),
    no_score: bool = typer.Option(False, "--no-score", help="Labels and evidence only."),
) -> None:
    """Check one capture and print a readable report."""
    from capturecheck.labeling.pipeline import label_capture

    cfg = get_config()
    result = label_capture(path, use_dino=not no_dino, keep_measurements=True)
    row = result.video_row
    evidence = result.evidence

    prediction = None
    if not no_score:
        from capturecheck.model.predict import predict_capture

        prediction = predict_capture(result, run_dir=run)

    if as_json:
        payload = {
            "capture_id": result.capture_id,
            "input_type": row["input_type"],
            "n_frames": row["n_frames"],
            "score": prediction.score if prediction else None,
            "band": quality_band(prediction.score) if prediction else None,
            "problems": {
                name: {
                    "flagged": int(row.get(name, 0) or 0),
                    "probability": (
                        prediction.problems.get(name) if prediction else None
                    ),
                    "evidence": evidence.get(name),
                }
                for name in schema.VIDEO_LABEL_NAMES
            },
            "stats": {
                k: row.get(k)
                for k in schema.VIDEO_STAT_COLUMNS
            },
            "review": int(row.get("review", 0) or 0),
            "review_reason": row.get("review_reason") or "",
            "time_s": row.get("time_s_total"),
            "model": prediction.run_id if prediction else None,
        }
        console.print_json(json.dumps(payload, default=float))
        return

    # --- human-readable report -------------------------------------------- #
    console.print()
    console.print(
        f"[bold]{result.capture_id}[/bold]  "
        f"({row['input_type']}, {row['n_frames']} frames)"
    )
    if prediction is not None:
        console.print(
            f"Score: [bold]{prediction.score:.2f}[/bold]  ({quality_band(prediction.score)})"
            + (f"   [dim]model: {prediction.run_id}[/dim]" if prediction.run_id else "")
        )
    else:
        console.print("[dim]Score: not computed (--no-score)[/dim]")

    flagged = result.problems
    if flagged:
        console.print("Problems:")
        for name in flagged:
            title = schema.LABEL_TITLES.get(name, name)
            detail = evidence.get(name, "")
            probability = prediction.problems.get(name) if prediction else None
            suffix = f"  [dim](p={probability:.2f})[/dim]" if probability is not None else ""
            console.print(f"  • [yellow]{title:<14}[/yellow] {detail}{suffix}")
    else:
        console.print("Problems: [green]none found[/green]")

    if prediction is not None:
        extra = [
            (name, p)
            for name, p in sorted(prediction.problems.items(), key=lambda kv: -kv[1])
            if p >= 0.5 and name not in flagged
        ]
        if extra:
            console.print("The model also suspects:")
            for name, p in extra:
                console.print(f"  • [dim]{schema.LABEL_TITLES.get(name, name)} (p={p:.2f})[/dim]")

    if int(row.get("review", 0) or 0):
        console.print(f"[magenta]Needs review:[/magenta] {row.get('review_reason')}")

    budget = cfg.runtime.time_budget_s
    total = float(row.get("time_s_total") or 0.0)
    colour = "green" if total <= budget else "red"
    console.print(f"Time: [{colour}]{total:.0f} s[/{colour}] (budget {budget:.0f} s)")
    console.print()


# --------------------------------------------------------------------------- #
# label: a whole folder
# --------------------------------------------------------------------------- #
@app.command()
def label(
    folder: Path = typer.Argument(..., help="data/ , data/raw/ , or any folder of captures."),
    out: Path | None = typer.Option(None, "--out", help="Where to write the CSVs."),
    no_dino: bool = typer.Option(False, "--no-dino", help="Skip DINOv2."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached DINOv2 features."),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N captures."),
) -> None:
    """Label every capture in a folder and write the label CSVs."""
    import pandas as pd
    from tqdm import tqdm

    from capturecheck.io import dataset
    from capturecheck.labeling.aggregate import summarize_labels
    from capturecheck.labeling.pipeline import label_capture

    cfg = get_config()
    folder = Path(folder)

    # ``data/`` means "raw plus degraded"; anything else is taken literally.
    if folder.name == "data" or (folder / "raw").is_dir():
        refs = dataset.discover_all(folder / "raw", folder / "degraded")
    else:
        refs = dataset.list_captures(folder)
    if limit:
        refs = refs[: int(limit)]
    if not refs:
        console.print(f"[red]No captures found in {folder}[/red]")
        raise typer.Exit(code=1)

    out_dir = Path(out) if out else cfg.paths.resolve("labels")
    frame_tables: list[pd.DataFrame] = []
    video_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for ref in tqdm(refs, desc="labelling", unit="capture"):
        try:
            result = label_capture(
                ref.path,
                scene_id=ref.scene_id,
                source=ref.source,
                capture_id=ref.capture_id,
                use_dino=not no_dino,
                use_cache=not no_cache,
                degrade_labels=ref.degrade_labels,
                degrade_type=ref.degrade_type,
                degrade_strength=ref.degrade_strength,
            )
        except Exception as exc:  # one bad capture must not stop the batch
            log.exception("failed on %s", ref.capture_id)
            failures.append((ref.capture_id, str(exc)))
            continue
        frame_tables.append(result.frame_rows)
        video_rows.append(result.video_row)

    if not video_rows:
        console.print("[red]Every capture failed.[/red]")
        raise typer.Exit(code=1)

    frame_df = pd.concat(frame_tables, ignore_index=True)
    video_df = pd.DataFrame(video_rows)
    video_df = dataset.merge_quality_targets(video_df, out_dir)

    dataset.write_frame_labels(frame_df, out_dir=out_dir)
    dataset.write_video_labels(video_df, out_dir=out_dir)

    queue = video_df[video_df["review"] == 1]
    dataset.write_review_queue(
        queue.reindex(columns=list(schema.REVIEW_QUEUE_COLUMNS)), out_dir=out_dir
    )

    table = Table(title=f"{len(video_df)} captures labelled", show_edge=False)
    table.add_column("capture label")
    table.add_column("captures", justify="right")
    table.add_column("share", justify="right")
    for _, row in summarize_labels(video_df).iterrows():
        table.add_row(row["label"], str(int(row["captures"])), f"{row['share'] * 100:.0f}%")
    console.print(table)

    console.print(
        f"frames: {len(frame_df)} · videos: {int((video_df['input_type'] == 'video').sum())} · "
        f"bundles: {int((video_df['input_type'] == 'images').sum())} · "
        f"needing review: {len(queue)}"
    )
    console.print(f"written to {out_dir}")
    if failures:
        console.print(f"[red]{len(failures)} capture(s) failed:[/red]")
        for capture_id, message in failures[:10]:
            console.print(f"  {capture_id}: {message}")


# --------------------------------------------------------------------------- #
# supporting commands
# --------------------------------------------------------------------------- #
@app.command()
def degrade(
    raw: Path | None = typer.Option(None, "--raw", help="Folder of real captures."),
    out: Path | None = typer.Option(None, "--out", help="Where to write the copies."),
    strength: list[str] = typer.Option(None, "--strength", help="light / medium / heavy."),
    op: list[str] = typer.Option(None, "--op", help="Only these ops."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Redo copies that exist."),
) -> None:
    """Make broken copies of every real capture (exact labels, no human needed)."""
    from capturecheck.degrade.make_variants import make_all

    variants = make_all(
        raw_dir=raw, out_dir=out, ops=list(op or []) or None,
        strengths=list(strength or []) or None, overwrite=overwrite,
    )
    table = Table(title=f"{len(variants)} broken copies", show_edge=False)
    for column in ("copy", "op", "strength", "frames"):
        table.add_column(column)
    for variant in variants[:40]:
        table.add_row(variant.capture_id, variant.op, variant.strength, str(variant.n_frames))
    console.print(table)
    if len(variants) > 40:
        console.print(f"[dim]... and {len(variants) - 40} more[/dim]")


@app.command()
def features(
    folder: Path | None = typer.Option(None, "--folder", help="Folder of captures."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Recompute cached features."),
) -> None:
    """Pre-compute and cache the DINOv2 features for every capture."""
    from tqdm import tqdm

    from capturecheck.features.dino import default_embedder
    from capturecheck.io import dataset
    from capturecheck.io.load import load_capture

    cfg = get_config()
    root = Path(folder) if folder else cfg.paths.resolve("raw").parent
    refs = (
        dataset.discover_all()
        if folder is None
        else dataset.list_captures(root)
    )
    if not refs:
        console.print("[red]No captures found.[/red]")
        raise typer.Exit(code=1)

    embedder = default_embedder()
    for ref in tqdm(refs, desc="features", unit="capture"):
        capture = load_capture(ref.path, capture_id=ref.capture_id)
        embedder.embed_capture(ref.capture_id, capture.frames, use_cache=not overwrite)
    console.print(f"cached features for {len(refs)} captures in {cfg.paths.resolve('features')}")


@app.command()
def splits(
    test_share: float = typer.Option(0.2, "--test-share", help="Share of scenes held out."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Split scenes into train and test (company captures always go to test)."""
    from capturecheck.model.splits import make_splits

    report = make_splits(test_share=test_share, seed=seed)
    console.print(report.summary())


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config", help="Override configs/train.yaml."),
    run_id: str | None = typer.Option(None, "--run-id", help="Name for models/<run_id>."),
    seeds: list[int] = typer.Option(None, "--seed", help="Override the seed ensemble."),
    no_stage_a: bool = typer.Option(False, "--no-stage-a", help="Skip the pretext stage."),
) -> None:
    """Train the quality model (stage A pretext, then stage B ranking)."""
    from capturecheck.model.train import train_run

    result = train_run(
        config_path=config, run_id=run_id,
        seeds=list(seeds or []) or None, run_stage_a=not no_stage_a,
    )
    console.print(f"run written to [bold]{result.run_dir}[/bold]")
    console.print(json.dumps(result.metrics, indent=2, default=float))


@app.command()
def evaluate(
    run: Path | None = typer.Option(None, "--run", help="Run to evaluate (default: newest)."),
    report: Path | None = typer.Option(None, "--report", help="Where to write the markdown."),
) -> None:
    """Evaluate a run against the baselines and write the evaluation report."""
    from capturecheck.model.evaluate import evaluate_run

    result = evaluate_run(run_dir=run, report_path=report)
    console.print(result.summary())


@app.command()
def export(
    run: Path | None = typer.Option(None, "--run", help="Run to export (default: newest)."),
) -> None:
    """Export a trained run to ONNX for fast CPU inference."""
    from capturecheck.model.export import export_run

    path = export_run(run_dir=run)
    console.print(f"exported {path}")


@app.command()
def review() -> None:
    """Print how to start the human review screen."""
    root = get_config().paths.resolve("labels")
    console.print("Start the review app with:")
    console.print("  [bold]streamlit run capturecheck/review/app.py[/bold]")
    console.print(f"It reads {root / 'review_queue.csv'} and writes human_labels.csv there.")


@app.command()
def labels() -> None:
    """Print every frame and capture label with its meaning."""
    console.print(schema.describe())


if __name__ == "__main__":  # pragma: no cover
    app()
