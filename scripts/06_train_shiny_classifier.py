#!/usr/bin/env python
"""Step 6: train the shiny/glass classifier on top of frozen DINOv2 features.

Needs ``data/classifier_train/labels.csv`` with columns ``image_path,shiny``.
Use ``--collect`` first to pull a spread of frames out of the captures, then
``--label`` to tag them one key at a time, then run this script again to train.

Shiny and glass surfaces break 3DGS badly (the reconstruction puts floaters where
the reflection moved), and about 200 labelled frames is enough, because the
DINOv2 features already separate them: only the boundary has to be learned.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from capturecheck.config import get_config
from capturecheck.features.dino import evenly_spaced
from capturecheck.io import dataset
from capturecheck.io.load import load_capture
from capturecheck.labeling.classifiers import train_shiny
from capturecheck.utils.logging import setup_logging

LABELS_FILE = "labels.csv"


def collect(out_dir: Path, per_capture: int, limit: int | None) -> Path:
    """Write a spread of frames to ``data/classifier_train/images`` for labelling."""
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    refs = [r for r in dataset.discover_all() if r.source != "degraded"]
    if limit:
        refs = refs[:limit]
    written = 0
    for ref in refs:
        capture = load_capture(ref.path, capture_id=ref.capture_id)
        for idx in evenly_spaced(capture.n_frames, per_capture):
            target = images_dir / f"{ref.capture_id}__{int(idx):04d}.jpg"
            cv2.imwrite(str(target), capture.frames[int(idx)])
            written += 1
    print(f"wrote {written} frames to {images_dir}")
    print(f"now label them:  python {Path(__file__).name} --label")
    return images_dir


def label_interactively(out_dir: Path) -> Path:
    """Show each frame; press 1 for shiny, 0 for not, s to skip, q to stop."""
    images_dir = out_dir / "images"
    csv_path = out_dir / LABELS_FILE
    photos = sorted(images_dir.glob("*.jpg"))
    if not photos:
        raise SystemExit(f"no frames in {images_dir}; run with --collect first")

    done: dict[str, int] = {}
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done[row["image_path"]] = int(row["shiny"])

    print("1 = shiny/glass · 0 = not · s = skip · q = save and quit")
    for photo in photos:
        key_name = str(photo.relative_to(out_dir))
        if key_name in done:
            continue
        image = cv2.imread(str(photo))
        cv2.imshow("shiny? 1=yes 0=no s=skip q=quit", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            continue
        if key in (ord("0"), ord("1")):
            done[key_name] = int(chr(key))
    cv2.destroyAllWindows()

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "shiny"])
        for name, value in sorted(done.items()):
            writer.writerow([name, value])
    positives = sum(done.values())
    print(f"wrote {csv_path}: {len(done)} labelled, {positives} shiny")
    return csv_path


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=cfg.paths.resolve("classifier_train"))
    parser.add_argument("--models", type=Path, default=cfg.paths.resolve("models"))
    parser.add_argument("--collect", action="store_true", help="Pull frames out of the captures.")
    parser.add_argument("--label", action="store_true", help="Label the collected frames.")
    parser.add_argument("--per-capture", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-precision", type=float, default=0.8)
    args = parser.parse_args()
    setup_logging()

    if args.collect:
        collect(args.data, args.per_capture, args.limit)
        return
    if args.label:
        label_interactively(args.data)
        return

    csv_path = args.data / LABELS_FILE
    if not csv_path.is_file():
        raise SystemExit(
            f"{csv_path} not found. Collect and label some frames first:\n"
            f"  python {Path(__file__).name} --collect\n"
            f"  python {Path(__file__).name} --label"
        )

    report = train_shiny(csv_path, models_dir=args.models)
    print(report.summary())
    if report.precision < args.min_precision:
        print(
            f"\nPrecision {report.precision:.2f} is below {args.min_precision:.2f}. "
            "Label more examples (especially ones the classifier gets wrong) and train again."
        )


if __name__ == "__main__":
    main()
