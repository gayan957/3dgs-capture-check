# CaptureCheck

**Predict 3D Gaussian Splatting quality before you train, and find out what's wrong with the capture.**

CaptureCheck looks at a capture, either a **video** or an **image bundle** (a folder or .zip of photos), and, in under 60 seconds on a laptop CPU, gives you:

- a **quality score** that predicts how good the splat will be, and
- a **diagnosis**: the problems found, with evidence.

```
$ capturecheck check ./office_scan.mp4
Score: 0.42  (likely poor)
Problems:
  • Motion blur     14.0–17.5 s (frames 28–35), 38% of frames blurry
  • Poor coverage   9 of 24 view bins empty: left, eye level; back, eye level
Time: 41 s

$ capturecheck check ./kitchen_photos.zip
Score: 0.58  (fair)
Problems:
  • Low overlap     gap between IMG_0412 and IMG_0413 (8% shared)
  • Motion blur     IMG_0520 to IMG_0524, shutter 1/30 s on 12% of photos
Time: 33 s
```

> Built for the Rysera Innovations TransitionX 2026 problem statement
> "Predicting reconstruction quality before you train". Licensed Apache 2.0.

---

## Contents
1. [Why](#why)
2. [How it works](#how-it-works)
3. [Supported inputs](#supported-inputs)
4. [Install](#install)
5. [Quick start](#quick-start)
6. [Labelling workflow](#labelling-workflow)
7. [Training](#training)
8. [Evaluation and results](#evaluation-and-results)
9. [Dataset and collection protocol](#dataset-and-collection-protocol)
10. [Project structure](#project-structure)
11. [Configuration](#configuration)
12. [Limitations](#limitations)
13. [Contributing](#contributing)
14. [Licence and acknowledgements](#licence-and-acknowledgements)

---

## Why
A 3DGS training run takes 20–60 minutes of GPU time, and about a third of runs are thrown away. When a run fails, the output doesn't say why. CaptureCheck checks the capture first, before anyone leaves the building.

## How it works
```
video        → 2 frames/s  ┐
image bundle → every photo ┘→ (max 600) → OpenCV checks (blur, light, noise, texture,
                                           overlap, speed, walked or turned, coverage)
                                         → DINOv2 on ≤ 80 frames (duplicates, diversity, shiny)
                                         → rules → frame labels → capture labels + evidence
                                         → small trained model → quality score
```
- **No 3DGS and no GPU** are needed to run it.
- DINOv2 is used **pretrained and frozen**; only a small model on top is trained.
- Blur is measured with OpenCV, not DINOv2. For photos, EXIF shutter speed and ISO add extra evidence.

### Walked or turned
The check worth explaining is the one that catches the most expensive mistake: filming by **turning on the spot**. A pivot gives 3DGS no parallax, so it cannot recover depth however sharp the frames are, and nothing in the images themselves looks wrong.

For each pair of frames CaptureCheck fits both a homography and an essential matrix to the same ORB matches, and compares how much of the total explanatory score the homography accounts for. A homography describes pure rotation exactly, whatever the scene; only real translation lets the essential matrix explain points a homography cannot.

The gap between the compared frames is chosen per capture. Neighbouring frames of a dense capture carry too little baseline to tell a walk from a pivot, so the gap widens until the motion is decisive or the frames no longer share enough features to match. That search only ever accepts a confident *walked* early: "turned" is what a homography says at any short baseline, while a camera that really pivoted reads as turned at every gap. When no gap can decide, the pair is reported as `unknown` rather than guessed at.

## Supported inputs
| Input | Example | How frames are picked |
|---|---|---|
| Video | `room.mp4`, `room.mov` | 2 frames per second (max 600) |
| Photo folder | `room_photos/*.jpg` / `*.png` | Every photo, ordered by EXIF time, else file name (max 600) |
| Zip of photos | `room_photos.zip` | Same as a photo folder |

Photo order comes from EXIF capture time where present, and otherwise from file names when they form a numbered sequence (`IMG_0412.jpg`, `DSC_0007.jpg`). A pile of unrelated names has no order to use, so each photo is compared with the photo it looks most like instead of "the next one".

## Install
```bash
git clone <repo-url> && cd capturecheck
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional: ffmpeg/ffprobe on PATH gives better video metadata (OpenCV is the fallback)
```

The first run downloads the DINOv2 weights (~84 MB) from torch.hub; after that it works offline.

## Quick start
```bash
capturecheck check path/to/video.mp4          # a video
capturecheck check path/to/photos/            # a folder of photos
capturecheck check path/to/photos.zip --json  # a zip, machine-readable output
capturecheck labels                           # what every label means
```

No captures yet? Generate a synthetic set and run the whole pipeline on it:
```bash
python scripts/00_make_demo_data.py --clean   # 7 synthetic captures in data/raw
capturecheck degrade                          # broken copies with exact labels
capturecheck label data                       # label everything
capturecheck splits && capturecheck train     # split by scene, then train
capturecheck evaluate                         # vs baselines, writes the report
```

## Labelling workflow
| Step | Command | Output |
|---|---|---|
| 1. Put captures in `data/raw/<capture_id>/` (a video, an `images/` folder, or a .zip) + optional `scene.txt` | n/a | n/a |
| 2. Make broken copies (exact, free labels) | `python scripts/02_make_degraded_copies.py` | `data/degraded/` |
| 3. Auto-label everything | `capturecheck label data` | `data/labels/*.csv` |
| 4. Review uncertain captures | `streamlit run capturecheck/review/app.py` | `human_labels.csv` |
| 5. Train shiny-surface classifier (~200 labelled frames) | `python scripts/06_train_shiny_classifier.py` | `models/shiny_classifier.joblib` |
| 6. Check label agreement | `python scripts/07_check_label_agreement.py` | `reports/label_agreement.md` |

### Labels
| Capture label | Meaning |
|---|---|
| `motion_blur` | Too many blurry frames or photos |
| `lighting` | Brightness changes or bad light |
| `no_walking` | Camera only turned on the spot |
| `poor_coverage` | Sides missing / too many repeated views |
| `too_few_views` | Not enough different frames or photos |
| `low_overlap` | Neighbouring photos share too little (gaps between shots) |
| `mixed_cameras` | Photos from different cameras or sizes |
| `reflective` | Many shiny or glass surfaces |
| `low_texture` | Many plain surfaces |
| `noisy` | Heavy sensor noise (high ISO or a very dark scene) |
| `REVIEW` | System unsure, a person checks |

A capture goes to `REVIEW` when a measurement sits within 15% of its threshold, or when two checks disagree — for example varied-looking views next to a camera path that recovered no motion.

## Training
```bash
python scripts/08_make_splits.py      # split by scene; company captures → test only
capturecheck train                    # stage A: spot damage; stage B: learn ranking/score
```

Two stages, because there will never be many captures with a real PSNR number:

- **Stage A (pretext).** Damaged copies of the clean captures are generated at the *feature* level — lower the sharpness numbers, drop a coverage sector, remove tokens — by the thousand, and the model is asked what was done and how strongly. No PSNR and no human labels needed.
- **Stage B (score).** The score head is trained by **ranking**, not regression, because PSNR values from different scenes are not comparable: a textured room scores higher than a white wall however well either was filmed. Pairs are only formed where the true order is known — the same scene clean versus broken, a lighter versus a heavier copy, or two captures with real PSNR values.

Five seeds are trained and kept as an ensemble; each validates on a different held-out scene, which is leave-one-scene-out across the ensemble. Saved runs go to `models/<run_id>/` (weights, ONNX, thresholds, feature statistics, metrics, splits, training log).

## Evaluation and results
```bash
capturecheck evaluate
```

Writes `reports/evaluation_report.md` with Spearman correlation (and a bootstrap 95% interval) for the model and every baseline — frame count, mean blur, COLMAP match count, LightGBM on the same features — plus precision and recall per problem, a predicted-vs-target scatter plot, and runtime.

The report calls it out when a baseline ranks the test captures as well as the model does. Read the confidence interval, not the point estimate: a test set of a few dozen captures cannot separate methods whose intervals overlap.

**The numbers in `reports/evaluation_report.md` are only as meaningful as the `psnr` column they are measured against.** On the synthetic demo set that column is a geometric stand-in, not a 3DGS measurement, so a result there says the model recovered the stand-in — nothing about real reconstruction quality. Fill `data/labels/quality_targets.csv` with PSNR from real 3DGS runs before quoting a result.

## Dataset and collection protocol
- **Sources:** company captures (test only), your own videos and photo bundles, and broken copies of good captures.
- **Recording rules:** walk around the subject rather than turning on the spot; keep neighbouring frames overlapping; lock the exposure if the camera allows it; cover every side at the height you care about.
- **Labels:** automatic (file info, EXIF, OpenCV, DINOv2), exact (broken copies), human (review screen, two reviewers on a shared subset).
- **Quality targets:** PSNR on held-out *clean* frames, where available.
- **Splits:** by `scene_id`, never by capture; see `data/splits/`.
- Full dataset card: [`data/README.md`](data/README.md).

## Project structure
```
capturecheck/
├── configs/        # all thresholds and settings
├── capturecheck/   # package: io (video + images), checks, features, labeling, degrade, review, model
├── scripts/        # numbered pipeline steps 00–11
├── data/           # videos, photo bundles, frames, features, labels, splits (not in git)
├── models/         # saved runs (not in git)
├── reports/        # label agreement + evaluation reports
├── notebooks/      # threshold exploration
└── tests/
```

## Configuration
Every threshold lives in `configs/`, and no module hard-codes one:

- `configs/labeling.yaml`: sampling (2 fps, max 600, photo order) and label thresholds
- `configs/degrade.yaml`: broken-copy strengths
- `configs/train.yaml`: model and training settings
- `configs/default.yaml`: paths and the 60-second budget

The defaults are a starting point, not a calibration. Run the review screen on a few dozen of your own captures and `scripts/07_check_label_agreement.py` will tell you which thresholds to move and in which direction.

## Limitations
- **The camera path is approximate.** Relative poses are chained with an unknown, drifting scale; it is enough to answer "did the camera ever get to the other side?" and not much more. When too few pairs give a pose, coverage is reported as unreliable rather than guessed.
- **A pan across a flat scene reads as `turned`.** A homography explains a planar scene exactly, so filming a wall while walking past it looks the same as pivoting. For 3DGS the consequence is similar — little usable parallax — but the label names the wrong cause.
- **Thresholds depend on camera and resolution.** Sharpness in particular is only comparable between frames measured at the same width; captures narrower than `resize_width` are measured at their own width, recorded as `measured_width`.
- **Photo bundles with neither EXIF time nor numbered names** are paired by visual similarity, which is less reliable than true capture order.
- **Shiny/glass detection** depends on a small hand-labelled classifier; with none trained, `reflective` is never flagged rather than guessed.
- **Predictions are validated only on the captures in the evaluation report.** The model has no way to know about a failure mode it has never seen.

## Contributing
`make lint && make test` must pass (tests cover videos and image bundles, and run offline). Keep thresholds in `configs/` and label names in `capturecheck/labeling/schema.py`. See [`CLAUDE.md`](CLAUDE.md) for the rules that must not be broken.

## Licence and acknowledgements
Apache 2.0. Uses DINOv2 (Meta AI, Apache 2.0), OpenCV, Pillow, PyTorch, scikit-learn, LightGBM, ONNX Runtime, Typer and Streamlit. The walked-versus-turned decision follows the model-selection approach used by ORB-SLAM's initialiser.
