# CaptureCheck — technical reference

The detail behind [README.md](README.md): how each check works, how the model is
trained, what the numbers mean, and where the limits are.

If you just want to run it, start with the README.

---

## Contents
1. [Why](#why)
2. [How it works](#how-it-works)
3. [Supported inputs](#supported-inputs)
4. [Install](#install)
5. [Quick start](#quick-start)
6. [The web interface](#the-web-interface)
7. [Labelling workflow](#labelling-workflow)
8. [Training](#training)
9. [Evaluation and results](#evaluation-and-results)
10. [Dataset and collection protocol](#dataset-and-collection-protocol)
11. [Project structure](#project-structure)
12. [Configuration](#configuration)
13. [Limitations](#limitations)
14. [Contributing](#contributing)
15. [Licence and acknowledgements](#licence-and-acknowledgements)

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

| Install | Gives you |
|---|---|
| `pip install -e .` | Checking and labelling captures, including the DINOv2 features |
| `pip install -e ".[model]"` | …plus training, the baselines and the ONNX export |
| `pip install -e ".[review]"` | …plus the human review screen |
| `pip install -e ".[dev]"` | Everything, plus pytest and ruff |

## Quick start
```bash
capturecheck check path/to/video.mp4          # a video
capturecheck check path/to/photos/            # a folder of photos
capturecheck check path/to/photos.zip --json  # a zip, machine-readable output
capturecheck labels                           # what every label means
```

No captures yet? Generate a synthetic set and run the whole pipeline on it:
```bash
make demo                 # 8 synthetic captures in data/raw (4 videos, 4 photo bundles)
make degrade              # broken copies with exact labels
make label                # label everything
make check-degradations   # confirm each broken copy really is broken
make splits train         # split by scene, then train
make export eval          # ONNX export, then the evaluation report
```

Every step is also a `capturecheck` subcommand (`capturecheck label data`, `capturecheck train`, …); `make` is just a shorthand that uses the project's virtualenv.

Checking one of those demo captures afterwards gives the real thing — this is
`lobby_pivot`, a capture that pivots on the spot instead of walking round the room:

```
$ capturecheck -q check data/raw/lobby_pivot

lobby_pivot  (video, 64 frames)
Score: 0.29  (likely unusable)   model: 2026-10-27_run01
Problems:
  • No walking     only 0% of frame pairs show real camera movement; the rest is
                   rotation on the spot  (p=1.00)
  • Poor coverage  camera never moved, so only one side was captured  (p=1.00)
Time: 1 s (budget 60 s)
```

## The web interface
A browser front end for people who should not have to touch a terminal: drag a
capture in, watch it being checked, read the score and the evidence.

```bash
pip install -e ".[api]"   # fastapi + uvicorn
make ui                   # npm install && npm run build
make serve                # http://127.0.0.1:8000
```

`capturecheck serve` runs the API and serves the built front end from the same
port. Without a build it serves the API alone, which is still useful — the
OpenAPI docs are at `/docs`.

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Whether a model is loaded, the time budget, accepted file types, score bands |
| `GET /api/labels` | Every capture problem and its meaning, straight from `schema.py` |
| `POST /api/captures` | Upload a video or a .zip of photos; returns a job to poll |
| `GET /api/captures/{id}` | Stage, elapsed time, and the result once it is done |

Checking a capture is allowed 60 seconds, which is too long to hold a request
open, so an upload returns a job id and the browser polls it. Jobs run one at a
time: the budget is measured on a machine doing nothing else, and two at once
would make both miss it.

Uploads are written to a fresh directory, deleted when the job ends, and
checked with the DINOv2 cache switched off. A visitor's video is not a reason
to write into `data/features/`, where the training inputs live.

For front-end work, run the dev server alongside:
```bash
make serve     # terminal 1: the API on :8000
make ui-dev    # terminal 2: Vite on :5173, proxying /api
```

See [frontend/README.md](frontend/README.md) for how the page is put together.

## Labelling workflow
| Step | Command | Output |
|---|---|---|
| 1. Put captures in `data/raw/<capture_id>/` (a video, an `images/` folder, or a .zip) + optional `scene.txt` | n/a | n/a |
| 2. Make broken copies (exact, free labels) | `python scripts/02_make_degraded_copies.py` | `data/degraded/` |
| 3. Auto-label (only what is new; see below) | `capturecheck label data` | `data/labels/*.csv` |
| 4. Review uncertain captures | `streamlit run capturecheck/review/app.py` | `human_labels.csv` |
| 5. Train shiny-surface classifier (~200 labelled frames) | `python scripts/06_train_shiny_classifier.py` | `models/shiny_classifier.joblib` |
| 6. Check label agreement | `python scripts/07_check_label_agreement.py` | `reports/label_agreement.md` |
| 7. Check the broken copies really are broken | `python scripts/12_check_degradations.py` | printed table |

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

### Adding one capture

Drop it into `data/raw/<capture_id>/` and run `capturecheck label data` again. Only the
new capture is measured; the rows already on disk are carried over untouched, and rows for
captures that have gone are dropped:

```
$ capturecheck label data
measured: 1 new · kept: 51 · removed: 0
```

So adding one capture costs one capture, not the whole folder.

Rows measured under two different rule sets must never share the training table, so
`data/labels/label_run.json` records a digest of every setting the measurements depend on —
the whole labelling config plus whether DINOv2 ran. Move a threshold, change the sampling
rate or pass `--no-dino`, and the digest no longer matches: the folder is remeasured in full
and the reason is printed.

| Flag | Effect |
|---|---|
| *(none)* | Measure what is new, keep what is unchanged, drop what has gone |
| `--from-scratch` | Remeasure everything, replacing all existing rows |
| `--new-only` | Stop with an error rather than remeasure everything |
| `--no-cache` | Ignore the cached DINOv2 features (use after replacing a file under the same capture id) |

Use `--new-only` in a script, where silently spending an hour remeasuring would be worse
than failing.

## Training
```bash
python scripts/08_make_splits.py      # split by scene; company captures → test only
capturecheck train                    # stage A: spot damage; stage B: learn ranking/score
```

Two stages, because there will never be many captures with a real PSNR number:

- **Stage A (pretext).** Damaged copies of the clean captures are generated at the *feature* level — lower the sharpness numbers, drop a coverage sector, remove tokens — by the thousand, and the model is asked what was done and how strongly. No PSNR and no human labels needed.
- **Stage B (score).** The score head is trained by **ranking**, not regression, because PSNR values from different scenes are not comparable: a textured room scores higher than a white wall however well either was filmed. Pairs are only formed where the true order is known — the same scene clean versus broken, a lighter versus a heavier copy, or two captures with real PSNR values.

Five seeds are trained and kept as an ensemble; each validates on a different held-out scene, which is leave-one-scene-out across the ensemble. Saved runs go to `models/<run_id>/` (weights, ONNX, thresholds, feature statistics, metrics, splits, training log).

Stage A needs clean captures to damage, so a dataset in which every capture is marked degraded leaves it with nothing to do. The run's `metrics.json` records whether it **ran**, not just whether it was asked for, and the training log's `stage` column shows both stages — worth checking, because a silently skipped stage A looks exactly like a fast training run.

On a laptop CPU the default settings — five seeds, 30 pretext epochs of 2000 generated variants each — take roughly 20 minutes, most of it stage A. Still less than one 3DGS run, and it is a one-off. For a quick check that the wiring works:

```bash
capturecheck train --seed 0 --no-stage-a    # one seed, score stage only: under a minute
```

Lower `stage_a.epochs` or `seed` in `configs/train.yaml` for anything in between.

## Evaluation and results
```bash
capturecheck evaluate
```

Writes `reports/evaluation_report.md` with Spearman correlation (and a bootstrap 95% interval) for the model and every baseline — frame count, mean blur, COLMAP match count, LightGBM on the same features — plus precision and recall per problem, a predicted-vs-target scatter plot, and runtime.

The report calls it out when a baseline ranks the test captures as well as the model does. Read the confidence interval, not the point estimate: a test set of a few dozen captures cannot separate methods whose intervals overlap.

### Results on the current dataset

> **These numbers do not say how well 3DGS will reconstruct anything.** No capture
> in the set has a real PSNR value yet, so `capturecheck evaluate` ranks against
> *damage severity* (clean > light > medium > heavy) and says so in the report.
> Put real PSNR into `data/labels/quality_targets.csv` before quoting a result.

Run `2026-11-16_run01`: 200 captures (11 real, 189 broken copies) across 11 scenes.
9 scenes train, 2 held out. 35 unseen test captures — 1 video, 34 image bundles.

| Method | Spearman vs target (95% CI) |
|---|---|
| Mean blur | −0.04 (−0.38 – 0.35) |
| Frame count | 0.20 (−0.15 – 0.52) |
| ORB match count (COLMAP stand-in) | 0.27 (−0.07 – 0.57) |
| **CaptureCheck** | **0.46 (0.12 – 0.72)** |
| LightGBM on the same features | 0.66 (0.41 – 0.83) |

**LightGBM on the same features beats the model, 0.66 to 0.46.** The report prints
that as a note rather than burying it: *"On this test set the trained model is not
yet earning its place."* Both still beat frame count and mean blur.

The previous run scored 0.69 on 17 test captures, 2 of which carried a PSNR
stand-in. That set was synthetic and has been deleted. The drop to 0.46 is a
change of test set — real rooms, no PSNR at all — not a regression.

Per-problem precision and recall against the rules' labels:

| Problem label | Precision | Recall | F1 | Positives in test |
|---|---|---|---|---|
| `lighting` | 1.00 | 1.00 | 1.00 | 35 |
| `low_overlap` | 1.00 | 1.00 | 1.00 | 35 |
| `motion_blur` | 0.86 | 0.86 | 0.86 | 14 |
| `noisy` | 0.56 | 0.83 | 0.67 | 6 |
| `poor_coverage` | 0.57 | 0.53 | 0.55 | 15 |
| `too_few_views` | 1.00 | 0.33 | 0.50 | 6 |
| `no_walking` | — | — | — | 0 |
| `mixed_cameras` | — | — | — | 0 |
| `reflective` | — | — | — | 0 |
| `low_texture` | — | — | — | 0 |

`lighting` and `low_overlap` score 1.00 because they fire on **all 35** test
captures. A label that is true of everything is trivially easy to predict and
carries no information; those two perfect scores mean the thresholds need
recalibrating, not that the model is good. `overlap.min_overlap` is the clearest
target.

Four labels have no positive examples, so nothing about them has been tested.
`reflective` cannot fire at all: no shiny classifier has been trained, so
`shiny_prob` is NaN and the capture reports 0% shiny with full confidence rather
than honestly reporting "unknown".

A caveat on all of these: they are measured against the **rules'** labels, not
human judgement. High precision says the model reproduces the rules, not that the
rules are right. [`reports/label_agreement.md`](reports/label_agreement.md)
explains how to close that gap.

Validation ranking accuracy across the five leave-one-scene-out folds averages
0.67, running 0.49 (holding out `courthouse`) to 0.82 (holding out `Robotic lab`).
One fold at chance is worth watching as the set grows.

Runtime on a laptop CPU, averaged over the test set:

| | Per capture |
|---|---|
| Measuring (OpenCV + DINOv2) | 7.61 s |
| Scoring (the model itself) | 0.006 s |
| **Total** | **7.62 s** |

Well inside the 60-second budget, and the per-frame cost is guarded by
`tests/test_budget.py`.

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
├── capturecheck/   # package: io (video + images), checks, features, labeling, degrade, review, model, api
├── scripts/        # numbered pipeline steps 00–12
├── data/           # videos, photo bundles, frames, features, labels, splits (not in git)
├── models/         # saved runs (not in git)
├── reports/        # label agreement + evaluation reports
├── frontend/       # the React web interface (node_modules and dist not in git)
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
- **The camera path is approximate.** Relative poses are chained with an unknown, drifting scale; it is enough to answer "did the camera ever get to the other side?" and not much more. Coverage is therefore judged on the *azimuth span* of the path, which survives that drift; the 8×3 bin grid is reported and used to name which directions are missing, but counting empty bins turned out to track how the path drifted rather than how much was filmed, so it is not what the label turns on. When too few pairs give a pose, coverage is reported as unreliable rather than guessed.
- **A pan across a flat scene reads as `turned`.** A homography explains a planar scene exactly, so filming a wall while walking past it looks the same as pivoting. For 3DGS the consequence is similar — little usable parallax — but the label names the wrong cause.
- **Thresholds depend on camera and resolution.** Sharpness in particular is only comparable between frames measured at the same width; captures narrower than `resize_width` are measured at their own width, recorded as `measured_width`.
- **Photo bundles with neither EXIF time nor numbered names** are paired by visual similarity, which is less reliable than true capture order.
- **Shiny/glass detection** depends on a small hand-labelled classifier; with none trained, `reflective` is never flagged rather than guessed.
- **Predictions are validated only on the captures in the evaluation report.** The model has no way to know about a failure mode it has never seen. On the synthetic demo set in particular, `mixed_cameras` and `reflective` have no positive examples at all, so nothing about them has been tested.
- **The thresholds have not been calibrated against human judgement.** They are the build guide's starting values, sanity-checked against synthetic captures whose ground truth is known. `noise.frame_max` in particular sits above anything the synthetic set produces. Run the review screen and `scripts/07_check_label_agreement.py` on real captures before trusting a label's precision.

## Contributing
`make lint && make test` must pass (tests cover videos and image bundles, and run offline). Keep thresholds in `configs/` and label names in `capturecheck/labeling/schema.py`. See [`CLAUDE.md`](CLAUDE.md) for the rules that must not be broken.

## Licence and acknowledgements
Apache 2.0. Uses DINOv2 (Meta AI, Apache 2.0), OpenCV, Pillow, PyTorch, scikit-learn, LightGBM, ONNX Runtime, Typer and Streamlit. The walked-versus-turned decision follows the model-selection approach used by ORB-SLAM's initialiser.
