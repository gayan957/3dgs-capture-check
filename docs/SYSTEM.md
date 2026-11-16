# CaptureCheck — system state

**As of 13 November 2026.** Branch `main`, 38 commits, working tree clean, 277 tests passing.

This document describes what the system does, what state it is in right now, what is
currently stale, and what is known to be wrong with it. Figures quoted here were read
off the artefacts in this repo, not remembered.

---

## 1. What the system is for

Given a **capture** — a video, or a bundle of photos — predict how well 3D Gaussian
Splatting will reconstruct it, **before** spending GPU hours training, and say *what is
wrong with the capture* in terms a person can act on ("you turned on the spot instead of
walking", "the left side of the room was never filmed").

Two outputs per capture:

1. A **quality score** in 0–1.
2. Up to ten **problem labels**, each with a sentence of evidence pointing at the frames
   or timestamps that caused it.

The constraint that shapes everything: **60 seconds per capture on a laptop CPU**, no GPU
required, including a 5-minute video or a 600-photo bundle.

---

## 2. State at a glance

| | |
|---|---|
| Captures in `data/raw/` | **9** — 8 synthetic demo captures + `backroom` (a real video, added 13 Nov) |
| Broken copies in `data/degraded/` | **117** |
| Rows in `video_labels.csv` | **126** (8 real, 1 company, 117 degraded) |
| Rows in `frame_labels.csv` | **8,629** |
| Captures with a real PSNR target | **0** (the 8 demo rows carry a geometric stand-in) |
| Trained model runs | **1** — `models/2026-10-27_run01` |
| Headline result | Spearman **0.69** (95% CI 0.24–0.94) on 17 held-out captures |
| Tests | 277, all passing offline (DINOv2 mocked) |

**Three things are stale right now.** See §8.

---

## 3. The pipeline

```
load_capture (video or bundle)
   └─ video: sample at 2 fps, cap 600 frames, resize to 960 px wide
   └─ bundle: order by EXIF time, else filename
        │
        ├─ OpenCV checks ────────── sharpness, exposure, noise, texture,
        │                           ORB overlap, optical flow,
        │                           walked-vs-turned, coverage
        │
        ├─ DINOv2 on ≤80 frames ── duplicates, diversity, shiny surfaces
        │                           (frozen; only a small head is ever trained)
        │
        ├─ rules ────────────────── frame labels, then capture labels + evidence
        │
        └─ small transformer ────── score + 10 problem heads
```

### What each check measures

| Check | Module | Produces |
|---|---|---|
| Quality | [quality.py](../capturecheck/checks/quality.py) | Laplacian-variance sharpness, brightness mean/std, MAD noise, clipping ratios |
| Texture | [texture.py](../capturecheck/checks/texture.py) | ORB feature count per frame |
| Motion | [motion.py](../capturecheck/checks/motion.py) | pairwise ORB **overlap**, Farneback flow speed, **walked-vs-turned** |
| Coverage | [coverage.py](../capturecheck/checks/coverage.py) | chained relative poses → camera path → azimuth span, empty bins |

**Overlap** is the share of frame A's ORB features that are found again in frame B.
**Walked-vs-turned** fits both a homography and an essential matrix to each pair and
compares their ORB-SLAM-style score share: a pure rotation is explained by the homography
alone, so no parallax means no reconstructable depth. This is what catches the single most
common capture mistake — standing still and panning.

### Where the numbers live

No thresholds in code (rule 5). Everything is in `configs/*.yaml`, read through
`capturecheck.config`. The ones that matter most:

```yaml
sampling:  fps: 2              max_frames: 600      dino_max_frames: 80
resize_width: 960
blur:      sharp_min_laplacian_var: 100             video_blur_ratio: 0.25
light:     dark_mean: 50       video_brightness_std: 12
overlap:   min_overlap: 0.3    low_overlap_ratio: 0.2
views:     duplicate_cosine: 0.97   min_unique_views: 25   min_diversity: 0.10
motion:    walked_ratio_min: 0.5
```

---

## 4. The labels

Six **frame** labels and ten **capture** labels. Names are defined once, in
[schema.py](../capturecheck/labeling/schema.py), and every other module imports them from
there (rule 6).

**Frame:** `label_blur`, `label_light`, `label_view`, `label_shiny`, `label_texture`,
`label_noise`.

**Capture** — these are the ten heads the model predicts:

| Label | Means |
|---|---|
| `motion_blur` | Too many blurry frames or photos |
| `lighting` | Brightness swings, or frames over/under exposed |
| `no_walking` | The camera only turned on the spot |
| `poor_coverage` | Whole directions around the subject were never filmed |
| `too_few_views` | Not enough genuinely different frames |
| `low_overlap` | Neighbouring frames share too little; there are gaps |
| `mixed_cameras` | Several cameras or image sizes in one bundle |
| `reflective` | Many shiny or glass surfaces |
| `low_texture` | Plain surfaces, so features cannot be matched |
| `noisy` | Heavy sensor noise |

Each label carries a **confidence** step: a measurement that lands near its threshold does
not produce a label, it produces a row in `review_queue.csv` with a sentence saying why.
Low diversity on its own *never* produces a label — it only ever asks for a person.

---

## 5. Where training data comes from

Real captures with known 3DGS results are expensive, so most of the signal is manufactured:
take a good capture and **break it in a known way**, which gives an exact label for free.

Seven ops in [ops.py](../capturecheck/degrade/ops.py), three strengths each:
`blur`, `drop_sector`, `rotation_only`, `subsample`, `exposure_jitter`, `noise`, `drop_gap`.

The governing rule (rule 10): **a broken copy must really be broken.** An op that cannot
create its defect on a given capture writes **nothing** rather than a copy carrying a label
that isn't true. `rotation_only` only produces output from a capture that genuinely contains
a rotation-only stretch; `drop_gap` verifies with ORB that each gap it cuts actually falls
below the overlap threshold, and refuses if the bundle is too short.
`scripts/12_check_degradations.py` audits this.

Two hard-won calibration notes are written into `configs/degrade.yaml` itself: JPEG quality
for the `noise` op is kept moderate because compressing hard enough *removes* the noise
being added (so "heavy" came out no noisier than "medium"), and `drop_gap` cuts two photos
per gap because one leaves the junction at ~0.31 against a 0.3 threshold.

---

## 6. The model

A small transformer, deliberately small — the dataset is ~125 captures.

- **Tokens per capture:** up to 80 frame tokens (395-d: DINOv2 384 + 11 measurements),
  pair tokens (9-d), and one global token (51-d).
- **Trunk:** `d_model=128`, 2 layers, 4 heads, dropout 0.3.
- **Heads:** score (sigmoid), 10 problems, plus two pretext heads used only in stage A.

**Two-stage training:**

- **Stage A (pretext)** — 30 epochs, 2,000 variants per epoch generated on the fly by
  damaging *clean features* in memory. The model learns to name which damage was applied and
  how strong. This is where it learns what damage looks like without needing labels.
- **Stage B (ranking)** — 50 epochs of pairwise ranking (margin 0.1) with early stopping,
  plus the problem heads, plus stage A kept on as an auxiliary at weight 0.3.

Ranking, not regression: the useful question is "is this capture better than that one",
and pairs are built **within a scene** so the comparison is like-for-like.

**Ensemble of 5 seeds, leave-one-scene-out.** Each seed holds out a different scene for
validation. `check` averages all five — which is itself a fixed bug, see §10.

---

## 7. Results of record — run `2026-10-27_run01`

17 unseen test captures (test scenes `company_room`, `hallway`), trained on 108 captures
across 6 scenes.

### Ranking

| Method | Spearman (95% CI) |
|---|---|
| **capturecheck** | **0.69** (0.24 – 0.94) |
| `lightgbm_features` | 0.75 (0.40 – 0.94) |
| `orb_matches` | 0.59 (0.14 – 0.91) |
| `frame_count` | 0.31 (−0.19 – 0.73) |
| `mean_blur` | −0.08 (−0.58 – 0.43) |

Validation ranking accuracy averaged 0.765 across the five folds, but the spread is wide:
0.90 on `kitchen`, 0.40 on `lobby`. One fold barely beat a coin toss.

### Problem heads

Five of ten are perfect (`lighting`, `low_overlap`, `low_texture`, `noisy`,
`poor_coverage`: precision and recall 1.00). `motion_blur` is 1.00 / 0.67.

The two that are not working:

- **`too_few_views`** — precision **0.27**. It predicted 11 positives where there were 3.
- **`no_walking`** — 1 prediction, 0 actual positives in the test set.

`mixed_cameras` and `reflective` have no positives in the test set at all, so they are
**untested**, not working.

### Runtime

Measuring a capture: **2.56 s** mean. Scoring it: **3.8 ms** (ONNX). Budget 60 s.

### How to read these numbers honestly

Four caveats, and they are large:

1. **Only 2 test captures have a PSNR target**, so the headline correlation is against
   *damage severity* (clean > light > medium > heavy), not against reconstruction quality.
2. **The PSNR column on the demo set is a geometric stand-in** — coverage, view count and
   baseline — not a 3DGS measurement. A result on this set says the model recovered that
   stand-in, and nothing about real reconstruction quality.
3. **LightGBM on the same summary features ranks as well or better** (0.75 vs 0.69). On
   this test set the transformer is not yet earning its place.
4. **A 17-capture test set cannot separate methods whose intervals overlap.** Every interval
   here overlaps every other. Read the CI, not the point estimate.

The honest summary: the plumbing is correct and verified end to end; the result is not yet
evidence that the approach works on real captures.

---

## 8. The new capture, and what it exposed

`data/raw/backroom/backroom.MOV` — 339 MB, 1920×1080, 29.98 fps, **186.4 s**, scene
`Robotic lab`, source `real` (defaulted; there is no `source.txt`). Sampled to **372 frames**
at 2 fps. Measuring it took **22.66 s** — within budget, but ~9× the demo captures.

What the system said about it:

```
low_overlap = 1          ← the only label raised
review      = 1          "brightness_std 11.7 is just under the 12 threshold"
evidence    worst gap at 125.7–126.2 s (frames 251–252) (1% shared),
            95% of neighbouring pairs share too little
```

Everything else came back clean, and genuinely well: `walked_ratio` **0.91** (you walked,
you didn't pivot), `azimuth_span` **319°**, `sharpness_mean` 3477 with `blur_ratio` **0.0**,
no noise, no duplicates, no shiny surfaces.

### The `low_overlap` label on this capture is an artefact of the sampling rate

I measured overlap on the same file at four sampling rates:

| Sampling | Frames | Mean overlap | Pairs below 0.30 | Verdict at `low_overlap_ratio: 0.2` |
|---|---:|---:|---:|---|
| **2 fps** (the default) | 372 | 0.162 | **94.9 %** | flagged |
| 4 fps | 745 | 0.284 | 54.2 % | flagged |
| 8 fps | 1,490 | 0.419 | 11.8 % | **clean** |
| 15 fps | 2,795 | 0.534 | 2.4 % | **clean** |

Overlap climbs monotonically with the sampling rate and crosses the threshold between 4 and
8 fps. **The video has ample overlap; the 372 frames sampled from it do not.** The label is
true of what the checks looked at and false of the capture.

This is a real design issue, not just a tuning miss. `fps: 2` was calibrated on the synthetic
demo set, which orbits at 2.5° per frame. A handheld walk at normal pace moves much further in
half a second, so the overlap check and the frame sampler are coupled and **nothing in the
system reports that coupling**. Two consequences:

- For real captures, `low_overlap` should be read as *"low overlap at the rate we sampled"*.
- A fix would either raise `fps` for real footage, choose it adaptively from measured flow
  speed, or report the sampling rate alongside the label. None of these is implemented.

A smaller one, visible in the same row: `unique_views` came out as exactly **80**, which is
`dino_max_frames`. Unique views are counted among the DINOv2 frames only, so the measurement
is **capped at 80** for any capture with more frames. That ceiling is plausibly part of why
`too_few_views` has 0.27 precision.

### What is stale because of this capture

1. **`data/splits/` does not contain `Robotic lab`.** It still lists only the 8 original
   scenes. Training reads those files, so the new capture is currently invisible to training.
2. **No broken copies of `backroom` exist** (0 in `data/degraded/`). A scene with one capture
   and no PSNR contributes **no training pairs at all**, because ranking pairs are built
   within a scene. Right now this capture adds nothing to training.
3. **`models/2026-10-27_run01` predates it**, so the published results above do not include it.

To bring it in:

```bash
capturecheck degrade     # broken copies of backroom (it is a video → copies written as bundles)
capturecheck splits      # adds "Robotic lab" to the split
capturecheck train
capturecheck evaluate
```

---

## 9. Open problems (known, not fixed)

| Problem | Why it matters |
|---|---|
| **Human review labels never reach the model.** `review/app.py` writes `human_labels.csv`; only `scripts/07_check_label_agreement.py` reads it. `samples_from_labels` ignores it entirely. | The whole review UI currently has no effect on training. ~20 lines to wire up `label_source == "human"` as an override. |
| **`quality_targets.csv` lives in pipeline-owned `data/labels/`.** | Clearing that directory destroys real PSNR ground truth, and only a quiet "0 with PSNR" warns. This has already happened once in development. Keep a copy elsewhere. |
| **The shiny classifier has never been trained.** `data/classifier_train/` is empty and `models/shiny_classifier.joblib` does not exist. | `reflective` is **structurally dead** — it can never fire. See below. |
| **No capture has a real PSNR target.** | Everything above is validated against a geometric stand-in, not a 3DGS result. This is the single biggest gap in the project. |
| **`unique_views` is capped at 80** by `dino_max_frames`. | The measurement saturates on any capture with more than 80 frames. |
| **Thresholds have never been calibrated against human judgement.** | They are the build guide's starting values, sanity-checked only against synthetic captures. `noise.frame_max` in particular sits above anything the synthetic set produces. |
| **Overlap depends on the sampling rate, unreported.** | §8. |
| **`too_few_views` precision 0.27; `no_walking`, `mixed_cameras`, `reflective` effectively untested.** | Four of ten heads are not demonstrated to work. |

### `reflective` cannot fire, and says 0.0 rather than "unknown"

Traced through: with no saved classifier, `predict_shiny` correctly returns **all-NaN**
([classifiers.py](../capturecheck/labeling/classifiers.py)). But `aggregate.py:150` then does
`shiny_mask = _num(label_shiny) > 0.5`, and `NaN > 0.5` is `False`, so the mask is all-False
and `shiny_ratio` is written as a finite **0.0**. The `reflective` guard at `aggregate.py:300`
checks `np.isfinite(shiny_ratio) and shiny_ratio > threshold` — which passes the finite test
and then fails the comparison, every time.

So every capture is reported as having **0 % shiny surfaces** with full confidence, when the
truth is that nobody measured it. That contradicts the project's own standing preference for
an honest "unknown" over a guess. Two pieces of work:

1. Collect ~200 hand-labelled frames into `data/classifier_train/labels.csv` and run
   `scripts/06_train_shiny_classifier.py`.
2. Independently of that, make `shiny_ratio` NaN when the classifier is absent, so the
   unknown is visible instead of looking like a clean result.

---

## 10. Three bugs, and the rules they left behind

Each of these passed its own local check and was only caught by **comparing two things that
should have matched**. That pattern is the most useful thing this project has produced so far.

**1. Seeding from `hash()` of a string.** Python salts `hash()` per process, so
`abs(hash(name)) % n` picks a different number on every run. The demo captures and the broken
copies were therefore different every run, and the headline Spearman moved by **0.14**. A
single process can never observe this, so the regression test has to spawn a subprocess.
Fixed by `stable_seed` in [seed.py](../capturecheck/utils/seed.py) (BLAKE2b digest).
→ now a written rule in `CLAUDE.md`.

**2. `drop_gap` wrote 21 copies labelled `low_overlap` that weren't.** It cut one gap,
producing a low-overlap ratio of 0.007 against a 0.2 threshold — a direct breach of rule 10.
Rewritten to make as many gaps as the target share requires and to **verify each junction
with ORB**, widening until it actually breaks, and to refuse outright when the bundle is too
short.

**3. `check` was scoring with one model out of five.** The export only wrapped `models[0]`,
while `verify_export` only ever compared ONNX against the wrapper it was handed — so it
passed. PyTorch and ONNX scores differed by up to **0.176**. Fixed by averaging the whole
ensemble in-graph; the backends now agree to 8e-8 with Spearman 1.0000.

A fourth, smaller: a path that isn't a capture used to produce a 30-line traceback. It now
produces a sentence.

---

## 11. Repo map

```
capturecheck/
├── io/          load video / photo bundles / zips, discover captures, read scene.txt
├── checks/      quality, texture, motion, coverage   (OpenCV, no learning)
├── features/    DINOv2 embedder (frozen) + similarity
├── labeling/    schema, frame rules, aggregate, confidence, pipeline
├── degrade/     the seven ops + make_variants
├── model/       data, net, train, losses, evaluate, baselines, export, predict, splits
├── review/      Streamlit review screen
└── utils/       seed, timing, logging
configs/         default, labeling, degrade, train   ← every threshold
scripts/         numbered pipeline steps
tests/           277 tests, offline (DINOv2 mocked)
reports/         evaluation_report.md, label_agreement.md
```

### Commands

```bash
make setup                      # pip install -e ".[dev]"
capturecheck check <path>        # one capture, prints a report, writes nothing
capturecheck label data          # label what is new; --from-scratch remeasures everything
capturecheck degrade             # broken copies
capturecheck splits              # scene-level split (company → test)
capturecheck train               # stage A then stage B
capturecheck evaluate            # vs baselines, writes the report
capturecheck export              # ONNX
capturecheck labels              # print every label with its meaning
make review                      # the human review screen
make lint && make test           # must pass before a commit
```

Two capabilities are reachable **only** through the numbered scripts, not the CLI:
`scripts/10_evaluate.py --colmap-db` (a COLMAP baseline, undocumented in the README) and
`scripts/11_export_onnx.py --out`.

---

## 12. Housekeeping

Unnecessary files identified and **not yet removed**:

- **`.coverage`** — a pytest artefact, committed to git in `b2e372f` and not gitignored.
- **`requirements.txt`** — duplicates `pyproject.toml`'s dependencies and has already
  drifted (it omits `ruff`, which `make lint` needs). Nothing installs from it.
- **`scripts/04_auto_label.py`**, **`scripts/09_train_model.py`**,
  **`scripts/05_review_app.sh`** — thin wrappers over CLI commands that expose nothing extra;
  referenced by nothing.

Deleting the three scripts leaves holes in the advertised "steps 00–12" numbering, so the
README's `scripts/` line needs updating at the same time.

Verified clean: no dead modules (every file under `capturecheck/` is imported by something
other than tests), the notebook stores no outputs and reads thresholds live from the config,
and the similarly-named test files cover genuinely different things.

---

## 13. The non-negotiables

From `CLAUDE.md`, because they explain most of the design decisions above:

1. **Apache 2.0.** Only Apache/BSD/MIT dependencies. Never Ultralytics YOLO or anything GPL/AGPL.
2. **Split by scene, never by capture.** Every broken copy of a room goes to the same side as the room.
3. **Company captures are the exam.** `source=company` is test-only: never trained on, never used to tune a threshold.
4. **Quality is measured against clean frames**, never against degraded ones.
5. **No magic numbers in code.** Every threshold in `configs/*.yaml`.
6. **Label names live in `schema.py`.**
7. **60 seconds per capture on a laptop CPU.**
8. **DINOv2 is frozen.** Only the small head on top is trained.
9. **Training and scoring must see the same inputs.** `data/features/<id>.npz` carries the
   pair table and coverage grid for exactly this reason — it is training data, not a cache.
10. **A broken copy must really be broken.**

And one standing preference, learned the hard way: prefer an honest "unknown" over a guess.
A measurement that failed should say so rather than produce a number nobody can trust.
