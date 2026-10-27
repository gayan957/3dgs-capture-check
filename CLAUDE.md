# CaptureCheck — project rules

Predict 3D Gaussian Splatting quality **before** training, and say what is wrong
with the capture. A capture is a **video** or an **image bundle** (a folder or
.zip of photos).

## Non-negotiables

1. **Licence: Apache 2.0.** Only Apache/BSD/MIT dependencies. Never add
   Ultralytics YOLO or anything GPL/AGPL.
2. **Split by scene, never by capture.** Every broken copy of a room goes to the
   same side of the split as the room.
3. **Company captures are the exam.** `source=company` is test-only: never train
   on them, never tune thresholds on them.
4. **Quality is measured against clean frames**, never against degraded ones.
5. **No magic numbers in code.** Every threshold lives in `configs/*.yaml` and is
   read through `capturecheck.config`.
6. **Label names live in `capturecheck/labeling/schema.py`.** Every other module
   imports them from there.
7. **60 seconds per capture on a laptop CPU**, including a 5-minute video (600
   frames) and a 600-photo bundle. `utils/timing.py` reports each step.
8. **DINOv2 is frozen.** Only the small head on top is trained.
9. **Training and scoring must see the same inputs.** A capture rebuilt from the
   label CSVs and the same capture measured live must produce identical token
   shapes; `data/features/<id>.npz` carries the pair table and coverage grid for
   exactly this reason, so it is part of the training data, not a cache.
10. **A broken copy must really be broken.** An op that cannot create its defect
    on a given capture produces nothing rather than a copy with a false label.
    `scripts/12_check_degradations.py` checks this.

## Shape of the pipeline

`load_capture` (video or bundle) → OpenCV checks (quality, texture, overlap,
flow, walked-vs-turned, coverage) → DINOv2 on ≤80 frames (duplicates, diversity,
shiny) → rules → frame labels → capture labels + evidence → the small
transformer → a quality score.

## Style

- Type hints, dataclasses for structured returns, `from __future__ import annotations`.
- Tests with pytest for every new behaviour; they must run offline (mock DINOv2).
- `make lint && make test` must pass before a commit.
- Prefer an honest "unknown" over a guess: a measurement that failed should say so
  rather than produce a number nobody can trust.
- Pandas reads a blank CSV cell as NaN, which is **truthy**: `value or default`
  silently yields the string `"nan"`. Use the `_text`/`_number` helpers in
  `model/data.py`. This one skipped the whole pretext stage once.
- **Never seed anything from `hash()` of a string.** It is salted per process, so
  `abs(hash(name)) % n` picks a new number every run. Use `stable_seed` in
  `utils/seed.py`. This one made the demo captures and the broken copies
  different on every run and moved the headline Spearman by 0.14; a single
  process can never see it, so the test has to spawn one.

## Commands

```bash
make setup                      # pip install -e ".[dev]"
capturecheck check <path>       # one capture
capturecheck label data         # everything in data/raw + data/degraded
capturecheck degrade            # broken copies
capturecheck splits             # scene-level split
capturecheck train              # stage A then stage B
capturecheck evaluate           # vs baselines, writes the report
make review                     # the human review screen
```
