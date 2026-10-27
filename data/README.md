# Dataset card

`data/` is not in git: it holds the captures, the cached features and the labels.

## Layout

```
data/
├── raw/<capture_id>/            a capture: video.mp4, or images/, or <id>.zip
│   ├── scene.txt                the room or object (copies of a scene share it)
│   └── source.txt               real | company | degraded (company = test only)
├── degraded/<capture_id>__<op>_<strength>/
│   ├── images/                  the broken copy (always written as photos)
│   ├── scene.txt                same scene as the original
│   ├── source.txt               degraded
│   └── labels.json              exact labels, written by the degrade script
├── frames/<capture_id>/         cached sampled frames (optional)
├── features/<capture_id>.npz    cached DINOv2 embeddings, the pair table and the
│                                coverage grid (the model reads all three, so this
│                                cache is part of the training inputs, not just a
│                                speed-up)
├── labels/
│   ├── frame_labels.csv         one row per frame
│   ├── video_labels.csv         one row per capture (the training table)
│   ├── review_queue.csv         captures waiting for a person
│   ├── human_labels.csv         reviewer decisions
│   └── quality_targets.csv      capture_id,psnr for captures with a known result
├── classifier_train/            ~200 hand-labelled frames for the shiny classifier
│   ├── images/
│   └── labels.csv               image_path,shiny
└── splits/
    ├── train_scenes.txt
    └── test_scenes.txt
```

## Collection protocol

- **Scene id.** Every capture of the same room or object gets the same
  `scene.txt`. The split is by scene, so this is what stops a capture being
  judged against a near-copy of itself.
- **Company captures.** Put `company` in `source.txt`. They are the exam: never
  trained on, never used to tune a threshold.
- **Recording.** Walk around the subject rather than turning on the spot; keep
  neighbouring frames overlapping; keep the exposure locked if the camera allows
  it; cover every side at the height you care about.
- **Quality targets.** Add a row to `quality_targets.csv` whenever a capture has
  actually been through 3DGS: `capture_id,psnr,ssim,notes`. PSNR should be
  measured on held-out *clean* frames of that capture.

## What the labels mean

`capturecheck labels` prints every frame and capture label with its meaning;
the definitions live in `capturecheck/labeling/schema.py`.

## The synthetic demo set

`python scripts/00_make_demo_data.py --clean` writes eight captures of a
textured box on a floor, rendered by projecting each planar face through a
homography so that neighbouring views share stable features:

| Capture | Filmed as | Motion | Arc | Why it is in the set |
|---|---|---|---|---|
| `studio_orbit` | photos | walk round | 360° | The capture you want |
| `kitchen_orbit` | video | walk round | 300° | Same, as a video |
| `office_orbit` | photos | walk round | 240° | One side unseen |
| `hallway_orbit` | video | walk round | 180° | Half the subject unseen |
| `bedroom_orbit` | photos | walk round | 120° | Most of the subject unseen |
| `garage_orbit` | video | walk round | 90° | A corner only, and too few views |
| `lobby_pivot` | video | **turn on the spot** | 160° | No parallax anywhere: the failure `no_walking` exists to catch |
| `company_room_scan` | photos | walk round | 330° | Marked `company`: test only |

Every capture is filmed at the same 2.5° step, so quality varies with *how far
round the camera got*, not with how fast it was filming. The broken copies made
from them are always written as photo bundles, which is why videos only reach
the test set as original captures.

Its `psnr` column is a geometric stand-in — coverage, view count and baseline —
not a 3DGS measurement. A result quoted on this set says the model recovered
that stand-in, and nothing about real reconstruction quality.

## Licence of the data

Captures you record are yours. The synthetic demo set written by
`scripts/00_make_demo_data.py` is generated, not recorded, and carries the same
Apache 2.0 licence as the code. Note that its `psnr` column is a geometric
stand-in, **not** a 3DGS measurement: any result quoted on the demo set says how
well the model recovers that stand-in, nothing more.
