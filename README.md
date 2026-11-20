# CaptureCheck

**Find out whether your footage will make a good 3D model — before you spend hours finding out the hard way.**

You film a room. You feed the footage to 3D reconstruction software. Three hours later you discover the footage was no good, and the only fix is to go back and film it again.

CaptureCheck looks at the footage first. In under a minute, on an ordinary laptop, with no graphics card needed.

```
$ capturecheck check ./my_room.mp4

my_room  (video, 412 frames)
Score: 0.42  (likely poor)

Problems:
  • Motion blur    38% of frames are too blurry, worst at 14.0–17.5 s
  • Poor coverage  the camera only got 83° around the room (you want 180° or more)

Time: 41 s
```

---

## The one thing worth understanding

3D reconstruction **does not care whether your video looks nice.** It cares whether the geometry can be solved.

Before any 3D model gets built, the software has to work out where the camera was for every frame. It does that by spotting the same physical point — a table corner, a mark on a wall — in several pictures and working backwards. It is the same trick your two eyes use to judge distance.

This has a consequence that catches people out constantly:

> **A beautiful, razor-sharp 4K video filmed from a tripod that slowly rotates is a total failure.** Every frame was shot from the same spot, so there is nothing to work backwards from.
>
> **A slightly shaky phone video where you walked all the way around the sofa works fine.** Uglier footage, but the camera actually moved.

That is why CaptureCheck measures movement and coverage rather than image quality. An ordinary video-quality tool would give that tripod video top marks. This one catches it.

---

## Try it

```bash
git clone <repo-url> && cd capturecheck
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

capturecheck check path/to/your_video.mp4
```

The first run downloads a vision model (~84 MB). After that it works offline.

Point it at a video, a folder of photos, or a `.zip` of photos:

```bash
capturecheck check room.mp4           # a video
capturecheck check room_photos/       # a folder of photos
capturecheck check room_photos.zip    # a zip of photos
capturecheck labels                   # what every problem name means
```

## Or use it in a browser

Better if you would rather not touch a terminal, or you want to hand it to someone else.

```bash
pip install -e ".[api]"
make ui          # build the web page (needs Node)
make serve       # open http://127.0.0.1:8000
```

Drag a video in, watch it being checked, read the score and what is wrong with it.

---

## What it looks for

| It checks | Because 3D reconstruction needs |
|---|---|
| **Did you walk, or turn on the spot?** | The camera to genuinely move. This is the big one. |
| **Did you film every side?** | Whatever you never pointed the camera at becomes a hole. |
| **Do neighbouring frames overlap?** | Move too fast and the software cannot link one frame to the next. |
| **Are the frames sharp?** | Blurry frames give unreliable reference points. |
| **Is the lighting steady?** | Flickering exposure makes it fit the flicker, not the room. |
| **Is there texture to grab onto?** | Blank walls give it nothing to recognise. |
| **Any mirrors or glass?** | Shiny surfaces look different from every angle, so it invents geometry. |
| **Enough genuinely different views?** | Fifty photos of the same wall is not fifty useful views. |

Each failure comes with **evidence** — a timestamp for a video, a filename for photos — so you know what to re-shoot.

---

## How it works

```
your video or photos
        ↓
  pull out frames          2 per second for video, up to 600
        ↓
  measure each frame       sharpness, brightness, noise, texture
        ↓
  compare neighbours       overlap, and walked-or-turned
        ↓
  map the camera path      which sides did you actually film?
        ↓
  look at the pictures     an AI vision model: duplicates, variety, shiny surfaces
        ↓
  apply the rules          anything past its limit becomes a named problem
        ↓
  score it                 a small trained model turns it all into one number
        ↓
  a score from 0 to 1, a list of problems, and the evidence
```

Everything except the last step is plain measurement. **If the model is ever wrong, the measurements and the evidence are still trustworthy.** That split is deliberate.

| Score | Means |
|---|---|
| 0.75 and up | Likely good |
| 0.55 – 0.75 | Fair |
| 0.35 – 0.55 | Likely poor |
| Below 0.35 | Likely unusable |

---

## Where this project actually stands

Being straight about this is more useful than a flattering summary.

**What works well:** the measurements and the evidence. Spotting blurry captures, finding the gap where you moved too fast, telling a walk from a pivot. That half is solid and useful today.

**What does not yet:** the score. No capture in the dataset has been through real 3D reconstruction, so the model has never seen a true quality number. It learned from footage deliberately damaged in known ways instead. On the held-out test rooms it scores 0.46 where a simple statistical method scores 0.66 — the project's own evaluation report says *"the trained model is not yet earning its place."*

So: **trust the problem list, treat the score as a rough direction.**

The fix is known, not mysterious. Run some captures through 3D reconstruction for real, measure how they turned out, and record it in `data/labels/quality_targets.csv`. That is the one thing that would change what the score means.

---

## What is in here

```
capturecheck/   the code
frontend/       the web page
configs/        every threshold and setting, in plain YAML
data/           your captures and labels (not in git)
models/         trained models (not in git)
tests/          316 tests, all run offline
```

Common commands:

| | |
|---|---|
| `capturecheck check <path>` | Check one capture |
| `capturecheck serve` | Start the web interface |
| `capturecheck label data` | Measure everything in `data/` |
| `capturecheck degrade` | Make deliberately broken copies, for training |
| `capturecheck train` | Train the scoring model |
| `capturecheck evaluate` | Test it and write the report |
| `make lint && make test` | Before committing |

---

## Going deeper

- **[REFERENCE.md](REFERENCE.md)** — how each check works, how the model is trained, all the numbers and all the limits
- **[data/README.md](data/README.md)** — how to organise captures, and the recording rules
- **[frontend/README.md](frontend/README.md)** — how the web page is built
- **[CLAUDE.md](CLAUDE.md)** — the project rules that must not be broken

## Licence

Apache 2.0. Built on DINOv2 (Meta AI), OpenCV, PyTorch, scikit-learn and React — all permissively licensed.

Built for the Rysera Innovations TransitionX 2026 problem statement, *"Predicting reconstruction quality before you train"*.
