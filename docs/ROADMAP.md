# Roadmap

## Current phase

**Prototype / validation.**

Goal: provide fast, evidence-backed diagnostics for 3DGS capture quality while validating whether the learned score is reliable against real reconstruction outcomes.

TODO: confirm the intended next release and target users.

## Milestones

| Priority | Milestone | Status | Evidence / next outcome |
|---:|---|---|---|
| 1 | Build the capture-analysis pipeline | Done | Video and image-bundle loading, OpenCV checks, DINOv2 signals, rules, evidence, CLI, and tests are in the repository. |
| 2 | Create training and evaluation workflow | Done | Degradation, scene-level splitting, ensemble training, export, baselines, and evaluation are implemented. |
| 3 | Deliver a browser interface | Done | React upload UI and FastAPI job API are implemented. |
| 4 | Validate scoring with real reconstruction outcomes | In progress | The repository identifies this as the essential next step: add true 3D reconstruction quality targets to `data/labels/quality_targets.csv`. |
| 5 | Calibrate thresholds on target cameras and captures | Planned | Recalibrate documented label thresholds using representative real captures. |
| 6 | Re-evaluate and select the scoring approach | Planned | Compare the transformer ensemble with the LightGBM feature baseline after adding real targets. |
| 7 | TODO: confirm deployment, user, and release requirements | Planned | No deployment, authentication, or release plan is documented in the codebase. |

## Must-have features

- Reliable capture diagnostics with evidence for blur, lighting, motion, coverage, overlap, texture, noise, duplicate views, mixed cameras, and reflective surfaces.
- Support for videos, photo folders, and photo ZIP files.
- A locally runnable CLI and browser workflow.
- A score trained and evaluated against real reconstruction-quality targets, with a baseline comparison.
- Scene-level train/test separation and test-only company captures.

## Stretch goals

- TODO: confirm. The repository does not define stretch goals beyond the implemented review, training, ONNX export, and browser capabilities.

## Technical debt and known issues

- The score model is not yet justified by the reported evaluation: LightGBM ranks the current test data better.
- The evaluation set is small and has only one video; the report cannot establish video-specific performance.
- Current targets are primarily damage severity rather than enough real PSNR/reconstruction targets.
- DINOv2 first use requires an external model download; offline first runs lose embedding-based measurements.
- API jobs are serial, in-memory, and unauthenticated; this is suitable for local use, not a multi-user service.
- The Makefile is POSIX-oriented despite the project being runnable on Windows through direct commands.
- Capture, feature, and model data are intentionally excluded from Git, so reproducible training needs separately managed data and model artifacts.

## Priority order

1. Gather representative real captures and reconstruction-quality targets.
2. Calibrate thresholds without leaking company/test-only captures into training or tuning.
3. Re-train and compare the ensemble, LightGBM baseline, and simple baselines on scene-level held-out data.
4. Decide whether the learned score should ship, be replaced, or remain advisory.
5. Confirm deployment, authentication, retention, and multi-user requirements before extending the API beyond local use.
