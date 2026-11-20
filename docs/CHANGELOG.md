# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project maintenance documentation: technical reference and roadmap.

### Changed

- None.

### Fixed

- None.

### Removed

- None.

## [0.1.0] - 2026-11-20

### Added

- Capture-quality analysis for videos, photo directories, and photo ZIP archives.
- OpenCV-based quality, texture, overlap, motion, and coverage checks.
- Frozen DINOv2 embeddings for duplicate-view, diversity, and reflective-surface signals.
- Evidence-backed capture labels, human review support, synthetic degradations, scene-level splits, model training, evaluation, and ONNX export.
- Command-line commands for checking captures, labelling datasets, degradation, training, evaluation, and serving.
- FastAPI upload API with asynchronous, single-worker job processing.
- React/Vite browser interface for uploading a capture and viewing results.
- Test coverage for the pipeline, API, CLI, model, and data-processing behavior.

### Changed

- The score model uses the complete ensemble rather than a single member when scoring a capture.
- Incremental labelling avoids remeasuring previously processed captures.
- The README was rewritten for a non-specialist audience and the browser UI was simplified.

### Fixed

- Coverage is evaluated by camera azimuth span rather than empty-bin count.
- Training and live scoring now use matching model inputs.
- Stage-A training, ONNX export, LightGBM baseline, deterministic seeds, and JSON CLI output were corrected.
- Synthetic gap and noise degradations were adjusted so their generated labels match measured defects.
- Invalid capture paths return a user-facing error instead of a traceback.

### Removed

- Generated feature-cache files were removed from version control.
