"""The 60-second budget: a capture must stay affordable per frame.

The budget is 60 s for a 600-frame capture, so the cost that matters is the cost
*per frame* at the measured width. This runs a smaller capture at the real
resize width and checks the per-frame cost leaves plenty of headroom, which
catches a check that quietly became quadratic without needing a 600-frame
fixture in the suite.
"""

from __future__ import annotations

import time

import pytest

from capturecheck.config import get_config
from capturecheck.io.load import load_capture
from capturecheck.labeling.pipeline import measure_capture

#: 60 s / 600 frames = 0.1 s per frame. Half of that is the line to hold.
MAX_SECONDS_PER_FRAME = 0.05


@pytest.fixture(scope="module")
def wide_bundle(tmp_path_factory):
    """60 photos at the real resize width, with parallax between them."""
    from conftest import translation_frames, write_photo

    cfg = get_config()
    folder = tmp_path_factory.mktemp("budget") / "wide"
    folder.mkdir()
    frames = translation_frames(60, width=cfg.resize_width, height=540, step=6)
    for i, frame in enumerate(frames):
        write_photo(folder / f"IMG_{i:04d}.jpg", frame, when=None)
    return folder


@pytest.mark.slow
def test_measuring_stays_within_the_per_frame_budget(wide_bundle):
    capture = load_capture(wide_bundle)
    assert capture.shape[1] == get_config().resize_width

    start = time.perf_counter()
    measurements = measure_capture(capture, use_dino=False)
    elapsed = time.perf_counter() - start

    per_frame = elapsed / capture.n_frames
    assert per_frame < MAX_SECONDS_PER_FRAME, (
        f"{per_frame * 1000:.0f} ms per frame leaves no room for a 600-frame capture "
        f"inside the {get_config().runtime.time_budget_s:.0f} s budget.\n"
        f"{measurements.timings.report()}"
    )


@pytest.mark.slow
def test_no_single_step_dominates(wide_bundle):
    """One step eating the whole budget is the shape of a performance bug."""
    capture = load_capture(wide_bundle)
    measurements = measure_capture(capture, use_dino=False)
    steps = measurements.timings.steps
    worst = max(steps, key=steps.get)
    assert steps[worst] < 0.75 * measurements.timings.total, (
        f"`{worst}` takes {steps[worst] / measurements.timings.total:.0%} of the time:\n"
        f"{measurements.timings.report()}"
    )


def test_the_budget_is_configured():
    assert get_config().runtime.time_budget_s == 60
