"""Run the review page for real with Streamlit's AppTest.

The earlier tests call the page's helper functions; this one executes the whole
script the way Streamlit does, so a mistake in the layout or in an API call
fails here instead of at the reviewer's screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from capturecheck.io import dataset
from capturecheck.labeling import schema

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "capturecheck" / "review" / "app.py"


def _video_row(capture_id: str, **overrides) -> dict:
    row = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
    row.update(
        capture_id=capture_id,
        scene_id="kitchen",
        input_type="images",
        source="real",
        n_frames=20,
        blur_ratio=0.26,
        overlap_mean=0.5,
        unique_views=22,
        diversity=0.3,
        walked_ratio=0.9,
        azimuth_span_deg=300.0,
        coverage_empty_bins=2,
        motion_blur=1,
        evidence=json.dumps({"motion_blur": "IMG_0005.jpg to IMG_0009.jpg, 26% blurry"}),
        review=1,
        review_reason="blur_ratio 0.26 is just over the 0.25 threshold",
        label_source="auto",
    )
    row.update(overrides)
    return row


def _frame_rows(capture_id: str, n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "capture_id": [capture_id] * n,
            "frame_idx": range(n),
            "frame_name": [f"IMG_{i:04d}.jpg" for i in range(n)],
            "time_s": [None] * n,
            "sharpness": [500.0] * n,
            "brightness": [120.0] * n,
            "features": [800] * n,
            "noise": [1.0] * n,
            "label_blur": ["BLURRY" if 5 <= i < 10 else "SHARP" for i in range(n)],
            "label_light": ["OK_LIGHT"] * n,
            "label_view": ["UNIQUE_VIEW"] * n,
            "label_texture": ["OK_TEXTURE"] * n,
            "label_shiny": [0] * n,
            "label_noise": [0] * n,
        }
    )


@pytest.fixture()
def labels_dir(tmp_path, tiny_bundle, monkeypatch):
    """A labels folder with one capture waiting for review."""
    folder = tmp_path / "labels"
    folder.mkdir()
    rows = pd.DataFrame([_video_row("kitchen_photos")])
    dataset.write_video_labels(rows, out_dir=folder)
    dataset.write_frame_labels(_frame_rows("kitchen_photos"), out_dir=folder)
    dataset.write_review_queue(
        rows.reindex(columns=list(schema.REVIEW_QUEUE_COLUMNS)), out_dir=folder
    )

    # The app finds captures on disk to build thumbnails.
    import capturecheck.review.app as app_module  # noqa: F401
    from capturecheck.io.dataset import CaptureRef

    monkeypatch.setattr(
        dataset,
        "discover_all",
        lambda *a, **k: [
            CaptureRef(
                capture_id="kitchen_photos", path=tiny_bundle,
                input_type="images", scene_id="kitchen",
            )
        ],
    )
    return folder


def run_app(labels_dir, reviewer: str = "") -> object:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.sidebar.text_input[0].set_value(str(labels_dir))
    if reviewer:
        app.sidebar.text_input[1].set_value(reviewer)
    app.run()
    return app


def test_the_page_runs_without_an_exception(labels_dir):
    app = run_app(labels_dir)
    assert not app.exception, [str(e) for e in app.exception]


def test_it_shows_the_capture_and_why_it_needs_review(labels_dir):
    app = run_app(labels_dir)
    assert not app.exception, [str(e) for e in app.exception]
    text = " ".join(str(w.value) for w in app.title) + " ".join(
        str(w.value) for w in app.warning
    )
    assert "kitchen_photos" in text
    assert "just over the 0.25 threshold" in text


def test_every_capture_label_has_a_checkbox(labels_dir):
    app = run_app(labels_dir)
    assert not app.exception, [str(e) for e in app.exception]
    labels = " ".join(str(c.label) for c in app.checkbox)
    for name in schema.VIDEO_LABEL_NAMES:
        assert schema.LABEL_TITLES[name] in labels, name
    # The automatic label is pre-ticked so confirming is one click.
    blur = next(c for c in app.checkbox if schema.LABEL_TITLES["motion_blur"] in str(c.label))
    assert blur.value is True


def test_saving_without_a_name_is_refused(labels_dir):
    app = run_app(labels_dir)
    app.button[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]
    assert any("name in the sidebar" in str(e.value) for e in app.error)
    assert dataset.read_human_labels(out_dir=labels_dir).empty


def test_saving_writes_the_human_label_and_clears_the_queue(labels_dir):
    app = run_app(labels_dir, reviewer="ann")
    # The reviewer disagrees: the blur is not really a problem. The value is set
    # and submitted in one go, because a form discards unsubmitted edits on a rerun.
    blur = next(c for c in app.checkbox if schema.LABEL_TITLES["motion_blur"] in str(c.label))
    blur.set_value(False)
    app.button[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]

    saved = dataset.read_human_labels(out_dir=labels_dir)
    assert len(saved) == 1
    assert saved.loc[0, "capture_id"] == "kitchen_photos"
    assert saved.loc[0, "reviewer"] == "ann"
    assert saved.loc[0, "motion_blur"] == 0  # the reviewer's decision, not the rule's
    assert saved.loc[0, "label_source"] == "human"
    assert saved.loc[0, "timestamp"]

    # And it is gone from the queue, so nobody reviews it twice.
    assert dataset.read_review_queue(out_dir=labels_dir).empty


def test_an_empty_queue_says_so(tmp_path):
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    app.sidebar.text_input[0].set_value(str(tmp_path))
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert any("review queue is empty" in str(i.value) for i in app.info)
