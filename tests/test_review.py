"""The review screen's logic (the Streamlit page itself is not rendered here)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from capturecheck.labeling import schema


@pytest.fixture(scope="module")
def app():
    """Import app.py without running its ``main()``."""
    pytest.importorskip("streamlit")
    path = Path(__file__).resolve().parents[1] / "capturecheck" / "review" / "app.py"
    source = path.read_text(encoding="utf-8").replace("\nmain()\n", "\n")
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("review_app", loader=None)
    )
    module.__file__ = str(path)
    sys.modules["review_app"] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def frame_table(n: int = 30, **flags) -> pd.DataFrame:
    data = {
        "capture_id": ["cap"] * n,
        "frame_idx": list(range(n)),
        "frame_name": [f"IMG_{i:04d}.jpg" for i in range(n)],
        "time_s": [i * 0.5 for i in range(n)],
        "label_blur": ["SHARP"] * n,
        "label_view": ["UNIQUE_VIEW"] * n,
        "label_light": ["OK_LIGHT"] * n,
        "label_texture": ["OK_TEXTURE"] * n,
        "label_shiny": [0] * n,
        "label_noise": [0] * n,
    }
    for column, (start, end, value) in flags.items():
        for i in range(start, end):
            data[column][i] = value
    return pd.DataFrame(data)


def test_pick_frames_returns_a_dozen(app):
    chosen = app.pick_frames(frame_table(40))
    assert len(chosen) == app.N_THUMBNAILS
    assert chosen == sorted(chosen)
    assert all(0 <= i < 40 for i in chosen)


def test_pick_frames_prefers_the_flagged_ones(app):
    table = frame_table(40, label_blur=(30, 36, "BLURRY"))
    chosen = app.pick_frames(table)
    assert any(30 <= i < 36 for i in chosen), "no blurry frame was shown"


def test_pick_frames_handles_a_short_capture(app):
    chosen = app.pick_frames(frame_table(5))
    assert len(chosen) == 5


def test_pick_frames_on_an_empty_table(app):
    assert app.pick_frames(pd.DataFrame()) == []


def test_frame_notes_name_every_problem(app):
    row = frame_table(1).iloc[0].copy()
    assert app.frame_notes(row) == []
    row["label_blur"] = "BLURRY"
    row["label_shiny"] = 1
    assert set(app.frame_notes(row)) == {"blurry", "shiny"}


def test_caption_shows_time_for_video_and_name_for_photos(app):
    row = frame_table(4).iloc[2]
    assert "1.0 s" in app.caption_for(row, is_video=True)
    assert "IMG_0002.jpg" in app.caption_for(row, is_video=False)
    assert "s · " not in app.caption_for(row, is_video=False)


def test_caption_marks_a_problem_frame(app):
    row = frame_table(4, label_view=(2, 3, "DUPLICATE")).iloc[2]
    assert "duplicate" in app.caption_for(row, is_video=False)


def test_every_capture_label_is_offered(app):
    """The reviewer must be able to change any label the rules can set."""
    assert set(schema.VIDEO_LABEL_NAMES) == set(schema.LABEL_TITLES)


def test_highlights_only_reference_real_frame_labels(app):
    columns = {column for column, _, _ in app.HIGHLIGHTS}
    assert columns <= set(schema.FRAME_LABEL_NAMES)
