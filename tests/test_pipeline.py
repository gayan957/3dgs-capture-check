"""The end-to-end labelling pipeline on the synthetic fixtures.

DINOv2 is switched off (``use_dino=False``) so the tests stay offline and fast;
the DINOv2 path has its own tests in ``test_similarity.py``.
"""

from __future__ import annotations

import json

from capturecheck.io.load import load_capture
from capturecheck.labeling import schema
from capturecheck.labeling.pipeline import label_capture, measure_capture


def test_label_capture_on_a_video(tiny_video):
    result = label_capture(tiny_video, use_dino=False)
    assert result.capture_id == "tiny"
    assert len(result.frame_rows) == 4
    assert set(schema.FRAME_LABEL_NAMES).issubset(result.frame_rows.columns)

    row = result.video_row
    assert row["input_type"] == "video"
    assert row["n_frames"] == 4
    for name in schema.VIDEO_LABEL_NAMES:
        assert row[name] in (0, 1)
    assert isinstance(json.loads(row["evidence"]), dict)
    assert row["label_source"] == "auto"
    assert row["time_s_total"] < 60


def test_label_capture_on_a_photo_bundle(tiny_bundle):
    result = label_capture(tiny_bundle, scene_id="kitchen", use_dino=False)
    row = result.video_row
    assert row["input_type"] == "images"
    assert row["scene_id"] == "kitchen"
    assert row["n_frames"] == 20
    assert row["n_cameras"] == 1
    assert result.frame_rows["frame_name"].iloc[0] == "IMG_0000.jpg"
    # EXIF made it into the frame table.
    assert result.frame_rows["exposure_s"].iloc[0] == 0.008
    assert result.frame_rows["iso"].iloc[0] == 200


def test_zip_and_folder_give_the_same_labels(tiny_bundle, tiny_bundle_zip):
    folder = label_capture(tiny_bundle, use_dino=False).video_row
    zipped = label_capture(tiny_bundle_zip, use_dino=False).video_row
    for name in schema.VIDEO_LABEL_NAMES:
        assert folder[name] == zipped[name], name


def test_degrade_labels_override_the_rules(tiny_bundle):
    result = label_capture(
        tiny_bundle,
        use_dino=False,
        degrade_labels={"motion_blur": 1},
        degrade_type="blur",
        degrade_strength="heavy",
    )
    assert result.video_row["motion_blur"] == 1
    assert result.video_row["label_source"] == "degrade_script"
    assert result.video_row["degrade_type"] == "blur"


def test_measure_capture_fills_every_measurement(tiny_bundle):
    capture = load_capture(tiny_bundle)
    measurements = measure_capture(capture, use_dino=False)
    frame_df = measurements.frame_df
    for column in ("sharpness", "brightness", "noise", "features", "overlap_next"):
        assert column in frame_df.columns
    assert frame_df["sharpness"].notna().all()
    # The last frame has no successor, so its overlap is empty by design.
    assert frame_df["overlap_next"].iloc[:-1].notna().all()
    assert not measurements.pair_df.empty
    assert measurements.timings.total > 0
    assert "quality" in measurements.timings.steps


def test_video_pairs_get_a_flow_speed(tiny_video):
    capture = load_capture(tiny_video)
    measurements = measure_capture(capture, use_dino=False)
    assert measurements.pair_df["flow_speed"].notna().any()


def test_photo_pairs_have_no_flow_speed(tiny_bundle):
    capture = load_capture(tiny_bundle)
    measurements = measure_capture(capture, use_dino=False)
    assert measurements.pair_df["flow_speed"].isna().all()
