"""Capture discovery and the label CSV helpers."""

from __future__ import annotations

import json

import pandas as pd

from capturecheck.io import dataset
from capturecheck.labeling import schema


def test_list_captures_finds_video_and_bundle(tmp_path, tiny_capture_dir, tiny_bundle):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "kitchen_video").symlink_to(tiny_capture_dir)
    (root / "kitchen_photos").symlink_to(tiny_bundle)

    refs = {r.capture_id: r for r in dataset.list_captures(root)}
    assert set(refs) == {"kitchen_video", "kitchen_photos"}
    assert refs["kitchen_video"].input_type == "video"
    assert refs["kitchen_video"].scene_id == "kitchen"  # from scene.txt
    assert refs["kitchen_photos"].input_type == "images"
    assert refs["kitchen_photos"].scene_id == "kitchen_photos"  # defaults to the id


def test_degraded_capture_is_detected(tmp_path, tiny_bundle):
    root = tmp_path / "degraded"
    folder = root / "kitchen__blur_medium"
    (folder / "images").mkdir(parents=True)
    for photo in sorted(tiny_bundle.glob("*.jpg"))[:3]:
        (folder / "images" / photo.name).write_bytes(photo.read_bytes())
    (folder / "scene.txt").write_text("kitchen\n")
    (folder / "labels.json").write_text(
        json.dumps({"motion_blur": 1, "degrade_type": "blur", "degrade_strength": "medium"})
    )

    (ref,) = dataset.list_captures(root)
    assert ref.source == "degraded"
    assert ref.scene_id == "kitchen"
    assert ref.degrade_type == "blur"
    assert ref.degrade_labels["motion_blur"] == 1


def test_label_csv_roundtrip(tmp_labels_dir):
    row = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
    row.update(capture_id="a", scene_id="s", input_type="video", motion_blur=1)
    dataset.write_video_labels(pd.DataFrame([row]), out_dir=tmp_labels_dir)

    back = dataset.read_video_labels(out_dir=tmp_labels_dir)
    assert list(back.columns) == list(schema.VIDEO_CSV_COLUMNS)
    assert back.loc[0, "motion_blur"] == 1


def test_append_human_labels(tmp_labels_dir):
    base = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
    dataset.append_human_labels(
        [{**base, "capture_id": "a", "reviewer": "ann"}], out_dir=tmp_labels_dir
    )
    dataset.append_human_labels(
        [{**base, "capture_id": "b", "reviewer": "bob"}], out_dir=tmp_labels_dir
    )
    df = dataset.read_human_labels(out_dir=tmp_labels_dir)
    assert df["capture_id"].tolist() == ["a", "b"]


def test_scene_lists_roundtrip(tmp_path):
    dataset.write_scene_list(["b", "a", "a"], "train_scenes.txt", path=tmp_path)
    assert dataset.read_scene_list("train_scenes.txt", path=tmp_path) == ["a", "b"]
