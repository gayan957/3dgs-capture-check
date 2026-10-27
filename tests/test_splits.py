"""Scene-level splits: the rule that must never be broken."""

from __future__ import annotations

import pytest

from capturecheck.model import splits as splits_mod


@pytest.fixture()
def fake_scenes(monkeypatch):
    """Six ordinary scenes plus one company scene, each with several captures."""
    table = {
        f"scene{i}": {
            "captures": [f"scene{i}_real", f"scene{i}__blur_light", f"scene{i}__noise_heavy"],
            "company": False,
            "real": 1,
            "degraded": 2,
        }
        for i in range(6)
    }
    table["client_site"] = {
        "captures": ["client_site_scan"], "company": True, "real": 1, "degraded": 0
    }
    monkeypatch.setattr(splits_mod, "scene_table", lambda: table)
    return table


def test_splits_are_disjoint_and_cover_every_scene(fake_scenes, tmp_path):
    report = splits_mod.make_splits(test_share=0.2, seed=0, out_dir=tmp_path)
    assert not set(report.train_scenes) & set(report.test_scenes)
    assert set(report.train_scenes) | set(report.test_scenes) == set(fake_scenes)


def test_company_scenes_are_always_test_only(fake_scenes, tmp_path):
    report = splits_mod.make_splits(test_share=0.2, seed=3, out_dir=tmp_path)
    assert "client_site" in report.test_scenes
    assert "client_site" not in report.train_scenes
    assert report.company_scenes == ["client_site"]


def test_every_copy_of_a_scene_lands_on_the_same_side(fake_scenes, tmp_path):
    """The whole point: a capture is never judged against a near-copy of itself."""
    report = splits_mod.make_splits(test_share=0.4, seed=1, out_dir=tmp_path)
    for scene, info in fake_scenes.items():
        side = "train" if scene in report.train_scenes else "test"
        for capture_id in info["captures"]:
            assert capture_id.startswith(scene)
        assert side in {"train", "test"}
    assert report.n_train_captures + report.n_test_captures == sum(
        len(info["captures"]) for info in fake_scenes.values()
    )


def test_splits_are_written_and_read_back(fake_scenes, tmp_path):
    report = splits_mod.make_splits(test_share=0.2, seed=0, out_dir=tmp_path)
    assert (tmp_path / splits_mod.TRAIN_FILE).is_file()
    assert (tmp_path / splits_mod.TEST_FILE).is_file()
    train, test = splits_mod.read_splits(out_dir=tmp_path)
    assert train == sorted(report.train_scenes)
    assert test == sorted(report.test_scenes)


def test_the_same_seed_gives_the_same_split(fake_scenes, tmp_path):
    first = splits_mod.make_splits(test_share=0.3, seed=7, out_dir=tmp_path)
    second = splits_mod.make_splits(test_share=0.3, seed=7, out_dir=tmp_path)
    assert first.train_scenes == second.train_scenes


def test_both_sides_are_non_empty(fake_scenes, tmp_path):
    report = splits_mod.make_splits(test_share=0.0, seed=0, out_dir=tmp_path)
    assert report.train_scenes and report.test_scenes


def test_no_captures_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(splits_mod, "scene_table", dict)
    with pytest.raises(RuntimeError, match="no captures"):
        splits_mod.make_splits(out_dir=tmp_path)
