"""The config is the project's single source of thresholds, so it must stay complete."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from capturecheck.config import Config, LabelingConfig, get_config, load_config


def walk_yaml(prefix: str, node: dict, obj) -> list[str]:
    """YAML keys that no dataclass field reads."""
    unread = []
    for key, value in (node or {}).items():
        if not hasattr(obj, key):
            unread.append(f"{prefix}{key}")
            continue
        if isinstance(value, dict):
            unread += walk_yaml(f"{prefix}{key}.", value, getattr(obj, key))
    return unread


def walk_fields(prefix: str, node: dict, obj) -> list[str]:
    """Dataclass fields with no YAML entry, which silently use a code default."""
    missing = []
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            missing += walk_fields(
                f"{prefix}{field.name}.", (node or {}).get(field.name, {}), value
            )
        elif field.name not in (node or {}):
            missing.append(f"{prefix}{field.name}")
    return missing


def test_every_threshold_in_the_yaml_is_read():
    cfg = get_config()
    assert walk_yaml("", cfg.raw["labeling"], cfg.labeling) == []


def test_no_threshold_falls_back_to_a_code_default():
    """A default in the dataclass that is absent from the YAML is a hidden knob."""
    cfg = get_config()
    assert walk_fields("", cfg.raw["labeling"], cfg.labeling) == []


def test_train_config_is_complete():
    cfg = get_config()
    assert walk_yaml("", cfg.raw["train"], cfg.train) == []


def test_config_can_be_overridden_for_a_run(tmp_path, monkeypatch):
    """Recalibrating must be a matter of editing YAML, not code."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "labeling.yaml").write_text(
        yaml.safe_dump({"blur": {"sharp_min_laplacian_var": 999}, "resize_width": 640})
    )
    monkeypatch.setenv("CAPTURECHECK_CONFIG_DIR", str(config_dir))
    cfg = load_config(config_dir)
    assert cfg.labeling.blur.sharp_min_laplacian_var == 999
    assert cfg.resize_width == 640
    # Anything the override leaves out keeps its built-in default.
    assert cfg.labeling.views.duplicate_cosine == 0.97


def test_unknown_yaml_keys_do_not_crash(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "labeling.yaml").write_text(
        yaml.safe_dump({"blur": {"sharp_min_laplacian_var": 50, "invented": 1}, "nonsense": 2})
    )
    monkeypatch.setenv("CAPTURECHECK_CONFIG_DIR", str(config_dir))
    cfg = load_config(config_dir)
    assert cfg.labeling.blur.sharp_min_laplacian_var == 50


def test_paths_resolve_under_the_repo():
    cfg = get_config()
    labels = cfg.paths.resolve("labels")
    assert labels.is_absolute()
    assert labels.name == "labels"


def test_defaults_match_the_build_guide():
    """The starting values the guide specifies, so a drift is deliberate."""
    lab = LabelingConfig()
    assert lab.sampling.fps == 2
    assert lab.sampling.max_frames == 600
    assert lab.sampling.dino_max_frames == 80
    assert lab.resize_width == 960
    assert lab.blur.sharp_min_laplacian_var == 100
    assert lab.blur.video_blur_ratio == 0.25
    assert lab.views.duplicate_cosine == 0.97
    assert lab.views.min_unique_views == 25
    assert lab.motion.walked_ratio_min == 0.5
    assert lab.overlap.min_overlap == 0.3
    assert lab.exif.slow_shutter_s == 0.01
    assert lab.exif.high_iso == 1600
    assert lab.review.margin == 0.15


def test_config_is_a_dataclass_tree():
    assert dataclasses.is_dataclass(Config())
    with pytest.raises(AttributeError):
        _ = get_config().labeling.blur.not_a_threshold
