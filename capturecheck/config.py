"""Loads the YAML files in ``configs/`` into typed settings.

No module in :mod:`capturecheck` should contain a threshold of its own: everything
comes from here, so a number can be recalibrated without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV_VAR = "CAPTURECHECK_CONFIG_DIR"


def repo_root() -> Path:
    """Project root: the folder that holds ``configs/``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir():
            return parent
    return here.parent.parent


def config_dir() -> Path:
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = config_dir() / name
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _from_dict(cls: type, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


# --------------------------------------------------------------------------- #
# default.yaml
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    raw: str = "data/raw"
    degraded: str = "data/degraded"
    frames: str = "data/frames"
    features: str = "data/features"
    labels: str = "data/labels"
    splits: str = "data/splits"
    classifier_train: str = "data/classifier_train"
    models: str = "models"
    reports: str = "reports"

    def resolve(self, key: str) -> Path:
        """Absolute path for one of the fields above."""
        return (repo_root() / getattr(self, key)).resolve()


@dataclass
class Runtime:
    time_budget_s: float = 60.0
    num_workers: int = 0
    device: str = "auto"


# --------------------------------------------------------------------------- #
# labeling.yaml
# --------------------------------------------------------------------------- #
@dataclass
class Sampling:
    fps: float = 2.0
    max_frames: int = 600
    pair_gap_frames: int = 2
    max_pair_gap_frames: int = 8
    dino_max_frames: int = 80


@dataclass
class ImagesCfg:
    extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    order: str = "exif_time_then_name"
    unknown_order: str = "dino_nearest"


@dataclass
class BlurCfg:
    sharp_min_laplacian_var: float = 100.0
    video_blur_ratio: float = 0.25


@dataclass
class LightCfg:
    dark_mean: float = 50.0
    bright_mean: float = 210.0
    video_brightness_std: float = 12.0
    video_bad_ratio: float = 0.25
    clip_dark_max: float = 0.10
    clip_bright_max: float = 0.10


@dataclass
class ViewsCfg:
    duplicate_cosine: float = 0.97
    min_unique_views: int = 25
    min_diversity: float = 0.10


@dataclass
class MotionCfg:
    walked_ratio_min: float = 0.5
    min_pair_overlap: float = 0.15
    min_pair_confidence: float = 0.35


@dataclass
class OverlapCfg:
    min_overlap: float = 0.3
    low_overlap_ratio: float = 0.2


@dataclass
class ExifCfg:
    slow_shutter_s: float = 0.01
    high_iso: float = 1600.0
    max_cameras: int = 1
    max_sizes: int = 1


@dataclass
class CoverageCfg:
    azimuth_bins: int = 8
    elevation_bins: int = 3
    min_azimuth_span_deg: float = 180.0
    max_empty_bins: int = 6

    @property
    def n_bins(self) -> int:
        return self.azimuth_bins * self.elevation_bins


@dataclass
class ShinyCfg:
    prob: float = 0.6
    video_ratio: float = 0.3


@dataclass
class TextureCfg:
    min_features: int = 150
    video_ratio: float = 0.4


@dataclass
class NoiseCfg:
    frame_max: float = 6.0
    video_ratio: float = 0.3


@dataclass
class ReviewCfg:
    margin: float = 0.15


@dataclass
class LabelingConfig:
    sampling: Sampling = field(default_factory=Sampling)
    images: ImagesCfg = field(default_factory=ImagesCfg)
    resize_width: int = 960
    blur: BlurCfg = field(default_factory=BlurCfg)
    light: LightCfg = field(default_factory=LightCfg)
    views: ViewsCfg = field(default_factory=ViewsCfg)
    motion: MotionCfg = field(default_factory=MotionCfg)
    overlap: OverlapCfg = field(default_factory=OverlapCfg)
    exif: ExifCfg = field(default_factory=ExifCfg)
    coverage: CoverageCfg = field(default_factory=CoverageCfg)
    shiny: ShinyCfg = field(default_factory=ShinyCfg)
    texture: TextureCfg = field(default_factory=TextureCfg)
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    review: ReviewCfg = field(default_factory=ReviewCfg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelingConfig:
        data = data or {}
        return cls(
            sampling=_from_dict(Sampling, data.get("sampling", {})),
            images=_from_dict(ImagesCfg, data.get("images", {})),
            resize_width=int(data.get("resize_width", 960)),
            blur=_from_dict(BlurCfg, data.get("blur", {})),
            light=_from_dict(LightCfg, data.get("light", {})),
            views=_from_dict(ViewsCfg, data.get("views", {})),
            motion=_from_dict(MotionCfg, data.get("motion", {})),
            overlap=_from_dict(OverlapCfg, data.get("overlap", {})),
            exif=_from_dict(ExifCfg, data.get("exif", {})),
            coverage=_from_dict(CoverageCfg, data.get("coverage", {})),
            shiny=_from_dict(ShinyCfg, data.get("shiny", {})),
            texture=_from_dict(TextureCfg, data.get("texture", {})),
            noise=_from_dict(NoiseCfg, data.get("noise", {})),
            review=_from_dict(ReviewCfg, data.get("review", {})),
        )


# --------------------------------------------------------------------------- #
# train.yaml / degrade.yaml
# --------------------------------------------------------------------------- #
@dataclass
class ModelCfg:
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.3
    max_tokens: int = 80


@dataclass
class StageACfg:
    epochs: int = 30
    variants_per_epoch: int = 2000
    batch_size: int = 16


@dataclass
class StageBCfg:
    epochs: int = 50
    loss: str = "ranking"
    aux_pretext_weight: float = 0.3
    problem_weight: float = 1.0
    batch_size: int = 8
    margin: float = 0.1
    patience: int = 8


@dataclass
class OptimCfg:
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-2


@dataclass
class EvalCfg:
    split: str = "leave_one_scene_out"
    bootstrap: int = 1000


@dataclass
class TrainConfig:
    seed: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    model: ModelCfg = field(default_factory=ModelCfg)
    stage_a: StageACfg = field(default_factory=StageACfg)
    stage_b: StageBCfg = field(default_factory=StageBCfg)
    optimizer: OptimCfg = field(default_factory=OptimCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        data = data or {}
        seed = data.get("seed", [0, 1, 2, 3, 4])
        if isinstance(seed, int):
            seed = [seed]
        return cls(
            seed=list(seed),
            model=_from_dict(ModelCfg, data.get("model", {})),
            stage_a=_from_dict(StageACfg, data.get("stage_a", {})),
            stage_b=_from_dict(StageBCfg, data.get("stage_b", {})),
            optimizer=_from_dict(OptimCfg, data.get("optimizer", {})),
            eval=_from_dict(EvalCfg, data.get("eval", {})),
        )


@dataclass
class DegradeConfig:
    strengths: list[str] = field(default_factory=lambda: ["light", "medium", "heavy"])
    ops: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    def params(self, op: str, strength: str) -> dict[str, Any]:
        """Parameters for one op at one strength (empty dict when not configured)."""
        return dict(self.ops.get(op, {}).get(strength, {}))

    @property
    def jpeg_quality(self) -> int:
        return int(self.output.get("jpeg_quality", 92))

    @property
    def seed(self) -> int:
        return int(self.output.get("seed", 0))


@dataclass
class Config:
    """Everything in ``configs/``, loaded once."""

    paths: Paths = field(default_factory=Paths)
    runtime: Runtime = field(default_factory=Runtime)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    degrade: DegradeConfig = field(default_factory=DegradeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    # Shortcuts used all over the pipeline.
    @property
    def resize_width(self) -> int:
        return self.labeling.resize_width

    @property
    def sampling(self) -> Sampling:
        return self.labeling.sampling


def load_config(config_dir_override: str | Path | None = None) -> Config:
    """Read the YAML files fresh (no caching). Use :func:`get_config` normally."""
    if config_dir_override is not None:
        os.environ[CONFIG_ENV_VAR] = str(config_dir_override)
        get_config.cache_clear()

    default = _load_yaml("default.yaml")
    labeling = _load_yaml("labeling.yaml")
    degrade = _load_yaml("degrade.yaml")
    train = _load_yaml("train.yaml")

    return Config(
        paths=_from_dict(Paths, default.get("paths", {})),
        runtime=_from_dict(Runtime, default.get("runtime", {})),
        labeling=LabelingConfig.from_dict(labeling),
        degrade=_from_dict(DegradeConfig, degrade),
        train=TrainConfig.from_dict(train),
        raw={"default": default, "labeling": labeling, "degrade": degrade, "train": train},
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """The shared config instance (loaded on first use)."""
    return load_config()


def labeling_config() -> LabelingConfig:
    return get_config().labeling
