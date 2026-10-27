"""Split scenes into train and test.

The one rule that must never be broken: the split is by **scene**, so every
broken copy of a room lands on the same side as the room itself. Company
captures are the exam and always go to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from capturecheck.config import get_config
from capturecheck.io import dataset
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

TRAIN_FILE = "train_scenes.txt"
TEST_FILE = "test_scenes.txt"


@dataclass
class SplitReport:
    train_scenes: list[str] = field(default_factory=list)
    test_scenes: list[str] = field(default_factory=list)
    company_scenes: list[str] = field(default_factory=list)
    n_train_captures: int = 0
    n_test_captures: int = 0

    def summary(self) -> str:
        return (
            f"train: {len(self.train_scenes)} scenes / {self.n_train_captures} captures\n"
            f"test:  {len(self.test_scenes)} scenes / {self.n_test_captures} captures"
            + (
                f"\ncompany scenes (test only): {', '.join(self.company_scenes)}"
                if self.company_scenes
                else ""
            )
        )

    def as_dict(self) -> dict:
        return {
            "train_scenes": sorted(self.train_scenes),
            "test_scenes": sorted(self.test_scenes),
            "company_scenes": sorted(self.company_scenes),
            "n_train_captures": self.n_train_captures,
            "n_test_captures": self.n_test_captures,
        }


def scene_table() -> dict[str, dict]:
    """scene_id -> how many captures it has and whether it is a company scene."""
    refs = dataset.discover_all()
    scenes: dict[str, dict] = {}
    for ref in refs:
        entry = scenes.setdefault(
            ref.scene_id, {"captures": [], "company": False, "real": 0, "degraded": 0}
        )
        entry["captures"].append(ref.capture_id)
        entry["company"] = entry["company"] or ref.source == "company"
        entry["degraded" if ref.source == "degraded" else "real"] += 1
    return scenes


def make_splits(
    test_share: float = 0.2,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> SplitReport:
    """Write ``train_scenes.txt`` and ``test_scenes.txt``, and check they are disjoint."""
    cfg = get_config()
    out_dir = Path(out_dir) if out_dir else cfg.paths.resolve("splits")

    scenes = scene_table()
    if not scenes:
        raise RuntimeError("no captures found; put some in data/raw first")

    company = sorted(s for s, info in scenes.items() if info["company"])
    rest = sorted(s for s in scenes if s not in set(company))

    rng = np.random.default_rng(seed)
    shuffled = list(rest)
    rng.shuffle(shuffled)

    n_test_extra = int(round(float(test_share) * len(rest)))
    if company:
        # Company scenes already fill part of the test side.
        n_test_extra = max(0, n_test_extra - len(company))
    # Always leave at least one scene on each side when that is possible.
    if len(rest) >= 2:
        n_test_extra = min(n_test_extra, len(rest) - 1)
        if not company:
            n_test_extra = max(1, n_test_extra)

    test = sorted(company + shuffled[:n_test_extra])
    train = sorted(shuffled[n_test_extra:])

    overlap = set(train) & set(test)
    if overlap:  # pragma: no cover - guarded by construction
        raise AssertionError(f"scenes in both splits: {sorted(overlap)}")

    dataset.write_scene_list(train, TRAIN_FILE, path=out_dir)
    dataset.write_scene_list(test, TEST_FILE, path=out_dir)

    report = SplitReport(
        train_scenes=train,
        test_scenes=test,
        company_scenes=company,
        n_train_captures=sum(len(scenes[s]["captures"]) for s in train),
        n_test_captures=sum(len(scenes[s]["captures"]) for s in test),
    )
    log.info("%s", report.summary())
    return report


def read_splits(out_dir: str | Path | None = None) -> tuple[list[str], list[str]]:
    """The scene lists written by :func:`make_splits`."""
    path = Path(out_dir) if out_dir else None
    return (
        dataset.read_scene_list(TRAIN_FILE, path=path),
        dataset.read_scene_list(TEST_FILE, path=path),
    )
