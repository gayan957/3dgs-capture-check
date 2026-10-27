"""Padding samples into batches, and choosing batches that contain usable pairs.

The ranking loss can only learn from a batch that holds at least two captures
whose order is known, so batches are built per scene rather than at random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from capturecheck.model.data import FRAME_DIM, PAIR_DIM, CaptureSample


@dataclass
class Batch:
    """A padded batch, plus everything the losses need."""

    frames: Tensor  # (B, T, FRAME_DIM)
    pairs: Tensor  # (B, P, PAIR_DIM)
    global_vec: Tensor  # (B, GLOBAL_DIM)
    frame_mask: Tensor  # (B, T) True where padded
    pair_mask: Tensor  # (B, P) True where padded
    problems: Tensor  # (B, K)
    psnr: Tensor  # (B,)
    damage: Tensor  # (B,)
    scene_ids: list[str]
    capture_ids: list[str]
    sources: list[str]
    pretext_type: Tensor | None = None
    pretext_strength: Tensor | None = None

    def to(self, device: str) -> Batch:
        move = lambda t: t.to(device) if isinstance(t, Tensor) else t  # noqa: E731
        return Batch(
            frames=move(self.frames),
            pairs=move(self.pairs),
            global_vec=move(self.global_vec),
            frame_mask=move(self.frame_mask),
            pair_mask=move(self.pair_mask),
            problems=move(self.problems),
            psnr=move(self.psnr),
            damage=move(self.damage),
            scene_ids=self.scene_ids,
            capture_ids=self.capture_ids,
            sources=self.sources,
            pretext_type=move(self.pretext_type) if self.pretext_type is not None else None,
            pretext_strength=(
                move(self.pretext_strength) if self.pretext_strength is not None else None
            ),
        )

    def __len__(self) -> int:
        return self.frames.shape[0]


def _pad(arrays: list[np.ndarray], dim: int) -> tuple[Tensor, Tensor]:
    """Pad a list of (n_i, dim) arrays to (B, max_n, dim) and give the mask."""
    lengths = [a.shape[0] for a in arrays]
    longest = max(lengths) if lengths else 0
    longest = max(longest, 1)  # the encoder needs at least one token slot
    out = np.zeros((len(arrays), longest, dim), dtype=np.float32)
    mask = np.ones((len(arrays), longest), dtype=bool)
    for i, array in enumerate(arrays):
        n = array.shape[0]
        if n:
            out[i, :n] = array
            mask[i, :n] = False
    return torch.from_numpy(out), torch.from_numpy(mask)


def collate(
    samples: list[CaptureSample],
    pretext_types: list[int] | None = None,
    pretext_strengths: list[float] | None = None,
) -> Batch:
    """Pad a list of samples into one :class:`Batch`."""
    frames, frame_mask = _pad([s.frames for s in samples], FRAME_DIM)
    pairs, pair_mask = _pad([s.pairs for s in samples], PAIR_DIM)
    return Batch(
        frames=frames,
        pairs=pairs,
        global_vec=torch.from_numpy(np.stack([s.global_vec for s in samples])),
        frame_mask=frame_mask,
        pair_mask=pair_mask,
        problems=torch.from_numpy(np.stack([s.problems for s in samples])),
        psnr=torch.tensor([s.psnr for s in samples], dtype=torch.float32),
        damage=torch.tensor([s.damage_rank for s in samples], dtype=torch.float32),
        scene_ids=[s.scene_id for s in samples],
        capture_ids=[s.capture_id for s in samples],
        sources=[s.source for s in samples],
        pretext_type=(
            torch.tensor(pretext_types, dtype=torch.long) if pretext_types is not None else None
        ),
        pretext_strength=(
            torch.tensor(pretext_strengths, dtype=torch.float32)
            if pretext_strengths is not None
            else None
        ),
    )


def scene_batches(
    samples: list[CaptureSample],
    batch_size: int,
    rng: np.random.Generator,
) -> list[list[CaptureSample]]:
    """Batches that each hold several captures of **one** scene.

    This is what makes the ranking loss work: within a scene the clean capture and
    its broken copies have a known order, while across scenes they do not.
    """
    by_scene: dict[str, list[CaptureSample]] = {}
    for sample in samples:
        by_scene.setdefault(sample.scene_id, []).append(sample)

    batches: list[list[CaptureSample]] = []
    for group in by_scene.values():
        clean = [s for s in group if s.damage_rank == 0.0]
        broken = [s for s in group if s.damage_rank > 0.0]
        rng.shuffle(broken)
        if not broken:
            continue
        step = max(1, batch_size - len(clean[:1]))
        for start in range(0, len(broken), step):
            chunk = clean[:1] + broken[start : start + step]
            if len(chunk) >= 2:
                batches.append(chunk)

    # Batches of captures with real PSNR values, which can be ranked across scenes.
    with_psnr = [s for s in samples if s.has_psnr]
    if len(with_psnr) >= 2:
        order = rng.permutation(len(with_psnr))
        for start in range(0, len(order), batch_size):
            chunk = [with_psnr[i] for i in order[start : start + batch_size]]
            if len({round(s.psnr, 6) for s in chunk}) >= 2:
                batches.append(chunk)

    rng.shuffle(batches)
    return batches
