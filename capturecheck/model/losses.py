"""Losses: ranking for the score, weighted BCE for the problems, pretext for stage A.

The score is trained by **ranking**, not regression, because what the user needs
is "is this capture better than that one?" and because PSNR values from different
scenes are not comparable: a textured room scores higher than a white wall however
well either was filmed. Pairs are only formed where the true order is known:

* the same scene, clean versus a broken copy (clean is better);
* the same scene, two broken copies of different strength (lighter is better);
* two captures with real PSNR values (higher PSNR is better).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def pairwise_ranking_loss(
    scores: Tensor,
    better: Tensor,
    worse: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Margin loss over index pairs where ``better`` should outscore ``worse``."""
    if better.numel() == 0:
        return scores.new_zeros(())
    difference = scores[better] - scores[worse]
    return F.relu(margin - difference).mean()


def build_pairs(
    scene_ids: list[str],
    psnr: Tensor,
    damage: Tensor,
    sources: list[str],
) -> tuple[Tensor, Tensor]:
    """Index pairs whose true order is known, as ``(better, worse)``.

    ``damage`` is 0 for a clean capture and rises with the strength of the damage.
    """
    better: list[int] = []
    worse: list[int] = []
    n = len(scene_ids)

    for i in range(n):
        for j in range(i + 1, n):
            # 1) Real PSNR beats everything else we know.
            pi, pj = float(psnr[i]), float(psnr[j])
            if torch.isfinite(psnr[i]) and torch.isfinite(psnr[j]) and abs(pi - pj) > 1e-6:
                if pi > pj:
                    better.append(i)
                    worse.append(j)
                else:
                    better.append(j)
                    worse.append(i)
                continue

            # 2) Same scene: less damage is better. Across scenes this says nothing.
            if scene_ids[i] != scene_ids[j]:
                continue
            di, dj = float(damage[i]), float(damage[j])
            if abs(di - dj) < 1e-6:
                continue
            if di < dj:
                better.append(i)
                worse.append(j)
            else:
                better.append(j)
                worse.append(i)

    device = psnr.device
    return (
        torch.tensor(better, dtype=torch.long, device=device),
        torch.tensor(worse, dtype=torch.long, device=device),
    )


def problem_bce_loss(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Class-weighted BCE over the K problem heads.

    Rare problems need the weight: with 10 labels and a few hundred captures, some
    appear in under 5% of them and an unweighted loss just predicts "no".
    """
    if logits.numel() == 0:
        return logits.new_zeros(())
    weight = None
    if class_weights is not None:
        # Weight the positive side of each label.
        weight = torch.ones_like(targets)
        weight = weight + targets * (class_weights.unsqueeze(0) - 1.0)
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)


def positive_weights(targets: Tensor, cap: float = 10.0) -> Tensor:
    """Per-label weight ``n_negative / n_positive``, capped so it cannot explode."""
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    weights = torch.where(
        positives > 0, negatives / positives.clamp(min=1.0), torch.ones_like(positives)
    )
    return weights.clamp(min=1.0, max=cap)


def pretext_loss(
    type_logits: Tensor,
    type_targets: Tensor,
    strength_pred: Tensor,
    strength_targets: Tensor,
    strength_weight: float = 0.5,
) -> Tensor:
    """Stage A: name the damage and say how strong it is."""
    if type_logits.numel() == 0:
        return type_logits.new_zeros(())
    classification = F.cross_entropy(type_logits, type_targets)
    regression = F.mse_loss(strength_pred, strength_targets)
    return classification + strength_weight * regression


def score_mse_loss(scores: Tensor, psnr: Tensor, lo: float = 18.0, hi: float = 32.0) -> Tensor:
    """Optional extra term: pull the score towards normalised PSNR where known.

    Only used when ``stage_b.loss`` is ``ranking+mse``; the mapping from PSNR to
    0-1 is a convenience, not a claim about absolute quality.
    """
    known = torch.isfinite(psnr)
    if not bool(known.any()):
        return scores.new_zeros(())
    target = ((psnr[known] - lo) / (hi - lo)).clamp(0.0, 1.0)
    return F.mse_loss(scores[known], target)
