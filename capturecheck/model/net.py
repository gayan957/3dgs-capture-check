"""CaptureQualityNet: a small transformer over frame, pair and global tokens.

Small on purpose. The useful signal is already in the frozen DINOv2 embeddings
and the OpenCV measurements, so what is left to learn is how to weigh them --
and there will only ever be a few hundred captures to learn it from. Two layers
at d=128 is about 0.5M parameters, trains on a laptop in minutes, and runs in
milliseconds on CPU.

The attention weights of the [CLS] token are returned, because "which frames made
the model say this" is the evidence a user can act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from capturecheck.labeling import schema
from capturecheck.model.data import FRAME_DIM, GLOBAL_DIM, PAIR_DIM, PRETEXT_TYPES


@dataclass
class NetOutputs:
    """What one forward pass produces."""

    score: Tensor  # (B,) in 0-1
    problems: Tensor  # (B, K) logits
    pretext_type: Tensor  # (B, n_types) logits
    pretext_strength: Tensor  # (B,) in 0-1
    attention: Tensor | None = None  # (B, T+P+1) weights of the CLS token


class TimeEmbedding(nn.Module):
    """Sinusoidal encoding of where a token sits in the capture (0-1)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half = max(1, dim // 2)
        self.register_buffer(
            "frequencies", torch.exp(torch.linspace(0.0, 4.0, half)) * torch.pi,
            persistent=False,
        )

    def forward(self, position: Tensor) -> Tensor:
        angles = position.unsqueeze(-1) * self.frequencies
        out = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if out.shape[-1] < self.dim:
            out = torch.nn.functional.pad(out, (0, self.dim - out.shape[-1]))
        return out[..., : self.dim]


class CaptureQualityNet(nn.Module):
    """Frame tokens + pair tokens + a global [CLS] token -> score and problems."""

    def __init__(
        self,
        d_model: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        n_problems: int = schema.N_PROBLEMS,
        n_pretext: int = len(PRETEXT_TYPES),
        frame_dim: int = FRAME_DIM,
        pair_dim: int = PAIR_DIM,
        global_dim: int = GLOBAL_DIM,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_problems = n_problems

        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(pair_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.time_embed = TimeEmbedding(d_model)
        # Learned markers so the encoder can tell the three token kinds apart.
        self.kind_embed = nn.Embedding(3, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

        self.score_head = nn.Linear(d_model, 1)
        self.problem_head = nn.Linear(d_model, n_problems)
        self.pretext_type_head = nn.Linear(d_model, n_pretext)
        self.pretext_strength_head = nn.Linear(d_model, 1)

    def forward(
        self,
        frames: Tensor,  # (B, T, FRAME_DIM)
        pairs: Tensor,  # (B, P, PAIR_DIM)
        global_vec: Tensor,  # (B, GLOBAL_DIM)
        frame_mask: Tensor | None = None,  # (B, T) True where padded
        pair_mask: Tensor | None = None,  # (B, P) True where padded
        return_attention: bool = False,
    ) -> NetOutputs:
        batch = frames.shape[0]
        n_frames, n_pairs = frames.shape[1], pairs.shape[1]

        frame_tokens = self.frame_proj(frames)
        if n_frames:
            # The last input column is the relative time of the frame.
            frame_tokens = frame_tokens + self.time_embed(frames[..., -1])
            frame_tokens = frame_tokens + self.kind_embed.weight[0]
        pair_tokens = self.pair_proj(pairs)
        if n_pairs:
            pair_tokens = pair_tokens + self.time_embed(pairs[..., -1])
            pair_tokens = pair_tokens + self.kind_embed.weight[1]

        cls = self.global_proj(global_vec).unsqueeze(1) + self.kind_embed.weight[2]

        tokens = torch.cat([cls, frame_tokens, pair_tokens], dim=1)
        tokens = self.dropout(tokens)

        masks = [torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)]
        masks.append(
            frame_mask
            if frame_mask is not None
            else torch.zeros(batch, n_frames, dtype=torch.bool, device=tokens.device)
        )
        masks.append(
            pair_mask
            if pair_mask is not None
            else torch.zeros(batch, n_pairs, dtype=torch.bool, device=tokens.device)
        )
        padding_mask = torch.cat(masks, dim=1)
        # A row that is entirely padding would make attention produce NaNs.
        padding_mask = padding_mask & ~padding_mask.all(dim=1, keepdim=True)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        summary = encoded[:, 0]

        attention = None
        if return_attention:
            attention = self._cls_attention(encoded, padding_mask)

        return NetOutputs(
            score=torch.sigmoid(self.score_head(summary)).squeeze(-1),
            problems=self.problem_head(summary),
            pretext_type=self.pretext_type_head(summary),
            pretext_strength=torch.sigmoid(self.pretext_strength_head(summary)).squeeze(-1),
            attention=attention,
        )

    def _cls_attention(self, encoded: Tensor, padding_mask: Tensor) -> Tensor:
        """How much each token looks like the summary: usable as evidence weights.

        A dot product against the [CLS] output, rather than the raw attention
        maps, so it stays a single interpretable number per token and survives
        ONNX export.
        """
        query = encoded[:, :1]
        logits = (encoded * query).sum(dim=-1) / (self.d_model**0.5)
        logits = logits.masked_fill(padding_mask, float("-inf"))
        return torch.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
