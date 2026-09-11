"""
src/model.py
=============
PyTorch implementation of the IDBR architecture:

    Raw window (B,3,T)
        -> ConvTransformerEncoder -> h (B, 256)
        -> IdentityBranch(h)      -> z_id  (B, 64), L2-normalized
        -> DegradationBranch(h)   -> z_deg (B, 32), capacity_hat (B, 1)
        -> GRL(z_deg) -> AdversarialClassifier -> cell-id logits (B, num_cells)

Also provides ``BaselineModel`` (shared encoder + regression head only,
no identity/adversarial branches) for the "single-branch baseline" used
as a degradation-accuracy comparison point in the paper's evaluation
protocol.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradientReversalFunction(torch.autograd.Function):
    """Identity on the forward pass; multiplies the incoming gradient by
    ``-lambda_`` on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps :class:`_GradientReversalFunction` as an ``nn.Module`` with a
    mutable ``lambda_`` attribute so the training loop can ramp its
    strength over the course of training (Eq. 6 in the paper)."""

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


def sigmoid_ramp(progress: float, k: float = 10.0) -> float:
    """Sigmoid ramp schedule for lambda in [0, 1] given training progress
    in [0, 1]: lambda(p) = 2 / (1 + exp(-k*p)) - 1. Matches the "sigmoid
    ramp, 0 -> 1 over training" schedule specified for the GRL strength.
    """
    progress = min(max(progress, 0.0), 1.0)
    import math

    return 2.0 / (1.0 + math.exp(-k * progress)) - 1.0


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learned-query attention pooling that aggregates a (B, L, D)
    sequence of hidden states into a single (B, D) vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)               # (B, 1, D)
        k = self.key_proj(x)                            # (B, L, D)
        v = self.value_proj(x)                           # (B, L, D)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B,1,L)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, v)               # (B, 1, D)
        return pooled.squeeze(1)                           # (B, D)


# ---------------------------------------------------------------------------
# Shared temporal encoder
# ---------------------------------------------------------------------------

class ConvTransformerEncoder(nn.Module):
    """3x Conv1D (kernels 7,5,3; channels 64,128,256, each with BatchNorm
    + ReLU) followed by a 2-layer Transformer encoder (d_model=256,
    nhead=4, dropout=0.1) and attention pooling, producing h in R^256.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels=(64, 128, 256),
        kernel_sizes=(7, 5, 3),
        d_model: int = 256,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert conv_channels[-1] == d_model, "final conv width must equal d_model"

        conv_layers = []
        c_in = in_channels
        for c_out, k in zip(conv_channels, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2)
            )
            conv_layers.append(nn.BatchNorm1d(c_out))
            conv_layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv_stack = nn.Sequential(*conv_layers)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.pos_embedding: Optional[nn.Parameter] = None
        self.d_model = d_model
        self.attention_pool = AttentionPooling(d_model)

    def _positional_encoding(self, length: int, device) -> torch.Tensor:
        import math

        position = torch.arange(length, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        conv_out = self.conv_stack(x)              # (B, D, T)
        seq = conv_out.transpose(1, 2)               # (B, T, D)
        seq = seq + self._positional_encoding(seq.size(1), seq.device)
        encoded = self.transformer(seq)               # (B, T, D)
        h = self.attention_pool(encoded)                # (B, D)
        return h


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------

class IdentityBranch(nn.Module):
    """MLP 256 -> 128 -> 64, L2-normalized output z_id."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, p=2, dim=-1)


class DegradationBranch(nn.Module):
    """MLP (256 + aux_dim) -> 128 -> 32 producing z_deg, plus a linear
    regression head predicting normalized capacity from z_deg."""

    def __init__(self, in_dim: int = 256, aux_dim: int = 2, hidden_dim: int = 128, out_dim: int = 32):
        super().__init__()
        self.aux_dim = aux_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + aux_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regression_head = nn.Linear(out_dim, 1)

    def forward(self, h: torch.Tensor, aux: Optional[torch.Tensor] = None):
        if aux is not None and self.aux_dim > 0:
            feat = torch.cat([h, aux], dim=-1)
        elif self.aux_dim > 0:
            feat = torch.cat([h, torch.zeros(h.size(0), self.aux_dim, device=h.device)], dim=-1)
        else:
            feat = h
        z_deg = self.net(feat)
        capacity_hat = self.regression_head(z_deg).squeeze(-1)
        return z_deg, capacity_hat


class AdversarialClassifier(nn.Module):
    """GRL(z_deg) -> MLP -> cell-id logits, used to adversarially
    discourage the degradation embedding from retaining identity
    information (Eq. 6)."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 64, num_cells: int = 10):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_cells),
        )

    def set_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)

    def forward(self, z_deg: torch.Tensor) -> torch.Tensor:
        reversed_z = self.grl(z_deg)
        return self.net(reversed_z)


# ---------------------------------------------------------------------------
# Unified IDBR model
# ---------------------------------------------------------------------------

class IDBRModel(nn.Module):
    """Full Identity-Disentangled Battery Representation model."""

    def __init__(
        self,
        num_cells: int,
        in_channels: int = 4,
        aux_dim: int = 2,
        d_model: int = 256,
        id_dim: int = 64,
        deg_dim: int = 32,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ConvTransformerEncoder(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
        )
        self.identity_branch = IdentityBranch(in_dim=d_model, out_dim=id_dim)
        self.degradation_branch = DegradationBranch(
            in_dim=d_model, aux_dim=aux_dim, out_dim=deg_dim
        )
        self.adversarial_classifier = AdversarialClassifier(
            in_dim=deg_dim, num_cells=num_cells
        )

    def set_grl_lambda(self, lambda_: float) -> None:
        self.adversarial_classifier.set_lambda(lambda_)

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> dict:
        h = self.encoder(x)
        z_id = self.identity_branch(h)
        z_deg, capacity_hat = self.degradation_branch(h, aux=aux)
        adv_logits = self.adversarial_classifier(z_deg)
        return {
            "h": h,
            "z_id": z_id,
            "z_deg": z_deg,
            "capacity_hat": capacity_hat,
            "adv_logits": adv_logits,
        }

    @torch.no_grad()
    def embed_identity(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only helper returning just z_id (used by the
        authentication module: only fθ and Eid are required at deploy
        time)."""
        self.eval()
        h = self.encoder(x)
        return self.identity_branch(h)


# ---------------------------------------------------------------------------
# Baseline model (single-branch, no disentanglement)
# ---------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """Shared encoder + regression head only. Used as the single-branch
    degradation-accuracy baseline referenced in Section VI/VII of the
    paper ("degradation branch's regression accuracy ... should be
    comparable to a single-branch baseline trained only on L_deg")."""

    def __init__(self, in_channels: int = 4, aux_dim: int = 2, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.aux_dim = aux_dim
        self.encoder = ConvTransformerEncoder(in_channels=in_channels, d_model=d_model, dropout=dropout)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model + aux_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> dict:
        h = self.encoder(x)
        if aux is not None and self.aux_dim > 0:
            feat = torch.cat([h, aux], dim=-1)
        elif self.aux_dim > 0:
            feat = torch.cat([h, torch.zeros(h.size(0), self.aux_dim, device=h.device)], dim=-1)
        else:
            feat = h
        capacity_hat = self.regression_head(feat).squeeze(-1)
        return {"h": h, "capacity_hat": capacity_hat}
