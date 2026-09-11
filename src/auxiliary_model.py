# src/auxiliary_model.py
"""Auxiliary branch model for CALCE lot‑reliability anomaly data.

The model receives a low‑dimensional feature vector per sample:
- `capacity_ratio` (float)
- `cycle_index` (int, normalized)
- `ambient_temp` (float, normalized)

It outputs a single regression value predicting the capacity ratio (or degradation).

Design choices:
- Simple three‑layer MLP with LayerNorm and GELU activations.
- Includes a lightweight Squeeze‑Excitation (SE) block to introduce non‑linear feature interaction – a modestly novel tweak while staying easy to implement.
- Returns a tensor of shape `(batch_size, 1)`.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze‑Excitation block for a vector input.
    Reduces dimensionality by a configurable reduction ratio, then re‑weights the input.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(1, dim // reduction)
        self.fc1 = nn.Linear(dim, reduced_dim, bias=False)
        self.fc2 = nn.Linear(reduced_dim, dim, bias=False)
        self.act = nn.GELU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sig(y)
        return x * y


class AuxiliaryMLP(nn.Module):
    """Three‑layer MLP with SE gating.
    Input dimension = 3 (capacity_ratio, cycle_index, ambient_temp).
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.norm0 = nn.Identity()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.se1 = SEBlock(hidden_dim)
        self.act1 = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.se2 = SEBlock(hidden_dim)
        self.act2 = nn.GELU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.norm0(x)
        x = self.fc1(x)
        x = self.se1(x)
        x = self.act1(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.se2(x)
        x = self.act2(x)
        x = self.norm2(x)
        out = self.fc_out(x)
        # Return shape (batch, output_dim); squeeze only when output_dim == 1
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out


def build_auxiliary_model() -> nn.Module:
    """Factory helper used by the training script.
    Returns an instantiated ``AuxiliaryMLP`` ready for training.
    """
    return AuxiliaryMLP(output_dim=2)
