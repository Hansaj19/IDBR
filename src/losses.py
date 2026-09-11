"""
src/losses.py
===============
Implements:

- ``InfoNCELoss``: temperature-scaled NT-Xent / InfoNCE contrastive loss
  (Eq. 5) using in-batch negative mining -- for a batch of B anchors and
  B positives, each anchor's positive is the matching positive view, and
  the other 2B-2 embeddings in the batch serve as negatives.

- ``TotalIDBRLoss``: combines the identity InfoNCE loss, the degradation
  MSE loss (Eq. 4), and the adversarial cross-entropy loss (Eq. 6) into
  the total objective of Eq. 7:
      L_total = L_id + lambda_1 * L_deg + lambda_2 * L_adv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Supervised Contrastive Loss (SupCon) / InfoNCE with class-aware
    positive aggregation and negative mining.

    Given L2-normalized anchor embeddings ``z_a`` (B, D) and positive
    embeddings ``z_p`` (B, D), along with cell integer IDs ``cell_id_int`` (B,):
    If ``cell_id_int`` is provided, all embeddings sharing the same cell ID
    are treated as positive pairs, and are excluded from the negative
    denominator, eliminating the false-negative penalty of standard InfoNCE.
    If ``cell_id_int`` is None, standard NT-Xent is computed.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_a: torch.Tensor,
        z_p: torch.Tensor,
        cell_id_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_a.size(0)
        device = z_a.device

        z_a = F.normalize(z_a, p=2, dim=-1)
        z_p = F.normalize(z_p, p=2, dim=-1)

        embeddings = torch.cat([z_a, z_p], dim=0)  # (2B, D)
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature  # (2B, 2B)

        # Numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        # Mask out self-contrast
        diag_mask = torch.eye(2 * B, device=device, dtype=torch.bool)

        if cell_id_int is not None:
            labels = torch.cat([cell_id_int, cell_id_int], dim=0)  # (2B,)
            label_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))  # (2B, 2B)
            pos_mask = label_mask & ~diag_mask

            # Denominator: exp(logits) over all non-self elements
            exp_logits = torch.exp(logits) * (~diag_mask).float()
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            pos_counts = pos_mask.sum(1)
            valid_rows = pos_counts > 0
            if valid_rows.sum() == 0:
                # Fall back to matched pair
                positive_indices = torch.arange(2 * B, device=device)
                positive_indices = (positive_indices + B) % (2 * B)
                return F.cross_entropy(logits.masked_fill(diag_mask, float("-inf")), positive_indices)

            mean_log_prob_pos = (pos_mask.float() * log_prob).sum(1)[valid_rows] / pos_counts[valid_rows].float()
            return -mean_log_prob_pos.mean()
        else:
            sim_matrix.masked_fill_(diag_mask, float("-inf"))
            positive_indices = torch.arange(2 * B, device=device)
            positive_indices = (positive_indices + B) % (2 * B)
            return F.cross_entropy(sim_matrix, positive_indices)


@dataclass
class LossWeights:
    lambda1: float = 25.0  # degradation loss weight (scaled to balance contrastive/adv magnitudes)
    lambda2: float = 1.0   # adversarial disentanglement loss weight


class TotalIDBRLoss(nn.Module):
    """Combines identity (Supervised InfoNCE), degradation (Smooth L1),
    and adversarial (cross-entropy) losses per Eq. 7:

        L_total = L_id + lambda1 * L_deg + lambda2 * L_adv
    """

    def __init__(self, temperature: float = 0.07, beta: float = 0.05):
        super().__init__()
        self.id_loss_fn = InfoNCELoss(temperature=temperature)
        self.deg_loss_fn = nn.SmoothL1Loss(beta=beta)
        self.adv_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        z_id_anchor: torch.Tensor,
        z_id_positive: torch.Tensor,
        capacity_hat: torch.Tensor,
        capacity_true: torch.Tensor,
        adv_logits: torch.Tensor,
        cell_id_int: torch.Tensor,
        weights: LossWeights,
    ) -> Dict[str, torch.Tensor]:
        l_id = self.id_loss_fn(z_id_anchor, z_id_positive, cell_id_int=cell_id_int)
        l_deg = self.deg_loss_fn(capacity_hat, capacity_true)
        l_adv = self.adv_loss_fn(adv_logits, cell_id_int)

        l_total = l_id + weights.lambda1 * l_deg + weights.lambda2 * l_adv

        return {
            "loss_total": l_total,
            "loss_id": l_id.detach(),
            "loss_deg": l_deg.detach(),
            "loss_adv": l_adv.detach(),
        }
