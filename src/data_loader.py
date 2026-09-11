"""
src/data_loader.py
====================
Cell-aware PyTorch ``Dataset``/``DataLoader`` utilities.

Two dataset classes are provided:

- ``BatteryWindowDataset``: a plain dataset over (window, cell_id_int,
  capacity) triples, used for the degradation branch and for building the
  fingerprint database at evaluation time.

- ``ContrastiveBatteryDataset``: for each requested index, returns an
  anchor window and an augmented positive view drawn from a *different
  cycle of the same physical cell* (as required for the InfoNCE identity
  objective). Negatives are not sampled explicitly -- they are formed
  implicitly from the other items in a mini-batch (batch negative
  mining), which is handled in ``losses.InfoNCELoss``.

Cell-level splitting is implemented in ``split_cells`` to strictly avoid
data leakage: a given physical cell's cycles appear in exactly one of
train/val/test.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import compose_identity_view
from .segmentation import WindowSet


# ---------------------------------------------------------------------------
# Cell-level splitting
# ---------------------------------------------------------------------------

def split_cells(
    cell_ids: Sequence[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split a list of unique cell ids into train/val/test partitions at
    the *cell* level (never split within a cell) to avoid identity
    leakage between splits.

    Returns
    -------
    (train_cells, val_cells, test_cells)
    """
    unique_cells = sorted(set(cell_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_cells).tolist()

    n = len(shuffled)
    n_train = max(int(round(n * train_ratio)), 1)
    n_val = max(int(round(n * val_ratio)), 1) if n - n_train > 1 else max(n - n_train - 1, 0)

    train_cells = shuffled[:n_train]
    val_cells = shuffled[n_train: n_train + n_val]
    test_cells = shuffled[n_train + n_val:]

    if not test_cells:
        # guarantee at least one cell in test if enough cells exist
        if len(val_cells) > 1:
            test_cells = [val_cells.pop()]
        elif len(train_cells) > 2:
            test_cells = [train_cells.pop()]

    return train_cells, val_cells, test_cells


def filter_window_set(window_set: WindowSet, keep_cell_ids: Sequence[str]) -> WindowSet:
    """Return a new WindowSet restricted to the given cell ids, preserving
    the original label_to_int encoding (so cell-id integer labels remain
    consistent across train/val/test splits)."""
    keep = set(keep_cell_ids)
    mask = np.array([cid in keep for cid in window_set.cell_ids])
    ambient = window_set.ambient_temps[mask] if window_set.ambient_temps is not None else None
    return WindowSet(
        windows=window_set.windows[mask],
        cell_ids=window_set.cell_ids[mask],
        cell_id_int=window_set.cell_id_int[mask],
        cycle_indices=window_set.cycle_indices[mask],
        capacities=window_set.capacities[mask],
        label_to_int=window_set.label_to_int,
        ambient_temps=ambient,
    )


# ---------------------------------------------------------------------------
# Plain dataset (degradation branch / fingerprinting)
# ---------------------------------------------------------------------------

def _compute_aux(ambient_temp: float, cycle_index: int) -> torch.Tensor:
    """Normalized ambient temperature and cycle index:
    T_norm = (T_amb - 24) / 20.0, Cyc_norm = cycle_index / 200.0.
    """
    t_norm = (float(ambient_temp) - 24.0) / 20.0
    c_norm = float(cycle_index) / 200.0
    return torch.tensor([t_norm, c_norm], dtype=torch.float32)


class BatteryWindowDataset(Dataset):
    """Simple dataset yielding (window, cell_id_int, capacity, cell_id_str,
    cycle_index, aux) tuples with no augmentation applied."""

    def __init__(self, window_set: WindowSet):
        self.window_set = window_set

    def __len__(self) -> int:
        return len(self.window_set)

    def __getitem__(self, idx: int):
        w = self.window_set
        ambient = float(w.ambient_temps[idx]) if w.ambient_temps is not None else 24.0
        cyc = int(w.cycle_indices[idx])
        return {
            "window": torch.from_numpy(w.windows[idx]).float(),
            "cell_id_int": torch.tensor(w.cell_id_int[idx], dtype=torch.long),
            "capacity": torch.tensor(w.capacities[idx], dtype=torch.float32),
            "cell_id": w.cell_ids[idx],
            "cycle_index": cyc,
            "aux": _compute_aux(ambient, cyc),
        }


# ---------------------------------------------------------------------------
# Contrastive dataset (identity branch)
# ---------------------------------------------------------------------------

class ContrastiveBatteryDataset(Dataset):
    """Yields (anchor, positive, cell_id_int, capacity, aux) tuples where the
    positive is an augmented view drawn from a different cycle of the same
    physical cell whenever the cell has more than one cycle available;
    otherwise the positive is an augmented view of the same cycle.

    Batch negatives for InfoNCE are handled with class-awareness in
    ``losses.InfoNCELoss`` via ``cell_id_int``.
    """

    def __init__(self, window_set: WindowSet, seed: int = 0):
        self.window_set = window_set
        self.rng = np.random.default_rng(seed)

        self._cell_to_indices: Dict[str, List[int]] = {}
        for i, cid in enumerate(window_set.cell_ids):
            self._cell_to_indices.setdefault(cid, []).append(i)

    def __len__(self) -> int:
        return len(self.window_set)

    def _sample_positive_index(self, idx: int) -> int:
        cid = self.window_set.cell_ids[idx]
        candidates = self._cell_to_indices[cid]
        if len(candidates) > 1:
            other = [j for j in candidates if j != idx]
            return int(self.rng.choice(other))
        return idx

    def __getitem__(self, idx: int):
        w = self.window_set
        anchor_raw = w.windows[idx]
        pos_idx = self._sample_positive_index(idx)
        positive_raw = w.windows[pos_idx]

        anchor_view = compose_identity_view(anchor_raw, rng=self.rng)
        positive_view = compose_identity_view(positive_raw, rng=self.rng)

        ambient = float(w.ambient_temps[idx]) if w.ambient_temps is not None else 24.0
        cyc = int(w.cycle_indices[idx])

        return {
            "anchor": torch.from_numpy(anchor_view).float(),
            "positive": torch.from_numpy(positive_view).float(),
            "cell_id_int": torch.tensor(w.cell_id_int[idx], dtype=torch.long),
            "capacity": torch.tensor(w.capacities[idx], dtype=torch.float32),
            "cell_id": w.cell_ids[idx],
            "cycle_index": cyc,
            "aux": _compute_aux(ambient, cyc),
        }


def contrastive_collate_fn(batch: List[dict]) -> dict:
    return {
        "anchor": torch.stack([b["anchor"] for b in batch]),
        "positive": torch.stack([b["positive"] for b in batch]),
        "cell_id_int": torch.stack([b["cell_id_int"] for b in batch]),
        "capacity": torch.stack([b["capacity"] for b in batch]),
        "cell_id": [b["cell_id"] for b in batch],
        "aux": torch.stack([b["aux"] for b in batch]),
    }


def plain_collate_fn(batch: List[dict]) -> dict:
    return {
        "window": torch.stack([b["window"] for b in batch]),
        "cell_id_int": torch.stack([b["cell_id_int"] for b in batch]),
        "capacity": torch.stack([b["capacity"] for b in batch]),
        "cell_id": [b["cell_id"] for b in batch],
        "cycle_index": [b["cycle_index"] for b in batch],
        "aux": torch.stack([b["aux"] for b in batch]),
    }
