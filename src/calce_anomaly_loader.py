# src/calce_anomaly_loader.py
"""Utility to load CALCE lot‑reliability anomaly data.

The CALCE repository provides:
- `Dataset2.mat` containing a variable ``Lot1`` (Nx2) where each row is
  ``[cycle_index, capacity_ratio]``.
- Two Excel files (``Qualified lots.xlsx`` and ``Subsequent lots for ongoing reliability testing.xlsx``)
  with columns for ``cell_id``, ``cycle_index``, ``capacity_ratio``, ``ambient_temperature`` and other metadata.

We expose a single PyTorch ``Dataset`` that yields a dictionary with:

* ``aux_features`` – ``torch.FloatTensor`` of shape ``(3,)`` containing
  ``[capacity_ratio, norm_cycle_idx, norm_ambient_temp]``.
* ``target`` – the same ``capacity_ratio`` (float) used as regression target for the auxiliary branch.

Normalization is performed on‑the‑fly using the mean/std of the NASA training set
(which will be supplied by the caller). If those statistics are not provided,
simple min‑max scaling to ``[0, 1]`` is applied.
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import scipy.io


def _load_mat_file(mat_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load ``Dataset2.mat`` and return tensors for cycle index and capacity ratio.
    Handles possible MATLAB object arrays and ensures a Nx2 float array.
    """
    mat = scipy.io.loadmat(mat_path)
    lot = mat.get('Lot1')
    if lot is None:
        raise KeyError(f"'Lot1' not found in {mat_path}")
    # Convert potential object array to plain numeric array
    if isinstance(lot, np.ndarray):
        if lot.dtype == np.dtype('O'):
            lot = np.array(lot.squeeze().tolist(), dtype=float)
        else:
            lot = lot.astype(float)
    else:
        raise TypeError(f"Unexpected type for 'Lot1' in {mat_path}: {type(lot)}")
    # If data is stored as a 3‑D array (e.g., (cells, cycles, 2)), flatten to Nx2
    if lot.ndim == 3 and lot.shape[2] == 2:
        lot = lot.reshape(-1, 2)
    # Ensure final shape is Nx2
    if lot.ndim != 2 or lot.shape[1] != 2:
        raise ValueError(f"'Lot1' should be Nx2 array, got shape {lot.shape}")
    cycles = torch.from_numpy(lot[:, 0]).float()
    ratios = torch.from_numpy(lot[:, 1]).float()
    return cycles, ratios


def _load_excel_file(xls_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load an Excel file and extract ``cycle_index``, ``capacity_ratio`` and ``ambient_temperature``.
    The function assumes the columns are named exactly as in the CALCE files.
    """
    df = pd.read_excel(xls_path)
    # Expected column names (case‑insensitive match)
    # Map columns by position because the file uses numeric headers
    # Expected layout: column 0 = cycle index, column 1 = capacity ratio,
    # column 2 = ambient temperature (if present). Use the first three columns.
    if df.shape[1] < 2:
        raise ValueError('Excel file does not contain enough columns')
    cycle_col = df.columns[0]
    ratio_col = df.columns[1]
    # ambient temperature may be in column 2; if not, use zeros later
    temp_col = df.columns[2] if df.shape[1] > 2 else None
    # Extract tensors
    cycles = torch.from_numpy(df[cycle_col].values).float()
    ratios = torch.from_numpy(df[ratio_col].values).float()
    temps = torch.from_numpy(df[temp_col].values).float() if temp_col is not None else torch.zeros_like(ratios)
    return cycles, ratios, temps


class CalceAnomalyDataset(Dataset):
    """PyTorch Dataset for the CALCE anomaly data.

    Each sample is a low‑dimensional feature vector used by the auxiliary branch.
    ``aux_features`` = ``[capacity_ratio, norm_cycle_idx, norm_ambient_temp]``.
    ``target`` = capacity ratio (regression target).
    """

    def __init__(
        self,
        root_dir: str,
        cycle_mean: Optional[float] = None,
        cycle_std: Optional[float] = None,
        temp_mean: Optional[float] = None,
        temp_std: Optional[float] = None,
    ):
        """Create the dataset.

        Parameters
        ----------
        root_dir: str
            Directory containing ``Dataset2.mat`` and the two Excel files.
        cycle_mean, cycle_std, temp_mean, temp_std: Optional[float]
            Statistics for normalisation. If ``None`` the dataset will compute
            min‑max scaling to ``[0, 1]`` internally.
        """
        self.root = root_dir
        # Load MAT file
        mat_path = os.path.join(root_dir, "Dataset2.mat")
        cycles_mat, ratios_mat = _load_mat_file(mat_path)
        # Load Excel files
        xl1 = os.path.join(root_dir, "Qualified lots.xlsx")
        xl2 = os.path.join(root_dir, "Subsequent lots for ongoing reliability testing.xlsx")
        cycles_x1, ratios_x1, temps_x1 = _load_excel_file(xl1)
        cycles_x2, ratios_x2, temps_x2 = _load_excel_file(xl2)
        # Concatenate all sources
        self.cycles = torch.cat([cycles_mat, cycles_x1, cycles_x2])
        self.ratios = torch.cat([ratios_mat, ratios_x1, ratios_x2])
        # For the MAT source we have no temperature; fill with zeros
        self.temps = torch.cat([torch.zeros_like(ratios_mat), temps_x1, temps_x2])
        # Normalisation parameters – if not supplied, use min‑max
        if cycle_mean is None or cycle_std is None:
            self.cycle_min = float(self.cycles.min())
            self.cycle_max = float(self.cycles.max())
            self.cycle_mean = None
            self.cycle_std = None
        else:
            self.cycle_mean = cycle_mean
            self.cycle_std = cycle_std
        if temp_mean is None or temp_std is None:
            self.temp_min = float(self.temps.min())
            self.temp_max = float(self.temps.max())
            self.temp_mean = None
            self.temp_std = None
        else:
            self.temp_mean = temp_mean
            self.temp_std = temp_std

    def _norm_cycle(self, x: torch.Tensor) -> torch.Tensor:
        if self.cycle_mean is not None:
            return (x - self.cycle_mean) / (self.cycle_std + 1e-8)
        else:
            return (x - self.cycle_min) / (self.cycle_max - self.cycle_min + 1e-8)

    def _norm_temp(self, x: torch.Tensor) -> torch.Tensor:
        if self.temp_mean is not None:
            return (x - self.temp_mean) / (self.temp_std + 1e-8)
        else:
            return (x - self.temp_min) / (self.temp_max - self.temp_min + 1e-8)

    def __len__(self) -> int:
        return self.ratios.size(0)

    def __getitem__(self, idx: int):
        ratio = self.ratios[idx]
        cycle = self.cycles[idx]
        temp = self.temps[idx]
        aux = torch.stack([
            ratio,
            self._norm_cycle(cycle),
            self._norm_temp(temp),
        ])
        return {"aux_features": aux.float(), "target": ratio.float()}


def build_anomaly_dataset(root_dir: str, cycle_stats: Tuple[float, float] = None, temp_stats: Tuple[float, float] = None) -> CalceAnomalyDataset:
    """Factory function used by the orchestrator script.
    ``cycle_stats`` and ``temp_stats`` are ``(mean, std)`` tuples calculated on the
    NASA training set, allowing consistent normalisation across both domains.
    """
    cycle_mean, cycle_std = (cycle_stats if cycle_stats is not None else (None, None))
    temp_mean, temp_std = (temp_stats if temp_stats is not None else (None, None))
    return CalceAnomalyDataset(root_dir, cycle_mean, cycle_std, temp_mean, temp_std)
