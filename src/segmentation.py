"""
src/segmentation.py
=====================
Segments per-cell telemetry into fixed-length windows suitable for the
shared temporal encoder. Two complementary segmentation strategies are
provided:

1. ``resample_cycles_to_windows`` -- treats each full discharge cycle as
   one window and resamples it to length T (this is the strategy used by
   default, matching the (N, 3, 100) input shape specified for IDBR).

2. ``sliding_window_segment`` -- a generic overlapping sliding-window
   segmenter for raw (non-resampled) variable-length telemetry, useful
   when a single cycle should be split into multiple sub-windows (e.g.
   for longer charge/discharge sequences or for streaming/online use).

Both produce a :class:`WindowSet`: parallel arrays of windows, cell ids,
cycle indices and capacity labels, plus an integer-encoded cell-id array
used by the InfoNCE / adversarial classifier losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .preprocessing import CellRecord, resample_cycle, DEFAULT_T


@dataclass
class WindowSet:
    windows: np.ndarray          # (N, C, T) float32
    cell_ids: np.ndarray         # (N,) string
    cell_id_int: np.ndarray      # (N,) int64, stable label encoding
    cycle_indices: np.ndarray    # (N,) int64
    capacities: np.ndarray       # (N,) float32
    label_to_int: Dict[str, int]
    ambient_temps: np.ndarray = None  # (N,) float32

    def __len__(self) -> int:
        return len(self.windows)


def _build_label_encoding(cell_ids: List[str]) -> Dict[str, int]:
    unique_sorted = sorted(set(cell_ids))
    return {cid: i for i, cid in enumerate(unique_sorted)}


def resample_cycles_to_windows(
    records: Dict[str, CellRecord], T: int = DEFAULT_T,
    label_to_int: Dict[str, int] = None,
) -> WindowSet:
    """Build a :class:`WindowSet` where each discharge cycle becomes one
    (C, T) window via linear-interpolation resampling.
    """
    all_windows, all_cell_ids, all_cycle_idx, all_capacity, all_ambient = [], [], [], [], []
    for cell_id, record in records.items():
        for cycle in record.cycles:
            all_windows.append(resample_cycle(cycle, T=T))
            all_cell_ids.append(cell_id)
            all_cycle_idx.append(cycle.cycle_index)
            all_capacity.append(cycle.capacity)
            all_ambient.append(float(getattr(cycle, "ambient_temperature", 24.0)))

    if not all_windows:
        raise ValueError("No cycles available to build windows from.")

    if label_to_int is None:
        label_to_int = _build_label_encoding(all_cell_ids)

    cell_id_int = np.array([label_to_int[c] for c in all_cell_ids], dtype=np.int64)

    return WindowSet(
        windows=np.stack(all_windows, axis=0).astype(np.float32),
        cell_ids=np.array(all_cell_ids),
        cell_id_int=cell_id_int,
        cycle_indices=np.array(all_cycle_idx, dtype=np.int64),
        capacities=np.array(all_capacity, dtype=np.float32),
        label_to_int=label_to_int,
        ambient_temps=np.array(all_ambient, dtype=np.float32),
    )


def sliding_window_segment(
    series: np.ndarray, window_size: int = 100, stride: int = 50
) -> np.ndarray:
    """Generic overlapping sliding-window segmentation of a (3, L) raw
    multichannel series into (num_windows, 3, window_size) windows. If the
    series is shorter than ``window_size`` it is resampled up via linear
    interpolation to exactly one window.

    Parameters
    ----------
    series : np.ndarray
        Shape (3, L) raw [V, I, T] telemetry.
    window_size : int
        Fixed output window length.
    stride : int
        Step size between consecutive window start indices.
    """
    channels, length = series.shape
    if length < window_size:
        grid_src = np.linspace(0, 1, length)
        grid_dst = np.linspace(0, 1, window_size)
        resampled = np.stack(
            [np.interp(grid_dst, grid_src, series[c]) for c in range(channels)],
            axis=0,
        )
        return resampled[np.newaxis, ...].astype(np.float32)

    starts = list(range(0, length - window_size + 1, stride))
    if not starts or starts[-1] != length - window_size:
        starts.append(length - window_size)

    windows = np.stack(
        [series[:, s: s + window_size] for s in starts], axis=0
    )
    return windows.astype(np.float32)


def segment_cell_record_sliding(
    record: CellRecord, window_size: int = 100, stride: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply :func:`sliding_window_segment` to every cycle of a cell,
    returning concatenated windows, their originating cycle index, and the
    (repeated) capacity label for that cycle."""
    all_windows, all_cycle_idx, all_capacity = [], [], []
    for cycle in record.cycles:
        raw = np.stack([cycle.voltage, cycle.current, cycle.temperature], axis=0)
        windows = sliding_window_segment(raw, window_size=window_size, stride=stride)
        all_windows.append(windows)
        all_cycle_idx.extend([cycle.cycle_index] * len(windows))
        all_capacity.extend([cycle.capacity] * len(windows))

    return (
        np.concatenate(all_windows, axis=0).astype(np.float32),
        np.array(all_cycle_idx, dtype=np.int64),
        np.array(all_capacity, dtype=np.float32),
    )
