"""
src/preprocessing.py
=====================
Loads the NASA PCoE Li-ion battery aging dataset, cleans and chronologically
sorts discharge cycles, resamples each cycle to a fixed length T=100 along
the [Voltage, Current, Temperature] channels, and computes/applies
min-max normalization statistics derived strictly from the training split
(to avoid identity/statistics leakage into validation or test cells).

The NASA PCoE dataset is distributed as per-cell ``.mat`` files (e.g.
``B0005.mat``) containing a top-level struct with a ``cycle`` field. Each
element of ``cycle`` has a ``type`` in {'charge', 'discharge',
'impedance'}, an ``ambient_temperature``, a ``time`` stamp vector, and a
``data`` struct whose fields differ by cycle type. For discharge cycles,
the relevant fields are:

    Voltage_measured, Current_measured, Temperature_measured, Time, Capacity

If no raw ``.mat`` files are found in ``data/raw`` (e.g. because the
NASA PCoE archive has not been downloaded into the environment), this
module falls back to a physically-motivated synthetic generator so that
the rest of the pipeline (segmentation, augmentation, training,
evaluation) remains fully runnable and testable end-to-end. The synthetic
generator is clearly namespaced (``generate_synthetic_nasa_dataset``) and
is never used silently in place of real data -- callers must explicitly
request it or it is used only as an automatic fallback when no raw files
are present, with a logged warning.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CHANNELS = ("voltage", "current", "temperature", "dv_dt")
DEFAULT_T = 100


@dataclass
class Cycle:
    """A single discharge cycle for one physical cell."""

    cell_id: str
    cycle_index: int
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    time: np.ndarray
    capacity: float
    ambient_temperature: float = 24.0

    def raw_length(self) -> int:
        return len(self.time)


@dataclass
class CellRecord:
    """All discharge cycles belonging to a single physical cell, sorted
    chronologically."""

    cell_id: str
    cycles: List[Cycle] = field(default_factory=list)

    def sort_chronologically(self) -> None:
        self.cycles.sort(key=lambda c: c.cycle_index)

    def capacities(self) -> np.ndarray:
        return np.array([c.capacity for c in self.cycles], dtype=np.float32)


# ---------------------------------------------------------------------------
# .mat parsing (real NASA PCoE data)
# ---------------------------------------------------------------------------

def _load_mat_file(path: str):
    """Load a NASA PCoE .mat file using scipy.io, falling back to h5py for
    MATLAB v7.3 files if necessary."""
    from scipy.io import loadmat

    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py

        return h5py.File(path, "r")


def parse_nasa_mat_file(path: str) -> CellRecord:
    """Parse a single NASA PCoE ``.mat`` file into a :class:`CellRecord`
    containing only the discharge cycles (the cycle type used for capacity
    fade / RUL / identity evaluation in this study).

    Parameters
    ----------
    path : str
        Path to a file such as ``data/raw/B0005.mat``.

    Returns
    -------
    CellRecord
        Chronologically-ordered discharge cycles for the cell.
    """
    cell_id = os.path.splitext(os.path.basename(path))[0]
    mat = _load_mat_file(path)

    if cell_id not in mat:
        # some distributions store the variable under a different key;
        # fall back to the first non-dunder key present.
        candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
        if not candidate_keys:
            raise ValueError(f"No usable variable found in {path}")
        struct = mat[candidate_keys[0]]
    else:
        struct = mat[cell_id]

    cycle_struct = struct.cycle
    record = CellRecord(cell_id=cell_id)

    cycle_index = 0
    for entry in np.atleast_1d(cycle_struct):
        if getattr(entry, "type", None) != "discharge":
            continue
        data = entry.data
        voltage = np.asarray(data.Voltage_measured, dtype=np.float64).ravel()
        current = np.asarray(data.Current_measured, dtype=np.float64).ravel()
        temperature = np.asarray(
            data.Temperature_measured, dtype=np.float64
        ).ravel()
        time = np.asarray(data.Time, dtype=np.float64).ravel()
        capacity_arr = np.asarray(data.Capacity).ravel()
        ambient = float(getattr(entry, "ambient_temperature", 24.0))

        min_len = min(len(voltage), len(current), len(temperature), len(time))
        if min_len < 5 or capacity_arr.size == 0:
            # discard degenerate cycles, including the small number of
            # real NASA PCoE discharge entries (e.g. some cycles in
            # B0050/B0052) logged without a Capacity reading
            continue
        capacity = float(capacity_arr[0])

        record.cycles.append(
            Cycle(
                cell_id=cell_id,
                cycle_index=cycle_index,
                voltage=voltage[:min_len],
                current=current[:min_len],
                temperature=temperature[:min_len],
                time=time[:min_len],
                capacity=capacity,
                ambient_temperature=ambient,
            )
        )
        cycle_index += 1

    record.sort_chronologically()
    return record


def load_nasa_pcoe_dataset(raw_dir: str) -> Dict[str, CellRecord]:
    """Load every ``.mat`` file found under ``raw_dir`` into a dict of
    ``cell_id -> CellRecord``. Returns an empty dict if none are found."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.mat")))
    dataset: Dict[str, CellRecord] = {}
    for p in paths:
        try:
            record = parse_nasa_mat_file(p)
            if record.cycles:
                dataset[record.cell_id] = record
        except Exception as exc:  # pragma: no cover - defensive parsing
            logger.warning("Failed to parse %s: %s", p, exc)
    return dataset


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_cycle(cycle: Cycle, max_temp_c: float = 80.0, min_voltage: float = 0.0) -> Optional[Cycle]:
    """Remove physically implausible samples (NaNs, temperature spikes,
    negative/zero voltage) from a single cycle. Returns ``None`` if the
    cycle becomes too short to be usable after cleaning."""
    mask = (
        np.isfinite(cycle.voltage)
        & np.isfinite(cycle.current)
        & np.isfinite(cycle.temperature)
        & (cycle.voltage > min_voltage)
        & (cycle.temperature < max_temp_c)
    )
    if mask.sum() < 5:
        return None
    return Cycle(
        cell_id=cycle.cell_id,
        cycle_index=cycle.cycle_index,
        voltage=cycle.voltage[mask],
        current=cycle.current[mask],
        temperature=cycle.temperature[mask],
        time=cycle.time[mask],
        capacity=cycle.capacity,
        ambient_temperature=cycle.ambient_temperature,
    )


def clean_dataset(dataset: Dict[str, CellRecord]) -> Dict[str, CellRecord]:
    cleaned: Dict[str, CellRecord] = {}
    for cell_id, record in dataset.items():
        new_record = CellRecord(cell_id=cell_id)
        for cycle in record.cycles:
            c = clean_cycle(cycle)
            if c is not None:
                new_record.cycles.append(c)
        new_record.sort_chronologically()
        if new_record.cycles:
            cleaned[cell_id] = new_record
    return cleaned


# ---------------------------------------------------------------------------
# Resampling to fixed length T
# ---------------------------------------------------------------------------

def resample_cycle(cycle: Cycle, T: int = DEFAULT_T) -> np.ndarray:
    """Resample a single (variable-length) discharge cycle onto a fixed
    grid of ``T`` points using linear interpolation over normalized time.

    Returns
    -------
    np.ndarray of shape (3, T)
        Channel order is [Voltage, Current, Temperature].
    """
    t = cycle.time.astype(np.float64)
    if t[-1] == t[0]:
        t_norm = np.linspace(0.0, 1.0, len(t))
    else:
        t_norm = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0.0, 1.0, T)

    out = np.zeros((4, T), dtype=np.float32)
    for i, series in enumerate((cycle.voltage, cycle.current, cycle.temperature)):
        f = interp1d(
            t_norm, series, kind="linear", bounds_error=False,
            fill_value=(series[0], series[-1]),
        )
        out[i] = f(grid).astype(np.float32)

    # 4th channel: numerical derivative dV/dt of the voltage curve
    dt = 1.0 / max(T - 1, 1)
    out[3] = np.gradient(out[0], dt).astype(np.float32)
    return out


def resample_cell_record(record: CellRecord, T: int = DEFAULT_T) -> np.ndarray:
    """Resample every cycle of a cell to shape (T,3,T)... actually returns
    an array of shape (num_cycles, 3, T)."""
    windows = np.stack([resample_cycle(c, T=T) for c in record.cycles], axis=0)
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization (train-split statistics only)
# ---------------------------------------------------------------------------

@dataclass
class NormalizationStats:
    channel_min: np.ndarray  # shape (3,)
    channel_max: np.ndarray  # shape (3,)
    capacity_min: float
    capacity_max: float

    def to_dict(self) -> dict:
        return {
            "channel_min": self.channel_min.tolist(),
            "channel_max": self.channel_max.tolist(),
            "capacity_min": self.capacity_min,
            "capacity_max": self.capacity_max,
        }

    @staticmethod
    def from_dict(d: dict) -> "NormalizationStats":
        return NormalizationStats(
            channel_min=np.array(d["channel_min"], dtype=np.float32),
            channel_max=np.array(d["channel_max"], dtype=np.float32),
            capacity_min=float(d["capacity_min"]),
            capacity_max=float(d["capacity_max"]),
        )


def compute_normalization_stats(
    train_windows: np.ndarray, train_capacities: np.ndarray
) -> NormalizationStats:
    """Compute per-channel min-max statistics strictly on the training
    split. ``train_windows`` has shape (N, 3, T)."""
    channel_min = train_windows.min(axis=(0, 2))
    channel_max = train_windows.max(axis=(0, 2))
    # guard against degenerate (constant) channels
    channel_max = np.where(channel_max > channel_min, channel_max, channel_min + 1e-6)
    return NormalizationStats(
        channel_min=channel_min.astype(np.float32),
        channel_max=channel_max.astype(np.float32),
        capacity_min=float(train_capacities.min()),
        capacity_max=float(max(train_capacities.max(), train_capacities.min() + 1e-6)),
    )


def apply_normalization(windows: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply min-max normalization (to [0, 1]) using pre-computed stats.
    ``windows`` has shape (N, C, T)."""
    C = len(stats.channel_min)
    cmin = stats.channel_min.reshape(1, C, 1)
    cmax = stats.channel_max.reshape(1, C, 1)
    return ((windows - cmin) / (cmax - cmin)).astype(np.float32)


def normalize_capacity(capacity: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    denom = max(stats.capacity_max - stats.capacity_min, 1e-6)
    return ((capacity - stats.capacity_min) / denom).astype(np.float32)


def denormalize_capacity(capacity_norm: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return capacity_norm * (stats.capacity_max - stats.capacity_min) + stats.capacity_min


# ---------------------------------------------------------------------------
# Synthetic fallback generator
# ---------------------------------------------------------------------------

def generate_synthetic_nasa_dataset(
    num_cells: int = 12,
    cycles_per_cell: int = 150,
    raw_length: int = 300,
    seed: int = 42,
) -> Dict[str, CellRecord]:
    """Generate a physically-motivated synthetic analogue of the NASA PCoE
    discharge-cycle dataset for pipeline development and testing when the
    real ``.mat`` archive is not available locally.

    Each cell is given a random initial capacity, a random exponential
    fade rate, and a distinct discharge-voltage "shape" (its identity
    signature) so that the resulting dataset exercises both the
    degradation branch (capacity regression) and the identity branch
    (cell-specific but aging-invariant structure) in the same way the
    real dataset would.
    """
    rng = np.random.default_rng(seed)
    dataset: Dict[str, CellRecord] = {}

    for cell_idx in range(num_cells):
        cell_id = f"SYN{cell_idx:04d}"
        initial_capacity = rng.uniform(1.8, 2.2)
        fade_rate = rng.uniform(0.0015, 0.0045)
        # Per-cell identity signature: a fixed random phase/offset/gain
        # triplet that perturbs the voltage/current/temperature curves in
        # a way that is stable across cycles (this is what the identity
        # branch should learn to recover).
        v_phase = rng.uniform(-0.05, 0.05)
        v_gain = rng.uniform(0.95, 1.05)
        i_offset = rng.uniform(-0.05, 0.05)
        t_offset = rng.uniform(-1.5, 1.5)

        record = CellRecord(cell_id=cell_id)
        for cyc in range(cycles_per_cell):
            frac_life = cyc / max(cycles_per_cell - 1, 1)
            capacity = initial_capacity * np.exp(-fade_rate * cyc) + rng.normal(0, 0.004)
            capacity = max(capacity, 0.3)

            t_norm = np.linspace(0, 1, raw_length)
            # Discharge voltage decays roughly linearly then knees down;
            # aging flattens/steepens the knee, identity signature shifts
            # phase and gain.
            knee = 0.75 - 0.15 * frac_life
            voltage = 4.2 - 1.6 * np.clip((t_norm - v_phase) / max(knee, 1e-3), 0, 1)
            voltage -= 0.4 * np.clip((t_norm - knee) / (1 - knee + 1e-6), 0, 1) ** 2
            voltage = voltage * v_gain + rng.normal(0, 0.01, size=raw_length)

            current = -2.0 + i_offset + rng.normal(0, 0.02, size=raw_length)
            temperature = (
                24.0
                + t_offset
                + 6.0 * t_norm * (1 + 0.3 * frac_life)
                + rng.normal(0, 0.15, size=raw_length)
            )
            time = np.linspace(0, 3000 * (1 - 0.1 * frac_life), raw_length)

            record.cycles.append(
                Cycle(
                    cell_id=cell_id,
                    cycle_index=cyc,
                    voltage=voltage.astype(np.float64),
                    current=current.astype(np.float64),
                    temperature=temperature.astype(np.float64),
                    time=time.astype(np.float64),
                    capacity=float(capacity),
                    ambient_temperature=24.0,
                )
            )
        record.sort_chronologically()
        dataset[cell_id] = record

    return dataset


def load_or_synthesize(raw_dir: str, **synthetic_kwargs) -> Tuple[Dict[str, CellRecord], bool]:
    """Load real NASA PCoE data from ``raw_dir``; if no ``.mat`` files are
    found, fall back to the synthetic generator. Returns the dataset and a
    boolean flag indicating whether synthetic data was used."""
    dataset = load_nasa_pcoe_dataset(raw_dir)
    if dataset:
        return clean_dataset(dataset), False
    logger.warning(
        "No NASA PCoE .mat files found under '%s'. Falling back to the "
        "physically-motivated synthetic dataset generator so the pipeline "
        "remains runnable. Place real .mat files (e.g. B0005.mat) in this "
        "directory to use genuine NASA PCoE data.",
        raw_dir,
    )
    return generate_synthetic_nasa_dataset(**synthetic_kwargs), True
