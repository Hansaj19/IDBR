"""
src/tamper.py
===============
Synthetic tamper/spoofing attack generator used to evaluate the
out-of-distribution rejection behavior described in Section IV-A: tamper
or spoofing detection is framed as an OOD rejection problem, where
altered telemetry should embed farther from its claimed cell's
fingerprint than genuine telemetry.

Three attack families are implemented, each parameterized by a severity
in the specified range [0.01, 0.02, 0.05, 0.10, 0.20]:

- ``replay_attack``: splices in a segment from a *different* cycle
  (optionally of a different cell) to simulate replaying old/foreign
  telemetry in place of the live signal.
- ``interpolation_attack``: smooths out / flattens a contiguous span of
  the signal via linear interpolation between its endpoints, simulating
  masking of a genuine transient (e.g. to hide an anomaly).
- ``value_injection_attack``: injects out-of-distribution additive noise
  or fixed offsets into a contiguous span, simulating sensor spoofing or
  fault injection.

Severity controls the *fraction of the window affected* for replay and
interpolation attacks, and the *relative magnitude of the injected
perturbation* for the value-injection attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

DEFAULT_SEVERITIES: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)
ATTACK_TYPES = ("replay", "interpolation", "value_injection")


@dataclass
class TamperedSample:
    window: np.ndarray       # (3, T) tampered window
    attack_type: str
    severity: float
    source_cell_id: str
    donor_cell_id: str = None  # only set for replay attacks with a foreign donor


class TamperEngine:
    """Generates tampered/spoofed versions of genuine telemetry windows."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Replay attack
    # ------------------------------------------------------------------
    def replay_attack(
        self, window: np.ndarray, severity: float, donor_window: np.ndarray = None
    ) -> np.ndarray:
        """Replace a contiguous span (fraction = severity of the window
        length) with a segment taken from ``donor_window`` (a different
        cycle, possibly from a different cell). If no donor is supplied, a
        time-reversed / shifted segment of the same window is spliced in
        instead (self-replay), which still constitutes tampering because
        it breaks the natural temporal continuity of the signal.
        """
        T = window.shape[1]
        span = max(int(round(T * severity)), 1)
        start = int(self.rng.integers(0, max(T - span, 1)))

        tampered = window.copy()
        if donor_window is not None:
            donor_start = int(self.rng.integers(0, max(donor_window.shape[1] - span, 1)))
            donor_segment = donor_window[:, donor_start: donor_start + span]
            if donor_segment.shape[1] != span:
                grid_src = np.linspace(0, 1, donor_segment.shape[1])
                grid_dst = np.linspace(0, 1, span)
                donor_segment = np.stack(
                    [np.interp(grid_dst, grid_src, donor_segment[c]) for c in range(3)],
                    axis=0,
                )
            tampered[:, start: start + span] = donor_segment
        else:
            reversed_segment = window[:, ::-1][:, start: start + span]
            tampered[:, start: start + span] = reversed_segment

        return tampered.astype(np.float32)

    # ------------------------------------------------------------------
    # Interpolation attack
    # ------------------------------------------------------------------
    def interpolation_attack(self, window: np.ndarray, severity: float) -> np.ndarray:
        """Flatten a contiguous span (fraction = severity) by linearly
        interpolating between its endpoint values, erasing any genuine
        transient structure within that span (e.g. to mask a fault)."""
        T = window.shape[1]
        span = max(int(round(T * severity)), 2)
        start = int(self.rng.integers(0, max(T - span, 1)))
        end = start + span

        tampered = window.copy()
        for c in range(window.shape[0]):
            start_val = window[c, start]
            end_val = window[c, min(end, T - 1)]
            tampered[c, start:end] = np.linspace(start_val, end_val, end - start)
        return tampered.astype(np.float32)

    # ------------------------------------------------------------------
    # Value injection attack
    # ------------------------------------------------------------------
    def value_injection_attack(self, window: np.ndarray, severity: float) -> np.ndarray:
        """Injects an out-of-distribution offset/noise burst into a
        contiguous span. ``severity`` controls both the affected fraction
        of the window and (scaled) the relative magnitude of the
        injected perturbation, expressed relative to each channel's
        current value range.
        """
        T = window.shape[1]
        span = max(int(round(T * min(severity * 2, 1.0))), 1)
        start = int(self.rng.integers(0, max(T - span, 1)))
        end = start + span

        tampered = window.copy()
        for c in range(window.shape[0]):
            channel_range = float(window[c].max() - window[c].min()) or 1.0
            offset = self.rng.choice([-1.0, 1.0]) * severity * channel_range * 3.0
            burst_noise = self.rng.normal(0, severity * channel_range, size=end - start)
            tampered[c, start:end] = tampered[c, start:end] + offset + burst_noise
        return tampered.astype(np.float32)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def apply(
        self,
        window: np.ndarray,
        attack_type: str,
        severity: float,
        donor_window: np.ndarray = None,
    ) -> np.ndarray:
        if attack_type == "replay":
            return self.replay_attack(window, severity, donor_window=donor_window)
        elif attack_type == "interpolation":
            return self.interpolation_attack(window, severity)
        elif attack_type == "value_injection":
            return self.value_injection_attack(window, severity)
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")


def generate_tampered_dataset(
    windows: np.ndarray,
    cell_ids: Sequence[str],
    attack_types: Sequence[str] = ATTACK_TYPES,
    severities: Sequence[float] = DEFAULT_SEVERITIES,
    seed: int = 0,
) -> List[TamperedSample]:
    """Applies every (attack_type, severity) combination to every input
    window, using a randomly chosen donor window (from a different cell)
    for replay attacks. Returns a flat list of :class:`TamperedSample`.
    """
    rng = np.random.default_rng(seed)
    engine = TamperEngine(rng=rng)
    cell_ids = np.asarray(cell_ids)
    n = len(windows)

    samples: List[TamperedSample] = []
    for i in range(n):
        window = windows[i]
        own_cell = cell_ids[i]
        other_indices = np.where(cell_ids != own_cell)[0]
        donor_idx = int(rng.choice(other_indices)) if len(other_indices) else i
        donor_window = windows[donor_idx]
        donor_cell = cell_ids[donor_idx]

        for attack_type in attack_types:
            for severity in severities:
                if attack_type == "replay":
                    tampered = engine.apply(
                        window, attack_type, severity, donor_window=donor_window
                    )
                    donor_cell_out = donor_cell
                else:
                    tampered = engine.apply(window, attack_type, severity)
                    donor_cell_out = None

                samples.append(
                    TamperedSample(
                        window=tampered,
                        attack_type=attack_type,
                        severity=severity,
                        source_cell_id=str(own_cell),
                        donor_cell_id=str(donor_cell_out) if donor_cell_out is not None else None,
                    )
                )
    return samples
