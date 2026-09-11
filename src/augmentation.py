"""
src/augmentation.py
=====================
Augmentations used to generate distinct "views" of the same physical
cell's telemetry for the contrastive identity objective, and to make the
identity branch invariant to legitimate measurement/operating-condition
variation rather than to window-specific artifacts. Implements:

- additive sensor noise (per-channel, magnitude consistent with realistic
  measurement error)
- small random temporal warping (non-linear resampling of the time axis)
- random cropping followed by resizing back to the fixed window length T

All functions operate on a single window of shape (3, T) and return an
array of the same shape.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Realistic per-channel measurement noise standard deviations, expressed
# in the *normalized* [0, 1] scale used after min-max normalization.
DEFAULT_NOISE_STD = (0.01, 0.015, 0.008, 0.02)  # voltage, current, temperature, dV/dt


def add_sensor_noise(
    window: np.ndarray,
    noise_std: Tuple[float, ...] = DEFAULT_NOISE_STD,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to each channel independently.

    Parameters
    ----------
    window : np.ndarray
        Shape (C, T).
    noise_std : tuple of floats
        Standard deviation of additive noise per channel.
    """
    rng = rng or np.random.default_rng()
    noisy = window.copy()
    C = window.shape[0]
    stds = list(noise_std)
    if len(stds) < C:
        stds.extend([0.01] * (C - len(stds)))
    for c in range(C):
        noisy[c] += rng.normal(0.0, stds[c], size=window.shape[1])
    return noisy.astype(np.float32)


def time_warp(
    window: np.ndarray, warp_strength: float = 0.1, rng: np.random.Generator = None
) -> np.ndarray:
    """Apply a smooth, monotonic non-linear warp to the time axis by
    perturbing a small number of control points and interpolating the
    original signal at the warped locations, then resampling back onto a
    uniform grid of the original length.

    Parameters
    ----------
    window : np.ndarray
        Shape (3, T).
    warp_strength : float
        Maximum fractional perturbation applied to control points.
    """
    rng = rng or np.random.default_rng()
    T = window.shape[1]
    num_control = 5
    control_x = np.linspace(0, 1, num_control)
    jitter = rng.uniform(-warp_strength, warp_strength, size=num_control) / num_control
    warped_control_x = np.clip(control_x + jitter, 0, 1)
    warped_control_x = np.sort(warped_control_x)
    warped_control_x[0], warped_control_x[-1] = 0.0, 1.0

    uniform_grid = np.linspace(0, 1, T)
    # Map each point on the uniform grid to a warped source time via
    # interpolation of the control-point mapping.
    warp_map = np.interp(uniform_grid, control_x, warped_control_x)

    out = np.zeros_like(window)
    for c in range(window.shape[0]):
        out[c] = np.interp(warp_map, uniform_grid, window[c])
    return out.astype(np.float32)


def random_crop(
    window: np.ndarray, crop_ratio: float = 0.8, rng: np.random.Generator = None
) -> np.ndarray:
    """Randomly crop a contiguous sub-segment of the window (of length
    ``crop_ratio * T``) and resize it back to length T via linear
    interpolation.

    Parameters
    ----------
    window : np.ndarray
        Shape (3, T).
    crop_ratio : float
        Fraction of the original length to keep, in (0, 1].
    """
    rng = rng or np.random.default_rng()
    T = window.shape[1]
    crop_len = max(int(round(T * crop_ratio)), 8)
    if crop_len >= T:
        return window.copy()

    start = rng.integers(0, T - crop_len + 1)
    cropped = window[:, start: start + crop_len]

    src_grid = np.linspace(0, 1, crop_len)
    dst_grid = np.linspace(0, 1, T)
    out = np.zeros_like(window)
    for c in range(window.shape[0]):
        out[c] = np.interp(dst_grid, src_grid, cropped[c])
    return out.astype(np.float32)


def compose_identity_view(
    window: np.ndarray,
    rng: np.random.Generator = None,
    noise_std: Tuple[float, float, float] = DEFAULT_NOISE_STD,
    warp_strength: float = 0.1,
    crop_ratio_range: Tuple[float, float] = (0.75, 1.0),
) -> np.ndarray:
    """Compose all three augmentations (crop -> warp -> noise) into a
    single augmented "view" of a window, used to generate positive pairs
    for the identity contrastive objective."""
    rng = rng or np.random.default_rng()
    crop_ratio = float(rng.uniform(*crop_ratio_range))
    view = random_crop(window, crop_ratio=crop_ratio, rng=rng)
    view = time_warp(view, warp_strength=warp_strength, rng=rng)
    view = add_sensor_noise(view, noise_std=noise_std, rng=rng)
    return view.astype(np.float32)
