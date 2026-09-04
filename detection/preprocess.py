"""Sentinel-1 SAR preprocessing.

The dataset scenes from Krestenitis et al. arrive already terrain-corrected and
calibrated (the ESA SNAP steps), so this module covers what is still needed at
training and inference time: dB conversion, speckle suppression and
normalisation.

Speckle matters more here than in optical segmentation. SAR speckle is
multiplicative, not additive, so ordinary Gaussian blurring both fails to remove
it and destroys the slick edges that characterisation later measures. A Lee
filter estimates local statistics and only smooths where the signal is
homogeneous, which preserves the boundary between a slick and open sea.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def to_db(amplitude: np.ndarray, floor_db: float = -50.0) -> np.ndarray:
    """Linear backscatter amplitude -> decibels.

    Oil suppresses capillary waves, so a slick reads as a *dark* patch. Working
    in dB makes that contrast roughly additive and stabilises the dynamic range
    across scenes taken at different incidence angles.
    """
    a = np.asarray(amplitude, dtype=np.float32)
    db = 10.0 * np.log10(np.maximum(a, EPS))
    return np.maximum(db, floor_db)


def lee_filter(img: np.ndarray, size: int = 5, cu: float = 0.523) -> np.ndarray:
    """Lee speckle filter.

    `cu` is the expected coefficient of variation of pure speckle; 0.523 is the
    standard value for single-look SAR intensity. Where local variation is at or
    below that level the pixel is treated as homogeneous and smoothed toward the
    local mean; where it exceeds it (an edge, a ship) the original value is kept.
    """
    from scipy.ndimage import uniform_filter

    img = np.asarray(img, dtype=np.float32)
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img ** 2, size)
    var = np.maximum(sq_mean - mean ** 2, 0.0)

    ci2 = var / np.maximum(mean ** 2, EPS)
    cu2 = cu ** 2
    # Weight -> 0 in flat regions (take the mean), -> 1 at edges (keep the pixel).
    w = np.clip((ci2 - cu2) / np.maximum(ci2, EPS), 0.0, 1.0)
    return mean + w * (img - mean)


def normalise(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0):
    """Percentile-stretch to [0, 1]; returns (image, (low, high)).

    Percentiles rather than min/max because a single bright scatterer -- a metal
    hull, a platform -- would otherwise compress the entire sea surface into a
    couple of grey levels.
    """
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi - lo < EPS:
        return np.zeros_like(img), (float(lo), float(hi))
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0), (float(lo), float(hi))


def preprocess_scene(
    img: np.ndarray,
    already_db: bool = True,
    despeckle: bool = True,
    filter_size: int = 5,
) -> np.ndarray:
    """Full chain: (dB) -> despeckle -> normalise. Returns float32 in [0, 1].

    `already_db` defaults True because the published dataset ships 8-bit
    greyscale renderings rather than raw linear backscatter.
    """
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3:
        x = x.mean(axis=2)
    if not already_db:
        x = to_db(x)
    if despeckle:
        x = lee_filter(x, size=filter_size)
    x, _ = normalise(x)
    return x.astype(np.float32)
