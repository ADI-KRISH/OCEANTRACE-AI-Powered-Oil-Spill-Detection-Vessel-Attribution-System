"""Toy advection-diffusion drift -- used ONLY to manufacture training scenarios.

This is not the drift model. The real one is the `drift` package (OpenDrift-style
ensemble hindcast with CMEMS or analytic forcing). This cheap analytic version
exists solely so the scenario simulator can generate labelled training data fast:
it needs a forward map (discharge -> observed slick) and a deliberately *biased*
backward map, thousands of times, which is wasteful to ask of the real model.

This is explicitly a *placeholder* for stage B. It is not a met-ocean model and
must not be presented as one: the current field is an analytic stream function,
not HYCOM/CMEMS, and the wind is a slowly-rotating constant. What it does provide
is the property the attribution stage actually needs from B during development --
a forward map from discharge to observed slick, and a backward map that returns a
*biased, spread-out* origin cloud rather than the truth. Attribution trained
against a perfect origin would learn nothing useful.

Swapping in real currents means replacing :func:`current_field` alone.
"""
from __future__ import annotations

import numpy as np

from .geo import KM_PER_DEG_LAT, from_local_km, km_per_deg_lon, local_km


def current_field(east_km, north_km, t_s, seed: int = 0):
    """Surface current (u, v) in km/h at local coordinates and time.

    A steady background flow plus two counter-rotating mesoscale eddies and a
    weak tidal oscillation. Divergence-free by construction (it is the curl of a
    stream function), which keeps patches from artificially piling up.
    """
    rng = np.random.default_rng(seed)
    bg_dir = rng.uniform(0, 2 * np.pi)
    bg_speed = rng.uniform(0.6, 1.8)            # km/h
    u0, v0 = bg_speed * np.cos(bg_dir), bg_speed * np.sin(bg_dir)

    L1, L2 = rng.uniform(30, 60), rng.uniform(15, 30)   # eddy scales, km
    a1, a2 = rng.uniform(0.4, 1.2), rng.uniform(0.2, 0.6)
    phase = rng.uniform(0, 2 * np.pi)

    e = np.asarray(east_km, dtype=float)
    n = np.asarray(north_km, dtype=float)
    tide = 0.35 * np.sin(2 * np.pi * np.asarray(t_s) / (12.42 * 3600) + phase)

    # psi = a1*L1*cos(e/L1)*sin(n/L1) + ... ; (u, v) = (-dpsi/dn, dpsi/de)
    u = (-a1 * np.cos(e / L1) * np.cos(n / L1)
         - a2 * np.cos(e / L2 + phase) * np.cos(n / L2))
    v = (-a1 * np.sin(e / L1) * np.sin(n / L1)
         - a2 * np.sin(e / L2 + phase) * np.sin(n / L2))
    return u0 + u + tide, v0 + v


def wind_field(t_s, seed: int = 0):
    """Wind (u, v) in km/h -- slowly veering, as a synoptic system passes."""
    rng = np.random.default_rng(seed + 991)
    speed = rng.uniform(8, 30)
    d0 = rng.uniform(0, 2 * np.pi)
    veer = rng.uniform(-1, 1) * 2 * np.pi / (48 * 3600)
    d = d0 + veer * np.asarray(t_s)
    return speed * np.cos(d), speed * np.sin(d)


#: Fraction of wind speed transferred to a surface slick. 3% is the standard
#: operational value used in oil-spill response models.
WIND_DRIFT_FACTOR = 0.03


def advect(
    lat, lon, t_start: float, duration_s: float, dt_s: float = 900.0,
    diffusion_km2_per_h: float = 0.5, seed: int = 0, backward: bool = False,
    field_seed: int | None = None,
):
    """Advect particles through the current + wind field with random-walk spread.

    Set `backward=True` to integrate in reverse, which is how an origin is
    hindcast from an observed slick.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).copy()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).copy()
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    fseed = seed if field_seed is None else field_seed

    e, n = local_km(lat, lon, lat0, lon0)
    rng = np.random.default_rng(seed + 7)
    sign = -1.0 if backward else 1.0
    n_steps = max(int(abs(duration_s) / dt_s), 1)
    h = dt_s / 3600.0

    t = t_start
    for _ in range(n_steps):
        cu, cv = current_field(e, n, t, seed=fseed)
        wu, wv = wind_field(t, seed=fseed)
        u = cu + WIND_DRIFT_FACTOR * wu
        v = cv + WIND_DRIFT_FACTOR * wv
        # Random walk with variance 2*K*dt gives the intended eddy diffusivity.
        sd = np.sqrt(2.0 * diffusion_km2_per_h * h)
        e = e + sign * u * h + rng.normal(0, sd, e.shape)
        n = n + sign * v * h + rng.normal(0, sd, n.shape)
        t += sign * dt_s

    return from_local_km(e, n, lat0, lon0)


def hindcast_origin(
    slick_lat, slick_lon, t_detect: float, age_s: float,
    n_samples: int = 200, seed: int = 0, field_seed: int | None = None,
    model_error: float = 0.35, age_error_frac: float = 0.25,
):
    """Back-track an observed slick to a cloud of possible discharge points.

    Two error sources are injected on purpose, because a real hindcast has both:

    * `model_error` -- the backward integration uses a *different* field seed
      from the forward one, standing in for the gap between the drift model's
      currents and the ocean's.
    * `age_error_frac` -- slick age from SAR is itself an estimate, so each
      sample is back-tracked for a slightly different duration.

    Returns (lat, lon, t_unix) arrays suitable for :class:`OriginHypothesis`.
    """
    rng = np.random.default_rng(seed + 31)
    slick_lat = np.atleast_1d(np.asarray(slick_lat, dtype=float))
    slick_lon = np.atleast_1d(np.asarray(slick_lon, dtype=float))

    # Seed the cloud by resampling the observed slick footprint.
    idx = rng.integers(0, len(slick_lat), n_samples)
    la, lo = slick_lat[idx], slick_lon[idx]

    # A model whose currents are perfect would make attribution trivially easy;
    # perturbing the field seed is what makes the learned ranker earn its keep.
    if field_seed is None:
        field_seed = seed
    back_seed = field_seed + (1000 if rng.random() < model_error else 0)

    ages = age_s * (1.0 + rng.normal(0, age_error_frac, n_samples))
    ages = np.clip(ages, 0.1 * age_s, 2.0 * age_s)

    out_la = np.empty(n_samples)
    out_lo = np.empty(n_samples)
    # Group samples by similar age so the integration stays vectorised.
    order = np.argsort(ages)
    for chunk in np.array_split(order, 12):
        if chunk.size == 0:
            continue
        dur = float(np.mean(ages[chunk]))
        bla, blo = advect(la[chunk], lo[chunk], t_detect, dur, backward=True,
                          seed=seed + int(chunk[0]), field_seed=back_seed,
                          diffusion_km2_per_h=1.5)
        out_la[chunk], out_lo[chunk] = bla, blo

    t_origin = t_detect - ages
    return out_la, out_lo, t_origin
