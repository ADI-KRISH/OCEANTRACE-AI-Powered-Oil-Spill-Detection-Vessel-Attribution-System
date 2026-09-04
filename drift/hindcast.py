"""Backward drift: from an observed slick to an origin heatmap and time window.

The output is deliberately **not** a point. Backward advection reverses cleanly
but diffusion does not -- you cannot un-mix -- so a backward run legitimately
produces a spreading probability cloud, and that cloud grows the further back you
go. Reporting a single origin would be false precision, which is exactly what the
project spec forbids.

Two further things shape the design:

* **Age dominates the error.** To know how far back to integrate you need the
  slick's age, and age-from-area is the weakest number in the whole system
  (t ~ r^4). So we do not commit to one age: the ensemble sweeps a *range* of
  ages and accumulates over (lat, lon, t), which yields the time window as a
  distribution rather than an assumption.

* **A slick from a moving ship is a line, not a point.** Back-tracking it returns
  a *track segment*. That is better for attribution, not worse -- matching a
  vessel's AIS against a curve is far more discriminating than proximity to a
  point -- so the result carries `origin_track` alongside the heatmap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

KM_PER_DEG_LAT = 111.32


def km_per_deg_lon(lat):
    return KM_PER_DEG_LAT * np.cos(np.radians(lat))


def local_km(lat, lon, lat0, lon0):
    return ((np.asarray(lon, float) - lon0) * km_per_deg_lon(lat0),
            (np.asarray(lat, float) - lat0) * KM_PER_DEG_LAT)


def from_local_km(e, n, lat0, lon0):
    return (lat0 + np.asarray(n, float) / KM_PER_DEG_LAT,
            lon0 + np.asarray(e, float) / km_per_deg_lon(lat0))


#: Fraction of wind speed transferred to surface oil. 3% is the standard
#: operational value; it is also the single largest error term over long
#: hindcasts, since a 5 m/s wind error displaces the origin ~6 km in 12 h.
WIND_DRIFT_FACTOR = 0.03


@dataclass
class OriginEstimate:
    """What Module 2 hands Module 3."""

    #: Particle positions at the end of the backward run, one per ensemble member.
    lat: np.ndarray
    lon: np.ndarray
    #: Release time assigned to each particle (unix seconds).
    t_unix: np.ndarray
    #: Gridded probability density, plus its geographic extent.
    heatmap: np.ndarray
    heatmap_bounds: tuple          # (lat_min, lat_max, lon_min, lon_max)
    #: Most likely discharge path, as [(lat, lon, t_unix), ...].
    origin_track: list
    #: Time window as (earliest, most_likely, latest) unix seconds.
    time_window: tuple
    slick_bearing_deg: float | None = None
    forcing: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def centroid(self):
        return float(np.mean(self.lat)), float(np.mean(self.lon))

    def spread_km(self) -> float:
        """RMS distance of particles from their centroid -- the uncertainty radius."""
        cla, clo = self.centroid
        e, n = local_km(self.lat, self.lon, cla, clo)
        return float(np.sqrt(np.mean(e ** 2 + n ** 2)))

    def summary(self) -> str:
        cla, clo = self.centroid
        t0, tm, t1 = self.time_window
        import pandas as pd
        return (
            f"Origin estimate ({self.forcing.get('source', '?')} forcing)\n"
            f"  centroid      {cla:.4f}, {clo:.4f}\n"
            f"  uncertainty   {self.spread_km():.1f} km RMS "
            f"({len(self.lat)} ensemble members)\n"
            f"  window        {pd.Timestamp(t0, unit='s'):%Y-%m-%d %H:%M} to "
            f"{pd.Timestamp(t1, unit='s'):%H:%M} UTC\n"
            f"  most likely   {pd.Timestamp(tm, unit='s'):%Y-%m-%d %H:%M} UTC\n"
            f"  track         {len(self.origin_track)} points"
        )

    def to_dict(self) -> dict:
        return {
            "centroid": {"lat": self.centroid[0], "lon": self.centroid[1]},
            "spread_km": round(self.spread_km(), 2),
            "n_particles": int(len(self.lat)),
            "particles": [[round(float(a), 5), round(float(o), 5)]
                          for a, o in zip(self.lat, self.lon)],
            "heatmap": {
                "grid": [[round(float(v), 6) for v in row] for row in self.heatmap],
                "bounds": [round(float(b), 5) for b in self.heatmap_bounds],
            },
            "origin_track": [[round(float(a), 5), round(float(o), 5), float(t)]
                             for a, o, t in self.origin_track],
            "time_window": {
                "earliest": float(self.time_window[0]),
                "most_likely": float(self.time_window[1]),
                "latest": float(self.time_window[2]),
            },
            "slick_bearing_deg": self.slick_bearing_deg,
            "forcing": self.forcing,
            "meta": self.meta,
        }


def _rk2_step(e, n, t, forcing, dt_s, sign):
    """One midpoint (RK2) step. Cheap, and far less biased than Euler in an eddy."""
    u1, v1 = forcing.current(e, n, t)
    wu, wv = forcing.wind(t)
    u1 = u1 + WIND_DRIFT_FACTOR * wu
    v1 = v1 + WIND_DRIFT_FACTOR * wv

    half = 0.5 * sign * dt_s / 1000.0          # m/s * s -> km
    em, nm = e + u1 * half, n + v1 * half
    u2, v2 = forcing.current(em, nm, t + 0.5 * sign * dt_s)
    wu2, wv2 = forcing.wind(t + 0.5 * sign * dt_s)
    u2 = u2 + WIND_DRIFT_FACTOR * wu2
    v2 = v2 + WIND_DRIFT_FACTOR * wv2

    step = sign * dt_s / 1000.0
    return e + u2 * step, n + v2 * step


def backtrack(
    slick_lat, slick_lon, t_detect: float, ages_s,
    forcing=None, dt_s: float = 900.0,
    diffusion_m2_s: float = 1.0, seed: int = 0,
):
    """Integrate slick particles backwards over a *range* of ages.

    `ages_s` is an array, one entry per particle -- sweeping it is what turns the
    unknown age into a distribution instead of an assumption.

    Returns (lat, lon, t_release) arrays.
    """
    from .forcing import AnalyticForcing

    forcing = forcing or AnalyticForcing(seed=seed)
    rng = np.random.default_rng(seed + 17)

    slick_lat = np.atleast_1d(np.asarray(slick_lat, float))
    slick_lon = np.atleast_1d(np.asarray(slick_lon, float))
    ages_s = np.atleast_1d(np.asarray(ages_s, float))
    n_part = len(ages_s)

    # Seed particles by resampling the observed slick footprint.
    idx = rng.integers(0, len(slick_lat), n_part)
    lat0 = float(np.mean(slick_lat))
    lon0 = float(np.mean(slick_lon))
    e, n = local_km(slick_lat[idx], slick_lon[idx], lat0, lon0)

    # Horizontal eddy diffusivity -> random-walk step with variance 2*K*dt.
    sd_km = math.sqrt(2.0 * diffusion_m2_s * dt_s) / 1000.0

    t = np.full(n_part, float(t_detect))
    remaining = ages_s.copy()
    max_steps = int(np.ceil(ages_s.max() / dt_s))

    for _ in range(max_steps):
        active = remaining > 0
        if not active.any():
            break
        step = np.minimum(remaining[active], dt_s)
        ea, na = e[active], n[active]
        # Integrate the active subset one step backwards.
        ea, na = _rk2_step(ea, na, t[active], forcing, float(np.mean(step)), -1.0)
        ea = ea + rng.normal(0, sd_km, ea.shape)
        na = na + rng.normal(0, sd_km, na.shape)
        e[active], n[active] = ea, na
        t[active] -= step
        remaining[active] -= step

    lat, lon = from_local_km(e, n, lat0, lon0)
    return lat, lon, t_detect - ages_s


def _heatmap(lat, lon, bins: int = 64, pad_frac: float = 0.15):
    """2-D histogram of particle positions, normalised to a max of 1."""
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    dlat = max((lat_max - lat_min) * pad_frac, 0.01)
    dlon = max((lon_max - lon_min) * pad_frac, 0.01)
    bounds = (lat_min - dlat, lat_max + dlat, lon_min - dlon, lon_max + dlon)

    H, _, _ = np.histogram2d(
        lat, lon, bins=bins,
        range=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]])
    # Light smoothing so the density reads as a field rather than shot noise.
    try:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=1.2)
    except ImportError:
        pass
    if H.max() > 0:
        H = H / H.max()
    return H, bounds


def _origin_track(lat, lon, t_release, n_nodes: int = 12):
    """Most likely discharge path: the median position per release-time bin.

    A slick laid down by a moving vessel back-tracks to a curve. Binning by
    release time and taking the median (not the mean) keeps the path robust to
    the ensemble's outlying members.
    """
    order = np.argsort(t_release)
    lat, lon, t = lat[order], lon[order], t_release[order]
    if len(t) < n_nodes:
        return [(float(a), float(o), float(tt)) for a, o, tt in zip(lat, lon, t)]

    edges = np.linspace(t[0], t[-1], n_nodes + 1)
    out = []
    for i in range(n_nodes):
        m = (t >= edges[i]) & (t <= edges[i + 1])
        if m.sum() < 3:
            continue
        out.append((float(np.median(lat[m])), float(np.median(lon[m])),
                    float(0.5 * (edges[i] + edges[i + 1]))))
    return out


def hindcast_origin(
    slick_lat, slick_lon, t_detect: float,
    age_h: float = 6.0, age_range_h: tuple | None = None,
    n_particles: int = 600, forcing=None, seed: int = 0,
    diffusion_m2_s: float = 1.0, slick_bearing_deg: float | None = None,
) -> OriginEstimate:
    """Full backward ensemble -> origin heatmap, track and time window.

    `age_range_h` should come straight from the detection stage's age *interval*
    (which is wide on purpose). If omitted, a factor-of-three bracket is used --
    the honest default given how weakly age is constrained.
    """
    from .forcing import AnalyticForcing

    forcing = forcing or AnalyticForcing(seed=seed)
    rng = np.random.default_rng(seed)

    if age_range_h is None:
        age_range_h = (age_h / 3.0, age_h * 3.0)
    lo_h, hi_h = float(age_range_h[0]), float(age_range_h[1])
    lo_h = max(lo_h, 0.1)
    hi_h = max(hi_h, lo_h + 0.1)

    # Sweep ages log-uniformly: age uncertainty is multiplicative, not additive,
    # so a linear sweep would over-sample the long ages.
    ages_s = np.exp(rng.uniform(np.log(lo_h), np.log(hi_h), n_particles)) * 3600.0

    lat, lon, t_rel = backtrack(
        slick_lat, slick_lon, t_detect, ages_s,
        forcing=forcing, seed=seed, diffusion_m2_s=diffusion_m2_s)

    keep = np.isfinite(lat) & np.isfinite(lon)
    lat, lon, t_rel = lat[keep], lon[keep], t_rel[keep]

    H, bounds = _heatmap(lat, lon)
    track = _origin_track(lat, lon, t_rel)

    window = (float(t_detect - hi_h * 3600.0),
              float(t_detect - age_h * 3600.0),
              float(t_detect - lo_h * 3600.0))

    return OriginEstimate(
        lat=lat, lon=lon, t_unix=t_rel,
        heatmap=H, heatmap_bounds=bounds,
        origin_track=track, time_window=window,
        slick_bearing_deg=slick_bearing_deg,
        forcing=forcing.describe(),
        meta={"age_h": age_h, "age_range_h": [lo_h, hi_h],
              "n_particles": int(len(lat)),
              "diffusion_m2_s": diffusion_m2_s,
              "wind_drift_factor": WIND_DRIFT_FACTOR},
    )


def forecast(slick_lat, slick_lon, t_start: float, hours: float = 24.0,
             n_particles: int = 600, forcing=None, seed: int = 0,
             diffusion_m2_s: float = 1.0, n_frames: int = 12):
    """Forward drift -- where the slick goes next. Returns timestamped frames.

    Drives the timeline scrubber in the UI, and is the operationally useful half
    for response planning.
    """
    from .forcing import AnalyticForcing

    forcing = forcing or AnalyticForcing(seed=seed)
    rng = np.random.default_rng(seed + 3)

    slick_lat = np.atleast_1d(np.asarray(slick_lat, float))
    slick_lon = np.atleast_1d(np.asarray(slick_lon, float))
    idx = rng.integers(0, len(slick_lat), n_particles)
    lat0, lon0 = float(np.mean(slick_lat)), float(np.mean(slick_lon))
    e, n = local_km(slick_lat[idx], slick_lon[idx], lat0, lon0)

    dt_s = hours * 3600.0 / (n_frames * 4)
    sd_km = math.sqrt(2.0 * diffusion_m2_s * dt_s) / 1000.0
    t = float(t_start)
    frames = []
    for f in range(n_frames + 1):
        if f:
            for _ in range(4):
                e, n = _rk2_step(e, n, t, forcing, dt_s, +1.0)
                e = e + rng.normal(0, sd_km, e.shape)
                n = n + rng.normal(0, sd_km, n.shape)
                t += dt_s
        la, lo = from_local_km(e, n, lat0, lon0)
        frames.append({
            "t_unix": t,
            "hours_from_start": round((t - t_start) / 3600.0, 2),
            "particles": [[round(float(a), 5), round(float(o), 5)]
                          for a, o in zip(la, lo)],
        })
    return {"frames": frames, "forcing": forcing.describe(),
            "n_particles": n_particles}
