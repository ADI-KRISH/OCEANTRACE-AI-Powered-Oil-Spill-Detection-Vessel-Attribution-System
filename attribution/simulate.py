"""Synthetic scenario generator -- the project's training-label factory.

There is no public "slick polygon -> guilty MMSI" dataset, so labels are
manufactured: take an AIS day, elect one moving vessel the culprit, extrude a
slick along the stretch of track it covered during a discharge window, drift that
slick forward to the moment a satellite would have seen it, then hand the
attribution stage a *deliberately imperfect* backward origin estimate.

Ground truth is known by construction, so Top-1 / Recall@3 / MRR are measurable.

`synth_ais_day` exists so the whole pipeline runs with no data files present. The
moment a real NOAA day is on disk, pass it to :func:`make_scenario` instead --
nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._sim_drift import advect, hindcast_origin
from .features import OriginHypothesis
from .geo import bearing_deg, destination, haversine_km

#: AIS vessel-type codes drawn for synthetic traffic, with realistic-ish mix.
_VTYPES = [80, 80, 80, 81, 70, 70, 71, 79, 30, 30, 31, 52, 60, 36, 37]
_VTYPE_W = [.10, .06, .05, .04, .12, .08, .06, .05, .12, .08, .05, .06, .06, .04, .03]


def synth_ais_day(
    n_vessels: int = 220,
    lat0: float = 39.5,
    lon0: float = -73.0,
    extent_km: float = 55.0,
    day_start: float = 1665446400.0,   # 2022-10-11T00:00:00Z
    hours: float = 24.0,
    seed: int = 0,
    n_lanes: int = 2,
    lane_frac: float = 0.45,
) -> pd.DataFrame:
    """Generate a plausible day of AIS traffic in NOAA MarineCadastre schema.

    Vessels come in three behavioural flavours, because a realistic candidate
    pool is what makes the attribution problem hard: transiting ships on long
    straight legs, fishing vessels working a small box at low speed with erratic
    heading, and anchored/drifting craft. A minority of tracks contain AIS gaps.

    Crucially, a `lane_frac` share of transiting vessels are placed in one of
    `n_lanes` shipping corridors, travelling near-parallel courses. Without lanes
    the culprit is trivially the only ship nearby; with them, a slick sits under a
    stream of large vessels on the same heading -- which is the real problem.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Lane geometry: an origin offset and a heading each corridor runs along.
    lanes = [{
        "offset_km": rng.uniform(-extent_km * 0.5, extent_km * 0.5),
        "course": rng.uniform(0, 360),
        "width_km": rng.uniform(2.0, 6.0),
    } for _ in range(n_lanes)]

    for i in range(n_vessels):
        mmsi = 200_000_000 + seed * 100_000 + i
        vtype = int(rng.choice(_VTYPES, p=np.array(_VTYPE_W) / np.sum(_VTYPE_W)))
        if 70 <= vtype <= 89:
            length = float(rng.uniform(90, 330))
            mode = "transit"
        elif vtype == 30:
            length = float(rng.uniform(15, 45))
            mode = "fishing" if rng.random() < 0.75 else "transit"
        else:
            length = float(rng.uniform(20, 120))
            mode = rng.choice(["transit", "fishing", "moored"], p=[.6, .2, .2])

        # Report interval: class A underway is ~10 s at speed but daily exports
        # are thinned; 2-6 min is representative of what MarineCadastre ships.
        dt = float(rng.uniform(120, 360))
        n_pts = int(hours * 3600 / dt)
        t = day_start + np.arange(n_pts) * dt + rng.uniform(0, dt)

        from .geo import from_local_km

        in_lane = mode == "transit" and rng.random() < lane_frac
        if in_lane:
            lane = lanes[int(rng.integers(0, len(lanes)))]
            # Sit across the corridor and anywhere along it.
            along = rng.uniform(-extent_km, extent_km)
            across = lane["offset_km"] + rng.normal(0, lane["width_km"])
            th = np.radians(90.0 - lane["course"])
            start_e = along * np.cos(th) - across * np.sin(th)
            start_n = along * np.sin(th) + across * np.cos(th)
        else:
            start_e = rng.uniform(-extent_km, extent_km)
            start_n = rng.uniform(-extent_km, extent_km)

        slat, slon = from_local_km(start_e, start_n, lat0, lon0)
        slat, slon = float(slat), float(slon)

        if mode == "transit":
            sog = float(rng.uniform(8, 18))
            if in_lane:
                # Same corridor, either direction, with modest steering noise.
                course = (lane["course"] + (0 if rng.random() < 0.5 else 180)
                          + rng.normal(0, 6)) % 360
            else:
                course = float(rng.uniform(0, 360))
            # Occasional course change, as at a waypoint.
            courses = np.full(n_pts, course)
            if not in_lane and rng.random() < 0.5:
                k = rng.integers(n_pts // 4, 3 * n_pts // 4)
                courses[k:] = (course + rng.uniform(-70, 70)) % 360
            sogs = np.clip(sog + rng.normal(0, 0.4, n_pts), 0, 25)
        elif mode == "fishing":
            sog = float(rng.uniform(1.5, 5.0))
            courses = (rng.uniform(0, 360) + np.cumsum(rng.normal(0, 22, n_pts))) % 360
            sogs = np.clip(sog + rng.normal(0, 1.0, n_pts), 0, 8)
        else:  # moored / drifting
            courses = (rng.uniform(0, 360) + np.cumsum(rng.normal(0, 6, n_pts))) % 360
            sogs = np.clip(rng.normal(0.2, 0.2, n_pts), 0, 1.5)

        lat, lon = _integrate_track(slat, slon, sogs, courses, dt)

        keep = np.ones(n_pts, dtype=bool)
        if rng.random() < 0.30:                       # AIS gap
            g_len = int(rng.uniform(20, 120) * 60 / dt)
            g0 = int(rng.integers(0, max(n_pts - g_len - 1, 1)))
            keep[g0:g0 + g_len] = False

        rows.append(pd.DataFrame({
            "MMSI": mmsi,
            "BaseDateTime": pd.to_datetime(t[keep], unit="s", utc=True),
            "LAT": lat[keep], "LON": lon[keep],
            "SOG": sogs[keep], "COG": courses[keep],
            "Heading": courses[keep],
            "VesselName": f"SIM_{mmsi % 100000:05d}",
            "VesselType": vtype, "Length": length,
            "Width": max(length / 7.0, 4.0), "Draft": max(length / 18.0, 1.5),
        }))

    df = pd.concat(rows, ignore_index=True)
    df["BaseDateTime"] = df.BaseDateTime.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df


def _integrate_track(lat0, lon0, sogs, courses, dt_s):
    """Dead-reckon a track from speeds and courses."""
    n = len(sogs)
    lat = np.empty(n)
    lon = np.empty(n)
    lat[0], lon[0] = lat0, lon0
    step_km = sogs * 1.852 * (dt_s / 3600.0)
    for i in range(1, n):
        la, lo = destination(lat[i - 1], lon[i - 1], courses[i - 1], step_km[i - 1])
        lat[i], lon[i] = float(la), float(lo)
    return lat, lon


@dataclass
class Scenario:
    """One synthetic spill with known ground truth."""

    origin: OriginHypothesis
    culprit_mmsi: int
    ais: pd.DataFrame
    slick_lat: np.ndarray
    slick_lon: np.ndarray
    t_detect: float
    true_lat: float
    true_lon: float
    true_t: float
    age_s: float


def make_scenario(
    ais_df: pd.DataFrame | None = None,
    seed: int = 0,
    n_vessels: int = 90,
    discharge_hours: float = 2.0,
    age_hours: float | None = None,
    n_origin_samples: int = 200,
    min_culprit_speed_kn: float = 2.0,
) -> Scenario | None:
    """Build one labelled scenario. Returns None if no usable culprit exists.

    The culprit must be genuinely under way -- attributing a slick to a vessel
    that never moved would be a degenerate, unrealistically easy problem.
    """
    rng = np.random.default_rng(seed)
    if ais_df is None:
        ais_df = synth_ais_day(n_vessels=n_vessels, seed=seed)

    from .ais import build_tracks, clean_tracks, load_ais
    clean = clean_tracks(load_ais(ais_df))
    tracks = build_tracks(clean)
    if not tracks:
        return None

    # --- elect a culprit: a vessel moving through the middle of the day -----
    movers = []
    for mmsi, tr in tracks.items():
        if len(tr) < 20:
            continue
        sog = tr.sog[np.isfinite(tr.sog)]
        if sog.size and np.median(sog) >= min_culprit_speed_kn:
            movers.append(mmsi)
    if not movers:
        return None
    culprit = int(rng.choice(movers))
    ctr = tracks[culprit]

    # --- discharge window over which oil is released along the track -------
    dur = discharge_hours * 3600.0
    t_lo, t_hi = ctr.span
    if t_hi - t_lo < dur + 6 * 3600:
        return None
    t_start = float(rng.uniform(t_lo + 3600, t_hi - dur - 4 * 3600))
    t_end = t_start + dur

    # --- extrude the slick along the culprit's track -----------------------
    n_parcels = 220
    t_release = np.linspace(t_start, t_end, n_parcels)
    rla, rlo, _ = ctr.position_at(t_release)
    # Oil spreads across-track as well as along it.
    jitter_km = rng.normal(0, 0.25, n_parcels)
    brg = bearing_deg(rla[0], rlo[0], rla[-1], rlo[-1]) + 90.0
    rla, rlo = destination(rla, rlo, brg, jitter_km)

    # Slick long-axis bearing, as stage A would measure it from the polygon,
    # with the measurement error a real SAR centreline fit would carry.
    axis = float(bearing_deg(rla[0], rlo[0], rla[-1], rlo[-1]))
    axis = (axis + float(rng.normal(0, 8))) % 360.0

    # --- drift forward to detection ----------------------------------------
    if age_hours is None:
        age_hours = float(rng.uniform(3.0, 10.0))
    age_s = age_hours * 3600.0
    t_detect = t_end + age_s
    field_seed = int(seed)
    sla, slo = advect(rla, rlo, t_end, age_s, seed=seed + 5,
                      field_seed=field_seed, diffusion_km2_per_h=0.8)

    # --- hindcast an imperfect origin (this is what stage C receives) ------
    ola, olo, ot = hindcast_origin(
        sla, slo, t_detect, age_s, n_samples=n_origin_samples,
        seed=seed, field_seed=field_seed,
    )

    true_mid_t = 0.5 * (t_start + t_end)
    tla, tlo, _ = ctr.position_at(true_mid_t)

    origin = OriginHypothesis(
        ola, olo, ot, slick_bearing_deg=axis,
        meta={"scenario_seed": seed, "age_hours": age_hours},
    )
    return Scenario(
        origin=origin, culprit_mmsi=culprit, ais=clean,
        slick_lat=sla, slick_lon=slo, t_detect=t_detect,
        true_lat=float(tla[0]), true_lon=float(tlo[0]), true_t=true_mid_t,
        age_s=age_s,
    )


def origin_error_km(sc: Scenario) -> float:
    """How far the hindcast cloud's centroid sits from the true discharge point.

    Reported in evaluation so the attribution metrics can be read against the
    quality of the origin they were handed.
    """
    cla, clo = sc.origin.centroid
    return float(haversine_km(cla, clo, sc.true_lat, sc.true_lon))
