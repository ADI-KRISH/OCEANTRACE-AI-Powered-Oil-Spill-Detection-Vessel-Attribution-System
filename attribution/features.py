"""The B-to-C interface (:class:`OriginHypothesis`) and the feature extractor.

`OriginHypothesis` is the *only* object the drift team has to produce, and the
only thing this module needs from them: a Monte-Carlo cloud of possible discharge
points in space and time, plus (optionally) the slick's long-axis bearing from the
detection team. Representing the origin as samples rather than a single point is
deliberate -- a backward drift solve is genuinely uncertain, and collapsing that
cloud to its mean would throw away the shape of the uncertainty that separates a
close-but-brief pass from a distant-but-persistent one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ais import GAP_S, Track
from .geo import axis_alignment, circular_variance, haversine_km

#: The 14 features, in a fixed order. Models are trained and served against this
#: list, so append -- never reorder or insert.
FEATURE_NAMES = [
    # proximity
    "prox_score",
    "min_dist_km",
    "mean_dist_km",
    "dwell_frac",
    # temporal
    "time_gap_min",
    # behavioural anomaly
    "slow_steaming",
    "loiter_score",
    "ais_gap_max_min",
    "gap_over_origin",
    "dark_frac",
    # trajectory shape
    "course_align",
    "cpa_sog_kn",
    # priors
    "vtype_prior",
    "size_score",
]

#: Proximity e-folding length (km). A vessel 5 km from the origin cloud scores
#: ~0.37 on `prox_score`; one at 20 km scores ~0.02.
PROX_SCALE_KM = 5.0


@dataclass
class OriginHypothesis:
    """Monte-Carlo estimate of where and when oil entered the water.

    Parameters
    ----------
    lat, lon, t_unix:
        Equal-length arrays of MC samples. Scalars are accepted and broadcast.
    slick_bearing_deg:
        Long-axis azimuth of the detected slick, degrees clockwise from north.
        Undirected -- 070 and 250 mean the same axis. `None` if unavailable, in
        which case `course_align` is returned as neutral.
    """

    lat: np.ndarray
    lon: np.ndarray
    t_unix: np.ndarray
    slick_bearing_deg: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lat = np.atleast_1d(np.asarray(self.lat, dtype=float))
        self.lon = np.atleast_1d(np.asarray(self.lon, dtype=float))
        self.t_unix = np.atleast_1d(np.asarray(self.t_unix, dtype=float))
        n = max(len(self.lat), len(self.lon), len(self.t_unix))
        if len(self.lat) == 1:
            self.lat = np.repeat(self.lat, n)
        if len(self.lon) == 1:
            self.lon = np.repeat(self.lon, n)
        if len(self.t_unix) == 1:
            self.t_unix = np.repeat(self.t_unix, n)
        if not (len(self.lat) == len(self.lon) == len(self.t_unix)):
            raise ValueError("lat/lon/t_unix must be the same length (or scalar)")

    def __len__(self) -> int:
        return len(self.lat)

    @property
    def centroid(self) -> tuple[float, float]:
        return float(np.mean(self.lat)), float(np.mean(self.lon))

    @property
    def t_center(self) -> float:
        return float(np.median(self.t_unix))

    @property
    def t_span(self) -> tuple[float, float]:
        return float(np.min(self.t_unix)), float(np.max(self.t_unix))

    def bbox(self, pad_km: float = 0.0) -> tuple[float, float, float, float]:
        """(lat_min, lat_max, lon_min, lon_max) around the cloud, padded."""
        from .geo import KM_PER_DEG_LAT, km_per_deg_lon

        dlat = pad_km / KM_PER_DEG_LAT
        dlon = pad_km / max(km_per_deg_lon(float(np.mean(self.lat))), 1e-6)
        return (float(self.lat.min()) - dlat, float(self.lat.max()) + dlat,
                float(self.lon.min()) - dlon, float(self.lon.max()) + dlon)

    def spread_km(self) -> float:
        """RMS distance of the samples from their own centroid."""
        cla, clo = self.centroid
        return float(np.sqrt(np.mean(haversine_km(self.lat, self.lon, cla, clo) ** 2)))

    @classmethod
    def from_point(
        cls,
        lat: float,
        lon: float,
        t_unix: float,
        n: int = 200,
        sigma_km: float = 5.0,
        sigma_min: float = 60.0,
        seed: int | None = 0,
        slick_bearing_deg: float | None = None,
    ) -> "OriginHypothesis":
        """Build a cloud around a single best-guess origin.

        Used when a caller has only a point estimate -- the pipeline's convenience
        entry point, and the fallback when no drift model is wired up yet.
        """
        from .geo import from_local_km

        rng = np.random.default_rng(seed)
        east = rng.normal(0, sigma_km, n)
        north = rng.normal(0, sigma_km, n)
        la, lo = from_local_km(east, north, lat, lon)
        ts = t_unix + rng.normal(0, sigma_min * 60.0, n)
        return cls(la, lo, ts, slick_bearing_deg=slick_bearing_deg,
                   meta={"synthetic_cloud": True})


# ---------------------------------------------------------------------------
# Vessel-type priors
# ---------------------------------------------------------------------------

def vtype_prior(vtype: float) -> float:
    """Prior likelihood that a vessel of this AIS type is an oil discharger.

    Grounded in what actually gets prosecuted: operational discharge (bilge,
    slops, tank washings) is overwhelmingly a tanker and cargo problem. These are
    deliberately soft -- they nudge ties, they should not convict.
    """
    if vtype is None or not np.isfinite(vtype):
        return 0.40
    v = int(vtype)
    if 80 <= v <= 89:      # tanker
        return 1.00
    if 70 <= v <= 79:      # cargo
        return 0.85
    if v == 30:            # fishing
        return 0.35
    if 31 <= v <= 32:      # towing
        return 0.60
    if 50 <= v <= 59:      # tug / pilot / special craft
        return 0.45
    if 60 <= v <= 69:      # passenger
        return 0.40
    if 40 <= v <= 49:      # high-speed craft
        return 0.25
    if v in (36, 37):      # sailing / pleasure
        return 0.12
    return 0.40


def size_score(length_m: float) -> float:
    """Normalised vessel size in [0, 1]; a proxy for slop-tank capacity."""
    if length_m is None or not np.isfinite(length_m) or length_m <= 0:
        return 0.35
    return float(np.clip(np.log10(length_m / 10.0) / np.log10(40.0), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def candidate_vessels(
    tracks: dict[int, Track],
    origin: OriginHypothesis,
    radius_km: float = 25.0,
    min_fixes: int = 2,
) -> list[int]:
    """Keep vessels with at least one *raw* fix near the origin cloud in time.

    This is the "filter irrelevant traffic" step. The test is on real observations
    rather than interpolated ones, so a vessel is never made a suspect purely by
    an interpolation drawn across a gap it was nowhere near.
    """
    cla, clo = origin.centroid
    t0, t1 = origin.t_span
    # Allow the cloud's own spatial spread on top of the search radius.
    reach = radius_km + origin.spread_km()
    keep = []
    for mmsi, tr in tracks.items():
        if len(tr) < min_fixes:
            continue
        in_win = (tr.t >= t0 - 6 * 3600) & (tr.t <= t1 + 6 * 3600)
        if not in_win.any():
            continue
        d = haversine_km(tr.lat[in_win], tr.lon[in_win], cla, clo)
        if np.nanmin(d) <= reach:
            keep.append(mmsi)
    return keep


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    track: Track,
    origin: OriginHypothesis,
    radius_km: float = 25.0,
    window_pad_s: float = 3 * 3600.0,
) -> dict[str, float]:
    """Compute the 14 attribution features for one vessel against one origin."""
    t0, t1 = origin.t_span
    w0, w1 = t0 - window_pad_s, t1 + window_pad_s
    in_win = (track.t >= w0) & (track.t <= w1)
    n_win = int(in_win.sum())

    # --- proximity: vessel position at each MC sample time ------------------
    la, lo, covered = track.position_at(origin.t_unix)
    d = haversine_km(la, lo, origin.lat, origin.lon)
    # Samples the track cannot speak to are penalised, not silently dropped --
    # otherwise a vessel observed for two minutes could win on one lucky sample.
    d = np.where(covered & np.isfinite(d), d, np.inf)

    finite = np.isfinite(d)
    if finite.any():
        min_dist = float(np.min(d[finite]))
        mean_dist = float(np.mean(np.minimum(d[finite], 200.0)))
    else:
        min_dist, mean_dist = 500.0, 500.0
    prox = float(np.mean(np.exp(-np.where(finite, d, 1e6) / PROX_SCALE_KM)))

    # --- dwell: share of real fixes loitering inside the search radius ------
    cla, clo = origin.centroid
    if n_win:
        d_fix = haversine_km(track.lat[in_win], track.lon[in_win], cla, clo)
        dwell = float(np.mean(d_fix <= radius_km))
    else:
        dwell = 0.0

    # --- temporal: how close a real observation sits to the origin time -----
    time_gap_min = float(np.min(track.nearest_fix_dt(origin.t_unix)) / 60.0)

    # --- behavioural anomaly ------------------------------------------------
    sog_win = track.sog[in_win] if n_win else np.array([])
    sog_win = sog_win[np.isfinite(sog_win)]
    # Deliberate slow-steaming is the classic signature: fast enough to make way
    # and stretch the slick, slow enough to discharge without atomising it.
    slow = float(np.mean((sog_win >= 2.0) & (sog_win <= 9.0))) if sog_win.size else 0.0

    cog_win = track.cog[in_win] if n_win else np.array([])
    slow_mask = np.isfinite(sog_win) & (sog_win < 5.0)
    if n_win and slow_mask.any() and len(cog_win) == len(sog_win):
        loiter = circular_variance(cog_win[slow_mask]) * float(np.mean(slow_mask))
    else:
        loiter = 0.0

    gaps = track.gaps()
    win_gaps = [(a, b) for a, b in gaps if b > w0 and a < w1]
    gap_max_min = max((b - a) / 60.0 for a, b in win_gaps) if win_gaps else 0.0

    # A gap that covers the origin in time *and* whose interpolated path passes
    # near it in space -- a vessel that went dark exactly over the spill.
    in_gap = track.in_gap(origin.t_unix)
    near = np.isfinite(d) & (d <= radius_km)
    gap_over_origin = float(np.mean(in_gap & near))

    # Fraction of the origin's own time span spent dark.
    span = max(t1 - t0, 1.0)
    dark = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in gaps) / span
    dark_frac = float(np.clip(dark, 0.0, 1.0))

    # --- trajectory shape ---------------------------------------------------
    if finite.any():
        i_cpa = int(np.argmin(np.where(finite, d, np.inf)))
        t_cpa = float(origin.t_unix[i_cpa])
        cog_cpa = float(track.value_at(t_cpa, "cog")[0])
        sog_cpa = float(track.value_at(t_cpa, "sog")[0])
    else:
        cog_cpa, sog_cpa = np.nan, np.nan

    if origin.slick_bearing_deg is None or not np.isfinite(cog_cpa):
        course_align = 0.5  # neutral: no axis known, or no course reported
    else:
        course_align = float(axis_alignment(cog_cpa, origin.slick_bearing_deg))

    return {
        "prox_score": prox,
        "min_dist_km": min(min_dist, 500.0),
        "mean_dist_km": min(mean_dist, 500.0),
        "dwell_frac": dwell,
        "time_gap_min": min(time_gap_min, 24 * 60.0),
        "slow_steaming": slow,
        "loiter_score": float(loiter),
        "ais_gap_max_min": min(gap_max_min, 24 * 60.0),
        "gap_over_origin": gap_over_origin,
        "dark_frac": dark_frac,
        "course_align": course_align,
        "cpa_sog_kn": float(sog_cpa) if np.isfinite(sog_cpa) else 6.0,
        "vtype_prior": vtype_prior(track.vtype),
        "size_score": size_score(track.length),
    }


def build_feature_table(
    tracks: dict[int, Track],
    origin: OriginHypothesis,
    candidates: list[int] | None = None,
    radius_km: float = 25.0,
) -> pd.DataFrame:
    """Feature rows for every candidate vessel, indexed by MMSI."""
    if candidates is None:
        candidates = candidate_vessels(tracks, origin, radius_km=radius_km)

    rows = []
    for mmsi in candidates:
        tr = tracks[mmsi]
        f = extract_features(tr, origin, radius_km=radius_km)
        f["mmsi"] = mmsi
        f["name"] = tr.name
        f["vtype"] = tr.vtype
        f["length"] = tr.length
        f["n_fixes"] = len(tr)
        rows.append(f)

    cols = ["mmsi", "name", "vtype", "length", "n_fixes"] + FEATURE_NAMES
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]
