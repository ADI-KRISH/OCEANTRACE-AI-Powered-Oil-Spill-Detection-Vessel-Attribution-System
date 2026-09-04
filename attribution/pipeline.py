"""Module 3 entry point: drift origin + AIS in, ranked suspects out.

    from drift.hindcast import hindcast_origin
    from attribution.pipeline import attribute

    est = hindcast_origin(slick_lat, slick_lon, t_detect, age_h=6)
    result = attribute("ais.csv", origin=est)

Scoring policy, set deliberately: the **transparent weighted score is always the
primary answer**, and every suspect carries plain-language evidence. A learned
re-ranker may be enabled alongside it, and when it is, both orderings are
returned so a reviewer can see where the model disagrees with the explainable
score. The learned model never replaces the explanation, and never appears
without it -- the spec rules out an opaque end-to-end guilt classifier, and this
keeps the accuracy without giving up the audit trail.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ais import build_tracks, clean_tracks, load_ais
from .features import OriginHypothesis, build_feature_table, candidate_vessels
from .geo import haversine_km
from .scoring import BaselineScorer, LearnedScorer

DEFAULT_RADIUS_KM = 25.0
DEFAULT_LOOKBACK_H = 12.0


def origin_from_drift(est) -> OriginHypothesis:
    """Adapt a `drift.hindcast.OriginEstimate` to the attribution input.

    The drift ensemble's particles *are* the Monte-Carlo cloud this stage wants,
    so no resampling or Gaussian fitting is needed -- the real, asymmetric shape
    of the uncertainty is carried straight through.
    """
    return OriginHypothesis(
        lat=np.asarray(est.lat, float),
        lon=np.asarray(est.lon, float),
        t_unix=np.asarray(est.t_unix, float),
        slick_bearing_deg=est.slick_bearing_deg,
        meta={"forcing": est.forcing, "from_drift": True,
              "time_window": list(est.time_window)},
    )


def track_match_score(track, origin_track, scale_km: float = 10.0) -> float:
    """How well a vessel's path follows the back-tracked discharge path.

    This is the feature that the "a slick is a line, not a point" insight buys.
    For each node of the hindcast origin track -- a (lat, lon, time) triple --
    the vessel's own interpolated position at that time is compared. A vessel
    that merely crossed the area once scores poorly; one that ran *along* the
    discharge path, in step with it, scores high.

    `scale_km` is the distance at which the match decays to 1/e. It is set from
    the origin cloud's own spread by the caller rather than fixed: a hindcast with
    17 km of uncertainty cannot expect a vessel within 5 km, and a fixed short
    scale would drive every candidate to zero and make the feature useless.

    Returns a value in [0, 1].
    """
    if not origin_track or len(track) < 2:
        return 0.0
    lats, lons, ts = map(np.asarray, zip(*origin_track))
    vlat, vlon, covered = track.position_at(ts)
    d = haversine_km(vlat, vlon, lats, lons)
    d = np.where(covered & np.isfinite(d), d, np.inf)
    if not np.isfinite(d).any():
        return 0.0
    return float(np.mean(np.exp(-d / max(scale_km, 2.0))))


@dataclass
class AttributionResult:
    suspects: pd.DataFrame
    origin: OriginHypothesis
    tracks: dict = field(default_factory=dict, repr=False)
    scorer: str = "baseline"
    n_vessels_screened: int = 0
    n_candidates: int = 0
    learned_order: list | None = None

    @property
    def top(self):
        return None if self.suspects.empty else self.suspects.iloc[0]

    def explain(self, n: int = 5) -> str:
        if self.suspects.empty:
            return "No candidate vessels near the estimated origin."
        lines = [f"Screened {self.n_vessels_screened} vessels -> "
                 f"{self.n_candidates} candidates ({self.scorer} scorer)"]
        for _, r in self.suspects.head(n).iterrows():
            lines.append(f"\n#{int(r['rank'])}  MMSI {int(r.mmsi)}  "
                         f"{r['name'] or '(unnamed)'}   "
                         f"{r.attribution_pct:.1f}% attribution")
            for e in r.evidence:
                lines.append(f"      - {e}")
        return "\n".join(lines)

    def to_dict(self, top_n: int = 10) -> dict:
        rows = []
        for _, r in self.suspects.head(top_n).iterrows():
            rows.append({
                "rank": int(r["rank"]),
                "mmsi": int(r.mmsi),
                "name": r["name"] or "",
                "attribution_pct": round(float(r.attribution_pct), 2),
                "vessel_type": None if not np.isfinite(r.vtype) else int(r.vtype),
                "length_m": None if not np.isfinite(r.length) else round(float(r.length), 1),
                "min_dist_km": round(float(r.min_dist_km), 2),
                "time_gap_min": round(float(r.time_gap_min), 1),
                "track_match": round(float(r.get("track_match", 0.0)), 3),
                "evidence": list(r.evidence),
                "learned_rank": (int(r["learned_rank"])
                                 if "learned_rank" in r and pd.notna(r["learned_rank"])
                                 else None),
                "track": self._track_points(int(r.mmsi)),
            })
        return {
            "scorer": self.scorer,
            "n_vessels_screened": self.n_vessels_screened,
            "n_candidates": self.n_candidates,
            "suspects": rows,
        }

    def _track_points(self, mmsi: int, max_pts: int = 400):
        tr = self.tracks.get(mmsi)
        if tr is None:
            return []
        step = max(len(tr) // max_pts, 1)
        return [[round(float(a), 5), round(float(o), 5), float(t)]
                for a, o, t in zip(tr.lat[::step], tr.lon[::step], tr.t[::step])]


def attribute(
    ais_source,
    origin=None,
    lat=None, lon=None, time=None,
    *,
    slick_bearing_deg: float | None = None,
    radius_km: float = DEFAULT_RADIUS_KM,
    lookback_h: float = DEFAULT_LOOKBACK_H,
    use_learned: bool = False,
    model_path: str | None = None,
    origin_track: list | None = None,
) -> AttributionResult:
    """Rank vessels that could have produced the slick.

    `origin` accepts either a `drift.hindcast.OriginEstimate` or an
    `OriginHypothesis`; passing the drift estimate also supplies `origin_track`
    automatically, which enables the track-matching feature.
    """
    # --- normalise the origin -------------------------------------------
    if origin is not None and hasattr(origin, "origin_track"):
        if origin_track is None:
            origin_track = origin.origin_track
        origin = origin_from_drift(origin)
    if origin is None:
        if lat is None or lon is None or time is None:
            raise ValueError("supply `origin=` or lat/lon/time")
        origin = OriginHypothesis.from_point(
            float(lat), float(lon), pd.Timestamp(time).timestamp(),
            slick_bearing_deg=slick_bearing_deg)
    if slick_bearing_deg is not None:
        origin.slick_bearing_deg = slick_bearing_deg

    # --- query, clean, filter -------------------------------------------
    t0, t1 = origin.t_span
    t_range = (t0 - lookback_h * 3600.0, t1 + 2 * 3600.0)
    bbox = origin.bbox(pad_km=radius_km + origin.spread_km() + 40.0)

    raw = load_ais(ais_source, bbox=bbox, t_range=t_range)
    if raw.empty:
        return AttributionResult(
            suspects=pd.DataFrame(columns=["mmsi", "rank", "attribution_pct"]),
            origin=origin)

    clean = clean_tracks(raw)
    tracks = build_tracks(clean)
    cands = candidate_vessels(tracks, origin, radius_km=radius_km)
    feats = build_feature_table(tracks, origin, cands, radius_km=radius_km)

    # --- track matching, when the drift stage supplied a path ------------
    if origin_track and not feats.empty:
        # Tolerance follows the hindcast's own uncertainty, floored so a
        # suspiciously tight cloud cannot make the test unpassable.
        scale = max(origin.spread_km() * 0.6, 5.0)
        feats["track_match"] = [track_match_score(tracks[int(m)], origin_track,
                                                  scale_km=scale)
                                for m in feats.mmsi]
    elif not feats.empty:
        feats["track_match"] = 0.0

    # --- score: transparent first, always --------------------------------
    baseline = BaselineScorer()
    suspects = baseline.score(feats)
    scorer_name = "transparent"
    learned_order = None

    if use_learned and model_path and os.path.exists(model_path) and not feats.empty:
        try:
            learned = LearnedScorer(model_path=model_path)
            lr = learned.score(feats)[["mmsi", "rank"]].rename(
                columns={"rank": "learned_rank"})
            suspects = suspects.merge(lr, on="mmsi", how="left")
            learned_order = lr.sort_values("learned_rank").mmsi.astype(int).tolist()
            scorer_name = "transparent + learned re-rank"
        except Exception:
            # A missing or stale model must never take the whole ranking down --
            # the transparent score is the primary answer and stands on its own.
            pass

    return AttributionResult(
        suspects=suspects, origin=origin,
        tracks={m: tracks[m] for m in cands},
        scorer=scorer_name,
        n_vessels_screened=len(tracks), n_candidates=len(cands),
        learned_order=learned_order,
    )
