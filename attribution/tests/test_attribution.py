"""Tests for Module 3 (AIS vessel attribution).

    python -m pytest attribution/tests -q

The emphasis is on the properties that make the output defensible rather than
merely present: evidence always accompanies a score, the transparent scorer is
never silently replaced, and the ranking beats chance end to end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.forcing import AnalyticForcing
from drift.hindcast import hindcast_origin

from ..ais import build_tracks, clean_tracks, load_ais
from ..features import (FEATURE_NAMES, OriginHypothesis, build_feature_table,
                        candidate_vessels)
from ..pipeline import attribute, origin_from_drift, track_match_score
from ..scoring import BaselineScorer
from ..simulate import make_scenario, synth_ais_day


# ------------------------------------------------------------------- basics --

def test_sixteen_features_are_defined():
    # 14 original + platform_score (infrastructure discriminator) +
    # track_match (drift-coupled; defaults to 0.0 until a hindcast supplies a
    # backward-track, then both scorers actually weight it -- see scoring.py).
    assert len(FEATURE_NAMES) == 16


def test_synthetic_ais_is_placed_where_asked():
    """Global demos depend on AIS being generated at the scene, not at a default."""
    df = load_ais(synth_ais_day(n_vessels=12, lat0=-33.9, lon0=18.4, seed=0))
    assert abs(df.lat.mean() - (-33.9)) < 1.5
    assert abs(df.lon.mean() - 18.4) < 1.5


def test_candidate_filter_drops_distant_traffic():
    o = OriginHypothesis.from_point(18.6, 71.6, 1.7e9, n=50, sigma_km=1)
    far = synth_ais_day(n_vessels=8, lat0=0.0, lon0=0.0, seed=1)
    tracks = build_tracks(clean_tracks(load_ais(far)))
    assert candidate_vessels(tracks, o, radius_km=25) == []


def test_features_never_nan():
    sc = make_scenario(seed=3, n_vessels=120)
    tracks = build_tracks(sc.ais)
    cands = candidate_vessels(tracks, sc.origin)
    ft = build_feature_table(tracks, sc.origin, cands)
    assert ft[FEATURE_NAMES].notna().all().all()


# ------------------------------------------------------- drift integration --

def test_origin_from_drift_carries_the_ensemble_through():
    """The drift particles ARE the MC cloud -- no resampling, no Gaussian fit."""
    la = 18.6 + np.linspace(0, 0.03, 40)
    lo = 71.6 + np.linspace(0, 0.05, 40)
    est = hindcast_origin(la, lo, 1.7e9, age_h=6, age_range_h=(2, 12),
                          n_particles=250, forcing=AnalyticForcing(seed=0),
                          slick_bearing_deg=55)
    o = origin_from_drift(est)
    assert len(o) == len(est.lat)
    assert o.slick_bearing_deg == 55
    assert np.allclose(o.lat, est.lat)


def test_track_match_rewards_following_the_discharge_path():
    """A vessel on the path must outscore one parked far away."""
    t0 = 1.7e9
    times = np.linspace(t0, t0 + 3600, 8)
    path = [(18.60 + 0.01 * i, 71.60 + 0.01 * i, float(t))
            for i, t in enumerate(times)]

    on = pd.DataFrame({
        "MMSI": 1, "BaseDateTime": pd.to_datetime(times, unit="s", utc=True),
        "LAT": [p[0] for p in path], "LON": [p[1] for p in path], "SOG": 6.0})
    off = on.copy()
    off["MMSI"] = 2
    off["LAT"] = 19.9
    off["LON"] = 72.9

    tr = build_tracks(clean_tracks(load_ais(pd.concat([on, off]))))
    assert track_match_score(tr[1], path, 10.0) > 0.8
    assert track_match_score(tr[2], path, 10.0) < 0.05


def test_track_match_handles_an_empty_path():
    tracks = build_tracks(clean_tracks(load_ais(synth_ais_day(n_vessels=3, seed=0))))
    any_track = next(iter(tracks.values()))
    assert track_match_score(any_track, []) == 0.0


# ---------------------------------------------------------------- scoring --

def test_every_suspect_carries_evidence():
    """A bare score is not actionable -- the spec requires a stated reason."""
    sc = make_scenario(seed=5, n_vessels=150)
    r = attribute(sc.ais, origin=sc.origin)
    assert not r.suspects.empty
    assert all(len(e) > 0 for e in r.suspects.evidence)


def test_attribution_percentages_sum_to_100_and_rank_in_order():
    sc = make_scenario(seed=5, n_vessels=150)
    r = attribute(sc.ais, origin=sc.origin)
    assert r.suspects.attribution_pct.sum() == pytest.approx(100.0)
    assert r.suspects.score.is_monotonic_decreasing
    assert list(r.suspects["rank"]) == list(range(1, len(r.suspects) + 1))


def test_transparent_scorer_is_the_default():
    """The learned model must never be applied unless explicitly requested."""
    sc = make_scenario(seed=5, n_vessels=120)
    r = attribute(sc.ais, origin=sc.origin)
    assert r.scorer == "transparent"
    assert r.learned_order is None


def test_missing_learned_model_falls_back_without_failing():
    sc = make_scenario(seed=5, n_vessels=120)
    r = attribute(sc.ais, origin=sc.origin, use_learned=True,
                  model_path="does/not/exist.txt")
    assert r.scorer == "transparent"
    assert not r.suspects.empty


def test_perpendicular_course_scores_below_aligned():
    base = BaselineScorer()
    row = {n: 0.0 for n in FEATURE_NAMES}
    row.update(min_dist_km=1.0, mean_dist_km=1.0, prox_score=0.8, vtype=80.0,
               length=200.0, mmsi=1, name="x", n_fixes=100, time_gap_min=1.0)
    along = base.score(pd.DataFrame([{**row, "course_align": 1.0}]))
    across = base.score(pd.DataFrame([{**row, "course_align": 0.0}]))
    assert along.score.iloc[0] > across.score.iloc[0]


# --------------------------------------------------------------- end to end --

def test_full_chain_beats_chance():
    """Slick -> real drift ensemble -> attribution, over several scenarios."""
    ranks, chance = [], []
    for seed in range(6):
        sc = make_scenario(seed=seed, n_vessels=180)
        if sc is None:
            continue
        age_h = sc.age_s / 3600.0
        est = hindcast_origin(
            sc.slick_lat, sc.slick_lon, sc.t_detect, age_h=age_h,
            age_range_h=(age_h / 2.5, age_h * 2.5),
            forcing=AnalyticForcing(seed=seed),
            slick_bearing_deg=sc.origin.slick_bearing_deg)
        r = attribute(sc.ais, origin=est)
        hit = r.suspects[r.suspects.mmsi == sc.culprit_mmsi]
        if hit.empty:
            continue
        ranks.append(int(hit["rank"].iloc[0]))
        chance.append((len(r.suspects) + 1) / 2.0)

    assert len(ranks) >= 4
    assert np.mean(ranks) < np.mean(chance), (
        f"mean rank {np.mean(ranks):.1f} is no better than chance "
        f"{np.mean(chance):.1f}")


def test_result_serialises_with_tracks_for_the_map():
    sc = make_scenario(seed=3, n_vessels=150)
    d = attribute(sc.ais, origin=sc.origin).to_dict(top_n=3)
    assert d["scorer"] and d["suspects"]
    first = d["suspects"][0]
    for k in ("rank", "mmsi", "attribution_pct", "evidence", "track"):
        assert k in first
    assert len(first["track"]) > 1
