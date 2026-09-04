"""Tests for Module 2 (drift) and Module 3 (attribution).

    python -m pytest drift/tests attribution/tests -q

Focused on the properties that would silently produce a wrong origin rather than
crash: integration direction, the age sweep, the shape of the uncertainty, and
the guarantee that attribution never reports a bare score without evidence.
"""
from __future__ import annotations

import numpy as np
import pytest

from ..forcing import AnalyticForcing, CMEMSForcing, get_forcing
from ..hindcast import (WIND_DRIFT_FACTOR, backtrack, forecast, from_local_km,
                        hindcast_origin, local_km)


# ------------------------------------------------------------------ forcing --

def test_analytic_forcing_is_deterministic():
    a, b = AnalyticForcing(seed=3), AnalyticForcing(seed=3)
    assert a.current(0, 0, 1e9) == b.current(0, 0, 1e9)
    assert a.wind(1e9) == b.wind(1e9)


def test_different_seeds_give_different_fields():
    a, b = AnalyticForcing(seed=1), AnalyticForcing(seed=2)
    assert a.current(10, 10, 1e9) != b.current(10, 10, 1e9)


def test_analytic_forcing_declares_itself_unrealistic():
    """The demo must never imply an analytic field is a met-ocean model."""
    d = AnalyticForcing().describe()
    assert d["realistic"] is False
    assert "NOT a met-ocean model" in d["note"]


def test_current_magnitude_is_oceanographically_plausible():
    f = AnalyticForcing(seed=0)
    speeds = [float(np.hypot(*f.current(e, n, 1e9)))
              for e in range(-50, 51, 10) for n in range(-50, 51, 10)]
    assert max(speeds) < 1.5, "surface currents above 1.5 m/s are implausible here"
    assert np.mean(speeds) > 0.02


def test_get_forcing_falls_back_to_analytic_offline():
    f = get_forcing("analytic")
    assert isinstance(f, AnalyticForcing)
    # "auto" must never raise just because there are no credentials.
    assert get_forcing("auto") is not None


def test_cmems_login_check_does_not_raise():
    assert isinstance(CMEMSForcing.logged_in(), bool)


# --------------------------------------------------------------- projection --

def test_local_km_roundtrip():
    lat, lon = 18.6, 71.6
    e, n = local_km(18.8, 71.9, lat, lon)
    la, lo = from_local_km(e, n, lat, lon)
    assert la == pytest.approx(18.8, abs=1e-9)
    assert lo == pytest.approx(71.9, abs=1e-9)


# ---------------------------------------------------------------- backtrack --

def test_backtrack_moves_upstream_not_downstream():
    """The whole module is wrong if the sign of the integration is wrong."""
    f = AnalyticForcing(seed=0, bg_speed=0.5, eddy_speed=0.0, wind_speed=0.0)
    lat = np.full(50, 18.6)
    lon = np.full(50, 71.6)
    t_det = 1.7e9
    back_lat, back_lon, _ = backtrack(lat, lon, t_det, np.full(50, 6 * 3600.0),
                                      forcing=f, diffusion_m2_s=0.0)
    # Forward from the back-tracked point should return roughly to the start.
    fwd = forecast(back_lat, back_lon, t_det - 6 * 3600.0, hours=6.0,
                   forcing=f, diffusion_m2_s=0.0, n_frames=1, n_particles=50)
    end = np.array(fwd["frames"][-1]["particles"])
    assert abs(float(np.mean(end[:, 0])) - 18.6) < 0.05
    assert abs(float(np.mean(end[:, 1])) - 71.6) < 0.05


def test_backtrack_respects_per_particle_age():
    """Older particles must end up further from the slick than younger ones."""
    f = AnalyticForcing(seed=1, bg_speed=0.4, eddy_speed=0.0, wind_speed=0.0)
    lat = np.full(200, 18.6)
    lon = np.full(200, 71.6)
    ages = np.concatenate([np.full(100, 2 * 3600.0), np.full(100, 12 * 3600.0)])
    bl, bo, _ = backtrack(lat, lon, 1.7e9, ages, forcing=f, diffusion_m2_s=0.0)
    d_young = np.hypot(bl[:100] - 18.6, bo[:100] - 71.6).mean()
    d_old = np.hypot(bl[100:] - 18.6, bo[100:] - 71.6).mean()
    assert d_old > d_young * 2


def test_diffusion_spreads_particles():
    f = AnalyticForcing(seed=0, eddy_speed=0.0)
    args = dict(forcing=f)
    tight = backtrack(np.full(200, 18.6), np.full(200, 71.6), 1.7e9,
                      np.full(200, 6 * 3600.0), diffusion_m2_s=0.0, **args)
    loose = backtrack(np.full(200, 18.6), np.full(200, 71.6), 1.7e9,
                      np.full(200, 6 * 3600.0), diffusion_m2_s=20.0, **args)
    assert np.std(loose[0]) > np.std(tight[0])


# ------------------------------------------------------------ hindcast API --

def _slick():
    n = 60
    return (18.60 + np.linspace(0, 0.03, n), 71.60 + np.linspace(0, 0.05, n))


def test_hindcast_returns_a_cloud_not_a_point():
    la, lo = _slick()
    est = hindcast_origin(la, lo, 1.7e9, age_h=6, age_range_h=(2, 18),
                          n_particles=300, forcing=AnalyticForcing(seed=0))
    assert len(est.lat) > 100
    assert est.spread_km() > 0.5, "a backward run must carry real uncertainty"


def test_hindcast_time_window_brackets_the_estimate():
    la, lo = _slick()
    est = hindcast_origin(la, lo, 1.7e9, age_h=6, age_range_h=(2, 18),
                          forcing=AnalyticForcing(seed=0))
    t0, tm, t1 = est.time_window
    assert t0 < tm < t1 <= 1.7e9


def test_wider_age_range_gives_a_wider_cloud():
    """Age is the dominant error term -- the ensemble must actually show that."""
    la, lo = _slick()
    f = AnalyticForcing(seed=0)
    narrow = hindcast_origin(la, lo, 1.7e9, age_h=6, age_range_h=(5, 7),
                             n_particles=400, forcing=f).spread_km()
    wide = hindcast_origin(la, lo, 1.7e9, age_h=6, age_range_h=(1, 24),
                           n_particles=400, forcing=f).spread_km()
    assert wide > narrow


def test_hindcast_produces_a_track_not_just_a_blob():
    la, lo = _slick()
    est = hindcast_origin(la, lo, 1.7e9, age_h=6, age_range_h=(2, 18),
                          forcing=AnalyticForcing(seed=0))
    assert len(est.origin_track) >= 3
    for node in est.origin_track:
        assert len(node) == 3          # (lat, lon, t)
    times = [n[2] for n in est.origin_track]
    assert times == sorted(times)


def test_heatmap_is_normalised_and_covers_the_particles():
    la, lo = _slick()
    est = hindcast_origin(la, lo, 1.7e9, age_h=6, forcing=AnalyticForcing(seed=0))
    assert est.heatmap.max() == pytest.approx(1.0)
    lo_lat, hi_lat, lo_lon, hi_lon = est.heatmap_bounds
    assert lo_lat <= est.lat.min() and est.lat.max() <= hi_lat
    assert lo_lon <= est.lon.min() and est.lon.max() <= hi_lon


def test_estimate_serialises_for_the_api():
    la, lo = _slick()
    d = hindcast_origin(la, lo, 1.7e9, age_h=6,
                        forcing=AnalyticForcing(seed=0)).to_dict()
    for k in ("centroid", "spread_km", "heatmap", "origin_track",
              "time_window", "forcing"):
        assert k in d
    assert d["forcing"]["realistic"] is False


def test_wind_drift_factor_is_the_operational_value():
    assert WIND_DRIFT_FACTOR == pytest.approx(0.03)


def test_forecast_particles_move_over_time():
    la, lo = _slick()
    fc = forecast(la, lo, 1.7e9, hours=12, n_frames=6,
                  forcing=AnalyticForcing(seed=0, bg_speed=0.4))
    first = np.array(fc["frames"][0]["particles"])
    last = np.array(fc["frames"][-1]["particles"])
    assert np.abs(last.mean(0) - first.mean(0)).max() > 1e-4
    assert fc["frames"][-1]["hours_from_start"] == pytest.approx(12.0, abs=0.1)
