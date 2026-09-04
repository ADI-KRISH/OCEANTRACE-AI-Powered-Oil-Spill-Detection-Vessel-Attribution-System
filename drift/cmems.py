"""Real Copernicus Marine surface currents, wrapped to the forcing interface.

Downloads a lat/lon/time subset once, caches it as netCDF, and interpolates it on
demand so :mod:`drift.hindcast` can use it exactly like the analytic field.

Two products are needed because they cover different eras:

* ``anfc`` (analysis / forecast) — recent days and a short forecast. Used for
  live scenes.
* ``my`` (multi-year reanalysis) — the historical archive. Used for real past
  incidents such as the 2021 Monterey Bay case, which the forecast product does
  not reach back to.

The right one is chosen from the requested date rather than configured, because
getting it wrong is a silent failure: the download simply returns nothing for the
period and the drift falls back with no obvious symptom.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

#: Surface currents, 6-hourly instantaneous, global 1/12 degree. Recent + forecast.
DATASET_FORECAST = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"
#: Multi-year reanalysis, daily mean, global 1/12 degree. Historical.
DATASET_REANALYSIS = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

#: The reanalysis lags real time; anything older than this uses it.
REANALYSIS_CUTOFF_DAYS = 120

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "forcing")

KM_PER_DEG_LAT = 111.32


def pick_dataset(t_unix: float) -> str:
    """Forecast product for recent dates, reanalysis for older ones."""
    age_days = (datetime.now(timezone.utc).timestamp() - t_unix) / 86400.0
    return DATASET_REANALYSIS if age_days > REANALYSIS_CUTOFF_DAYS else DATASET_FORECAST


class CMEMSCurrents:
    """Real surface currents for one region and time window.

    Implements the same ``current`` / ``wind`` / ``describe`` interface as
    :class:`drift.forcing.AnalyticForcing`, so the hindcast is unchanged.
    """

    name = "cmems"

    def __init__(self, lat0: float, lon0: float, t_center: float,
                 half_width_deg: float = 2.0, days_back: float = 3.0,
                 days_forward: float = 1.0, cache_dir: str = DATA_DIR,
                 wind_speed_ms: float = 0.0, wind_dir_deg: float = 0.0):
        self.lat0, self.lon0, self.t_center = float(lat0), float(lon0), float(t_center)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Windage is NOT included in the ocean model. CMEMS surface currents carry
        # wind-driven Ekman transport, but not the extra ~3% the slick itself
        # picks up from direct wind stress. Left at zero unless the caller
        # supplies a wind, rather than inventing one.
        self.wind_speed_ms = wind_speed_ms
        self.wind_dir_deg = wind_dir_deg

        self.dataset_id = pick_dataset(t_center)
        self._path = self._fetch(half_width_deg, days_back, days_forward)
        self._load()

    # -- download ---------------------------------------------------------
    def _fetch(self, half_deg, days_back, days_forward) -> str:
        import copernicusmarine
        import pandas as pd

        t0 = self.t_center - days_back * 86400.0
        t1 = self.t_center + days_forward * 86400.0
        key = (f"cur_{self.dataset_id[-14:]}_{self.lat0:.1f}_{self.lon0:.1f}"
               f"_{pd.Timestamp(t0, unit='s'):%Y%m%d}"
               f"_{pd.Timestamp(t1, unit='s'):%Y%m%d}.nc")
        out = os.path.join(self.cache_dir, key)
        if os.path.exists(out):
            return out

        copernicusmarine.subset(
            dataset_id=self.dataset_id,
            variables=["uo", "vo"],
            minimum_latitude=self.lat0 - half_deg,
            maximum_latitude=self.lat0 + half_deg,
            minimum_longitude=self.lon0 - half_deg,
            maximum_longitude=self.lon0 + half_deg,
            start_datetime=pd.Timestamp(t0, unit="s").isoformat(),
            end_datetime=pd.Timestamp(t1, unit="s").isoformat(),
            minimum_depth=0, maximum_depth=1,
            output_filename=os.path.basename(out),
            output_directory=self.cache_dir,
            overwrite=True,
        )
        if not os.path.exists(out):
            raise FileNotFoundError(f"CMEMS subset produced no file at {out}")
        return out

    # -- interpolation ----------------------------------------------------
    def _load(self):
        import xarray as xr

        ds = xr.open_dataset(self._path)
        if "depth" in ds.dims:
            ds = ds.isel(depth=0)
        self._ds = ds
        self._lats = ds["latitude"].values
        self._lons = ds["longitude"].values
        self._times = ds["time"].values.astype("datetime64[s]").astype(np.int64)
        self._u = np.asarray(ds["uo"].values, dtype=np.float32)
        self._v = np.asarray(ds["vo"].values, dtype=np.float32)
        # Land and missing cells arrive as NaN; treated as still water so a
        # particle drifting over one stops rather than jumping to garbage.
        self._u = np.nan_to_num(self._u, nan=0.0)
        self._v = np.nan_to_num(self._v, nan=0.0)

    def _km_per_deg_lon(self):
        return KM_PER_DEG_LAT * np.cos(np.radians(self.lat0))

    def current(self, east_km, north_km, t_s):
        """Trilinear-ish lookup: nearest time slice, bilinear in space (m/s)."""
        e = np.atleast_1d(np.asarray(east_km, dtype=float))
        n = np.atleast_1d(np.asarray(north_km, dtype=float))

        lat = self.lat0 + n / KM_PER_DEG_LAT
        lon = self.lon0 + e / self._km_per_deg_lon()

        ti = int(np.argmin(np.abs(self._times - float(np.mean(np.atleast_1d(t_s))))))
        u_slice, v_slice = self._u[ti], self._v[ti]

        # Fractional grid indices, clamped to the subset's extent.
        yi = np.interp(lat, self._lats, np.arange(len(self._lats)))
        xi = np.interp(lon, self._lons, np.arange(len(self._lons)))
        y0 = np.clip(np.floor(yi).astype(int), 0, len(self._lats) - 2)
        x0 = np.clip(np.floor(xi).astype(int), 0, len(self._lons) - 2)
        fy, fx = yi - y0, xi - x0

        def bilin(a):
            return ((1 - fy) * ((1 - fx) * a[y0, x0] + fx * a[y0, x0 + 1])
                    + fy * ((1 - fx) * a[y0 + 1, x0] + fx * a[y0 + 1, x0 + 1]))

        return bilin(u_slice), bilin(v_slice)

    def wind(self, t_s):
        """Constant wind, default zero -- see the note in ``__init__``."""
        d = np.radians(self.wind_dir_deg)
        s = self.wind_speed_ms
        shape = np.shape(np.atleast_1d(t_s))
        return np.full(shape, s * np.cos(d)), np.full(shape, s * np.sin(d))

    def describe(self) -> dict:
        import pandas as pd
        return {
            "source": "cmems",
            "realistic": True,
            "dataset": self.dataset_id,
            "kind": ("reanalysis" if self.dataset_id == DATASET_REANALYSIS
                     else "analysis/forecast"),
            "time_coverage": [
                str(pd.Timestamp(int(self._times[0]), unit="s")),
                str(pd.Timestamp(int(self._times[-1]), unit="s")),
            ],
            "n_time_steps": int(len(self._times)),
            "grid": f"{len(self._lats)}x{len(self._lons)} @ 1/12 deg",
            "windage_included": self.wind_speed_ms > 0,
            "note": ("Copernicus Marine surface currents. Windage is not applied "
                     "unless a wind is supplied: the model's surface current "
                     "already carries Ekman transport, but not the extra ~3% a "
                     "slick picks up from direct wind stress."),
        }
