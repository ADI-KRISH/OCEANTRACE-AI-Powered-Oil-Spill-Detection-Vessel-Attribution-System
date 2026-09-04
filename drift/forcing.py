"""Met-ocean forcing for the drift model.

Two sources, one interface:

* :class:`CMEMSForcing` -- real Copernicus Marine currents (and wind/waves where
  the product carries them), for when `copernicusmarine login` has been run.
* :class:`AnalyticForcing` -- a deterministic synthetic field, so the module runs
  offline with no account, no download and no network.

The analytic field is **not** a met-ocean model and is never presented as one. It
exists so the hindcast machinery, the ensemble and the whole UI can be built,
tested and demoed without a 2 GB download blocking the path. Every result carries
`forcing: "analytic"` so a viewer can tell which produced it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "forcing")


@dataclass
class AnalyticForcing:
    """Divergence-free analytic surface current plus a slowly veering wind.

    Built as the curl of a stream function so the flow does not artificially
    converge -- particles spread by diffusion, not by a numerical artefact of the
    field. Deterministic in `seed`, so a hindcast is reproducible.
    """

    seed: int = 0
    #: Background flow speed, m/s. Typical shelf currents are 0.1-0.5 m/s.
    bg_speed: float = 0.25
    #: Mesoscale eddy amplitude, m/s.
    eddy_speed: float = 0.15
    #: Eddy length scales, km.
    scale_km: tuple = (45.0, 18.0)
    wind_speed: float = 6.0          # m/s
    name: str = "analytic"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self._bg_dir = rng.uniform(0, 2 * np.pi)
        self._phase = rng.uniform(0, 2 * np.pi)
        self._wind_dir = rng.uniform(0, 2 * np.pi)
        self._veer = rng.uniform(-1, 1) * 2 * np.pi / (72 * 3600)

    def current(self, east_km, north_km, t_s):
        """Surface current (u, v) in m/s at local km coordinates and unix time."""
        e = np.asarray(east_km, dtype=float)
        n = np.asarray(north_km, dtype=float)
        L1, L2 = self.scale_km

        u = self.bg_speed * np.cos(self._bg_dir)
        v = self.bg_speed * np.sin(self._bg_dir)
        u = u - self.eddy_speed * np.cos(e / L1) * np.cos(n / L1) \
              - 0.5 * self.eddy_speed * np.cos(e / L2 + self._phase) * np.cos(n / L2)
        v = v - self.eddy_speed * np.sin(e / L1) * np.sin(n / L1) \
              - 0.5 * self.eddy_speed * np.sin(e / L2 + self._phase) * np.sin(n / L2)

        # M2 semi-diurnal tide.
        tide = 0.08 * np.sin(2 * np.pi * np.asarray(t_s) / (12.42 * 3600) + self._phase)
        return u + tide, v

    def wind(self, t_s):
        """Wind (u, v) in m/s -- constant speed, slowly veering."""
        d = self._wind_dir + self._veer * np.asarray(t_s)
        return self.wind_speed * np.cos(d), self.wind_speed * np.sin(d)

    def describe(self) -> dict:
        return {
            "source": "analytic",
            "realistic": False,
            "note": ("Deterministic synthetic field, NOT a met-ocean model. "
                     "Origin estimates from it demonstrate the method, not real "
                     "ocean transport. Use CMEMS forcing for real results."),
            "bg_speed_ms": self.bg_speed,
            "wind_speed_ms": self.wind_speed,
        }


class CMEMSForcing:
    """Copernicus Marine currents, cached to `data/forcing/`.

    Requires `copernicusmarine login` to have been run once. Kept deliberately
    thin: it downloads a lat/lon/time subset to netCDF and hands the path to
    OpenDrift's generic CF reader, which is the supported path rather than
    reimplementing interpolation.
    """

    #: Global analysis/forecast product: hourly surface currents at 1/12 degree.
    DATASET = "cmems_mod_glo_phy-cur_anfc_0.083deg_PT1H-i"
    VARIABLES = ["uo", "vo"]
    name = "cmems"

    def __init__(self, cache_dir: str = DATA_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def logged_in() -> bool:
        """True if a Copernicus Marine credentials file is present."""
        for p in (os.path.expanduser("~/.copernicusmarine/.copernicusmarine-credentials"),
                  os.path.expanduser("~/.copernicusmarine-credentials")):
            if os.path.exists(p):
                return True
        return bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"))

    def fetch(self, lat_min, lat_max, lon_min, lon_max, t_start, t_end,
              pad_deg: float = 1.0) -> str:
        """Download a subset and return the netCDF path (cached by extent)."""
        import copernicusmarine
        import pandas as pd

        key = (f"cur_{lat_min:.1f}_{lat_max:.1f}_{lon_min:.1f}_{lon_max:.1f}"
               f"_{pd.Timestamp(t_start, unit='s'):%Y%m%d}"
               f"_{pd.Timestamp(t_end, unit='s'):%Y%m%d}.nc")
        out = os.path.join(self.cache_dir, key)
        if os.path.exists(out):
            return out

        copernicusmarine.subset(
            dataset_id=self.DATASET,
            variables=self.VARIABLES,
            minimum_latitude=lat_min - pad_deg, maximum_latitude=lat_max + pad_deg,
            minimum_longitude=lon_min - pad_deg, maximum_longitude=lon_max + pad_deg,
            start_datetime=pd.Timestamp(t_start, unit="s").isoformat(),
            end_datetime=pd.Timestamp(t_end, unit="s").isoformat(),
            minimum_depth=0, maximum_depth=1,
            output_filename=os.path.basename(out),
            output_directory=self.cache_dir,
        )
        return out

    def describe(self) -> dict:
        return {"source": "cmems", "realistic": True,
                "dataset": self.DATASET,
                "note": "Copernicus Marine global analysis, hourly surface currents."}


def get_forcing(prefer: str = "auto", seed: int = 0):
    """Pick a forcing source.

    ``"auto"`` uses CMEMS when credentials exist and falls back to analytic,
    saying so, rather than failing -- an offline demo must never hard-error.
    """
    if prefer == "analytic":
        return AnalyticForcing(seed=seed)
    if prefer == "cmems":
        return CMEMSForcing()
    if CMEMSForcing.logged_in():
        return CMEMSForcing()
    return AnalyticForcing(seed=seed)
