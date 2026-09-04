"""Small geodesy helpers.

Everything in this project happens inside a few hundred km of a slick, so a local
equirectangular approximation is accurate to well under a percent and is orders of
magnitude faster than a full geodesic solve on the arrays we push through it.
"""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0088
KM_PER_DEG_LAT = 111.32
KN_TO_KMH = 1.852


def km_per_deg_lon(lat_deg: float | np.ndarray) -> np.ndarray:
    return KM_PER_DEG_LAT * np.cos(np.radians(lat_deg))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance, vectorised over any broadcastable shapes."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def local_km(lat, lon, lat0: float, lon0: float):
    """Project lat/lon to local east/north kilometres about (lat0, lon0)."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    east = (lon - lon0) * km_per_deg_lon(lat0)
    north = (lat - lat0) * KM_PER_DEG_LAT
    return east, north


def from_local_km(east, north, lat0: float, lon0: float):
    """Inverse of :func:`local_km`."""
    east, north = np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    lat = lat0 + north / KM_PER_DEG_LAT
    lon = lon0 + east / km_per_deg_lon(lat0)
    return lat, lon


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def destination(lat, lon, bearing_deg_, dist_km):
    """Point reached from (lat, lon) travelling `dist_km` along `bearing_deg_`."""
    d = np.asarray(dist_km, dtype=float) / R_EARTH_KM
    b = np.radians(np.asarray(bearing_deg_, dtype=float))
    p1, l1 = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b))
    l2 = l1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540) % 360 - 180


def angdiff_deg(a, b):
    """Signed smallest difference a - b, wrapped to [-180, 180)."""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180) % 360 - 180


def axis_alignment(course_deg, axis_deg):
    """Alignment of a *directed* course with an *undirected* axis, in [0, 1].

    A slick's long axis has no head or tail, so a vessel steaming 070 and one
    steaming 250 are equally consistent with an axis of 070. cos^2 is exactly the
    pi-periodic measure that wants: 1 when parallel (either way round), 0 when
    perpendicular, and a smooth ramp between.
    """
    return np.cos(np.radians(angdiff_deg(course_deg, axis_deg))) ** 2


def circular_variance(angles_deg) -> float:
    """1 - resultant length of unit vectors at `angles_deg`; 0 = tight, 1 = uniform."""
    a = np.radians(np.asarray(angles_deg, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(1.0 - np.hypot(np.cos(a).mean(), np.sin(a).mean()))
