"""AIS ingest, cleaning and trajectory reconstruction.

Handles the two schemas the project actually sees -- NOAA MarineCadastre daily
exports (MMSI, BaseDateTime, LAT, LON, SOG, ...) and the lower-cased Kaggle
variant -- and normalises both to one internal frame:

    mmsi, t, lat, lon, sog, cog, heading, name, vtype, length, width, draft

``t`` is unix seconds (UTC). Everything downstream assumes that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import haversine_km

# Canonical column -> the aliases seen in the wild, lower-cased for matching.
_ALIASES = {
    "mmsi": ["mmsi"],
    # "t" first so that re-normalising an already-canonical frame is a no-op.
    "t": ["t", "basedatetime", "timestamp", "time", "datetime", "datetimeutc"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "long", "longitude", "x"],
    "sog": ["sog", "speed", "speedoverground"],
    "cog": ["cog", "course", "courseoverground"],
    "heading": ["heading", "trueheading"],
    "name": ["vesselname", "name", "shipname"],
    "vtype": ["vtype", "vesseltype", "shiptype", "type"],
    "length": ["length", "loa"],
    "width": ["width", "beam"],
    "draft": ["draft", "draught"],
}

#: Implied speed above which a fix is treated as a position error, not motion.
TELEPORT_KN = 60.0

#: An AIS silence longer than this counts as a "dark" interval.
GAP_S = 900.0


def _to_unix_seconds(ts: pd.Series) -> np.ndarray:
    """tz-aware datetime Series -> float unix seconds.

    Not ``ts.astype("int64") / 1e9``: pandas >= 2.2 stores datetimes at
    whatever resolution it parsed (often ``datetime64[us]`` since pandas 3.0),
    and ``astype("int64")`` returns a count in *that* unit, not nanoseconds --
    dividing by 1e9 then silently understates every timestamp by 1000x (a
    "one day" AIS file collapses to about 90 seconds of track). Normalising to
    ``datetime64[ns]`` first fixes the unit regardless of what pandas chose.
    """
    return ts.dt.tz_convert(None).to_numpy("datetime64[ns]").astype("int64") / 1e9


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    lower = {str(c).lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    out = {}
    for canon, aliases in _ALIASES.items():
        for a in aliases:
            if a in lower:
                out[canon] = lower[a]
                break
    return out


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map an arbitrary AIS frame onto the canonical schema."""
    cols = _resolve_columns(df)
    missing = {"mmsi", "t", "lat", "lon"} - set(cols)
    if missing:
        raise ValueError(
            f"AIS frame is missing required column(s) {sorted(missing)}; "
            f"saw columns {list(df.columns)[:20]}"
        )

    out = pd.DataFrame(index=df.index)
    out["mmsi"] = pd.to_numeric(df[cols["mmsi"]], errors="coerce")

    raw_t = df[cols["t"]]
    if pd.api.types.is_datetime64_any_dtype(raw_t):
        # Already datetimes -- localise naive values to UTC rather than assume.
        ts = raw_t if raw_t.dt.tz is not None else raw_t.dt.tz_localize("UTC")
        out["t"] = _to_unix_seconds(ts)
    elif pd.api.types.is_numeric_dtype(raw_t):
        # Already epoch -- guess the unit from magnitude (ms exports are common).
        vals = pd.to_numeric(raw_t, errors="coerce").to_numpy(dtype=float)
        out["t"] = np.where(vals > 1e11, vals / 1000.0, vals)
    else:
        ts = pd.to_datetime(raw_t, errors="coerce", utc=True, format="mixed")
        out["t"] = _to_unix_seconds(ts)

    for c in ("lat", "lon", "sog", "cog", "heading", "length", "width", "draft"):
        out[c] = pd.to_numeric(df[cols[c]], errors="coerce") if c in cols else np.nan
    out["name"] = df[cols["name"]].astype(str) if "name" in cols else ""
    out["vtype"] = pd.to_numeric(df[cols["vtype"]], errors="coerce") if "vtype" in cols else np.nan

    out = out.dropna(subset=["mmsi", "t", "lat", "lon"])
    out = out[out.lat.between(-90, 90) & out.lon.between(-180, 180)]
    # 511 is the AIS "heading unavailable" sentinel; SOG 102.3 means "not available".
    out.loc[out.heading >= 511, "heading"] = np.nan
    out.loc[out.sog > 102, "sog"] = np.nan
    out.loc[out.cog > 360, "cog"] = np.nan
    out["mmsi"] = out.mmsi.astype("int64")
    return out.reset_index(drop=True)


def _apply_filters(df, bbox, t_range):
    if bbox is not None:
        la0, la1, lo0, lo1 = bbox
        df = df[df.lat.between(la0, la1) & df.lon.between(lo0, lo1)]
    if t_range is not None:
        df = df[df.t.between(t_range[0], t_range[1])]
    return df


def load_ais(
    source,
    bbox: tuple[float, float, float, float] | None = None,
    t_range: tuple[float, float] | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Load AIS from a CSV path or DataFrame, filtered to `bbox` / `t_range`.

    A NOAA daily file runs to millions of rows, so CSV paths are read in chunks
    and filtered before anything is concatenated -- the spatiotemporal query
    happens *during* ingest rather than after it.

    `bbox` is (lat_min, lat_max, lon_min, lon_max); `t_range` is unix seconds.
    """
    if isinstance(source, pd.DataFrame):
        return _apply_filters(normalise(source), bbox, t_range).reset_index(drop=True)

    parts = []
    for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
        part = _apply_filters(normalise(chunk), bbox, t_range)
        if len(part):
            parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=["mmsi", "t", "lat", "lon", "sog", "cog", "heading",
                     "length", "width", "draft", "name", "vtype"]
        )
    return pd.concat(parts, ignore_index=True)


def clean_tracks(df: pd.DataFrame, teleport_kn: float = TELEPORT_KN) -> pd.DataFrame:
    """Per-MMSI de-duplication and outlier removal.

    Drops repeated timestamps and fixes whose implied speed from the previous
    *accepted* fix exceeds `teleport_kn`. The rejection is sequential rather than
    vectorised because discarding a bad fix changes the baseline for the next one
    -- vectorising it would let one wild position condemn its innocent neighbour.
    """
    if df.empty:
        return df
    df = df.sort_values(["mmsi", "t"]).drop_duplicates(["mmsi", "t"], keep="first")

    keep = np.ones(len(df), dtype=bool)
    mmsi = df.mmsi.to_numpy()
    lat, lon, t = df.lat.to_numpy(), df.lon.to_numpy(), df.t.to_numpy()

    starts = np.flatnonzero(np.r_[True, mmsi[1:] != mmsi[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        anchor = s
        for i in range(s + 1, e):
            dt_h = (t[i] - t[anchor]) / 3600.0
            if dt_h <= 0:
                keep[i] = False
                continue
            d_nm = haversine_km(lat[anchor], lon[anchor], lat[i], lon[i]) / 1.852
            if d_nm / dt_h > teleport_kn:
                keep[i] = False
            else:
                anchor = i
    return df[keep].reset_index(drop=True)


@dataclass
class Track:
    """One vessel's reconstructed trajectory over the query window."""

    mmsi: int
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    sog: np.ndarray
    cog: np.ndarray
    name: str = ""
    vtype: float = np.nan
    length: float = np.nan
    width: float = np.nan
    draft: float = np.nan

    def __len__(self) -> int:
        return len(self.t)

    @property
    def span(self) -> tuple[float, float]:
        return float(self.t[0]), float(self.t[-1])

    def gaps(self, min_gap_s: float = GAP_S) -> list[tuple[float, float]]:
        """Intervals with no AIS fix longer than `min_gap_s` ("dark" periods)."""
        if len(self.t) < 2:
            return []
        d = np.diff(self.t)
        return [(float(self.t[i]), float(self.t[i + 1]))
                for i in np.flatnonzero(d > min_gap_s)]

    def position_at(self, ts):
        """Interpolate position at `ts` (scalar or array of unix seconds).

        Returns ``(lat, lon, covered)`` where `covered` is False for times outside
        the track's own span. Times *inside* an AIS gap are still interpolated: a
        dark interval is exactly where a discharge hides, so we must be able to
        reason about where the vessel probably was. The resulting uncertainty is
        reported by the gap features rather than by refusing to answer.
        """
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        if len(self.t) == 0:
            nan = np.full(ts.shape, np.nan)
            return nan, nan, np.zeros(ts.shape, dtype=bool)
        if len(self.t) == 1:
            return (np.full(ts.shape, self.lat[0]), np.full(ts.shape, self.lon[0]),
                    np.abs(ts - self.t[0]) < 3600)
        la = np.interp(ts, self.t, self.lat)
        lo = np.interp(ts, self.t, self.lon)
        covered = (ts >= self.t[0]) & (ts <= self.t[-1])
        return la, lo, covered

    def value_at(self, ts, field: str):
        """Nearest-observation value of `sog` / `cog` at each time in `ts`."""
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        arr = getattr(self, field)
        if len(self.t) == 0:
            return np.full(ts.shape, np.nan)
        idx = np.clip(np.searchsorted(self.t, ts), 0, len(self.t) - 1)
        prev = np.clip(idx - 1, 0, len(self.t) - 1)
        take_prev = np.abs(ts - self.t[prev]) < np.abs(self.t[idx] - ts)
        return arr[np.where(take_prev, prev, idx)]

    def nearest_fix_dt(self, ts) -> np.ndarray:
        """Seconds from each `ts` to this track's closest actual observation."""
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        if len(self.t) == 0:
            return np.full(ts.shape, np.inf)
        if len(self.t) == 1:
            return np.abs(ts - self.t[0])
        idx = np.clip(np.searchsorted(self.t, ts), 1, len(self.t) - 1)
        return np.minimum(np.abs(ts - self.t[idx - 1]), np.abs(self.t[idx] - ts))

    def in_gap(self, ts, min_gap_s: float = GAP_S) -> np.ndarray:
        """True where `ts` falls inside an AIS gap longer than `min_gap_s`."""
        ts = np.atleast_1d(np.asarray(ts, dtype=float))
        out = np.zeros(ts.shape, dtype=bool)
        for g0, g1 in self.gaps(min_gap_s):
            out |= (ts > g0) & (ts < g1)
        return out


def _first_valid(series):
    s = series.dropna()
    return float(s.iloc[0]) if len(s) else np.nan


def build_tracks(df: pd.DataFrame) -> dict[int, Track]:
    """Group a cleaned AIS frame into per-MMSI :class:`Track` objects."""
    tracks: dict[int, Track] = {}
    if df.empty:
        return tracks
    for mmsi, g in df.groupby("mmsi", sort=False):
        g = g.sort_values("t")
        name = ""
        if "name" in g:
            # pandas' native "str" dtype (default since 3.0) represents a missing
            # value as a bare float NaN even after `.astype(str)` -- it does not
            # become the string "nan" the way object-dtype columns used to, so
            # `.lower()` on it raises. Filter those out with isinstance first.
            names = [n for n in g.name.astype(str).unique()
                     if isinstance(n, str) and n and n.lower() not in ("nan", "none", "")]
            if names:
                name = names[0]
        tracks[int(mmsi)] = Track(
            mmsi=int(mmsi),
            t=g.t.to_numpy(float),
            lat=g.lat.to_numpy(float),
            lon=g.lon.to_numpy(float),
            sog=g.sog.to_numpy(float),
            cog=g.cog.to_numpy(float),
            name=name,
            vtype=_first_valid(g.vtype) if "vtype" in g else np.nan,
            length=_first_valid(g.length) if "length" in g else np.nan,
            width=_first_valid(g.width) if "width" in g else np.nan,
            draft=_first_valid(g.draft) if "draft" in g else np.nan,
        )
    return tracks
