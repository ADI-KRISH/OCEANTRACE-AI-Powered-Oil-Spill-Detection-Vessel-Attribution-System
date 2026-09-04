"""Pull a regional benchmark from SkyTruth's Cerulean OGC API.

Cerulean runs a CNN over every Sentinel-1 scene and cross-correlates detected
slicks against AIS and fixed infrastructure, publishing per-slick a ranked source
list with MMSI and a collated score.

    python -m oilspill_attribution.tools.fetch_cerulean --bbox 68 6 90 22 \
        --start 2023-01-01 --end 2023-12-31

IMPORTANT -- how this benchmark may and may not be used: Cerulean's ranker also
consumes AIS, so its answer is *not* independent ground truth. Agreement with it
measures whether two AIS-based methods concur, not whether either is right.
Report agreement rate; never report it as accuracy.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests

BASE = "https://api.cerulean.skytruth.org/collections"
SLICK_COLLECTION = "public.slick_plus"
SOURCE_COLLECTION = "public.source_plus"
UA = {"User-Agent": "SIH26143-oilspill-attribution/1.0 (research)"}


def _get(url: str, params: dict, timeout: int = 60, retries: int = 3):
    """GET with linear backoff; the API rate-limits bulk pulls."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Cerulean request failed after {retries} tries -- {last}")


def fetch_items(collection: str, bbox=None, datetime_range=None,
                limit: int = 1000, max_items: int = 20000,
                extra_params: dict | None = None) -> pd.DataFrame:
    """Page through an OGC Features collection into a flat DataFrame.

    The API answers `f=json` with a **flat list** of already-merged property
    dicts (geometry as a WKT string under `geometry`), not a GeoJSON
    FeatureCollection -- despite `f=json` looking like it should behave like the
    unparametrised endpoint, which does return `{"type": "FeatureCollection",
    ...}`. `_extract_rows` accepts either shape so a change on either side does
    not silently start dropping every row.
    """
    url = f"{BASE}/{collection}/items"
    params = {"limit": limit, "f": "json", **(extra_params or {})}
    if bbox:
        params["bbox"] = ",".join(str(b) for b in bbox)
    if datetime_range:
        params["datetime"] = datetime_range

    rows, offset = [], 0
    while len(rows) < max_items:
        params["offset"] = offset
        data = _get(url, params)
        page = _extract_rows(data)
        if not page:
            break
        rows.extend(page)
        print(f"  {collection}: {len(rows)} items", file=sys.stderr)
        if len(page) < limit:
            break
        offset += limit
    return pd.DataFrame(rows)


def _extract_rows(data) -> list[dict]:
    """Normalise either API response shape into a list of flat property dicts
    with `lon`/`lat` added as the geometry's vertex centroid."""
    if isinstance(data, list):
        items = data
        get_props = lambda item: item                     # noqa: E731
        get_geom = lambda item: item.get("geometry")       # noqa: E731
    else:
        items = data.get("features", [])
        get_props = lambda item: dict(item.get("properties", {}))  # noqa: E731
        get_geom = lambda item: item.get("geometry")               # noqa: E731

    out = []
    for item in items:
        rec = dict(get_props(item))
        rec.setdefault("id", item.get("id") if isinstance(item, dict) else None)
        geom = get_geom(item)
        pt = _centroid_wkt(geom) if isinstance(geom, str) else _centroid_geojson(geom)
        if pt:
            rec["lon"], rec["lat"] = pt
        out.append(rec)
    return out


def _centroid_geojson(geom):
    """Mean (lon, lat) over every vertex in a nested GeoJSON coordinate array."""
    if not isinstance(geom, dict):
        return None
    pts = []

    def walk(node):
        if (isinstance(node, (list, tuple)) and len(node) >= 2
                and all(isinstance(c, (int, float)) for c in node[:2])):
            pts.append((float(node[0]), float(node[1])))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(geom.get("coordinates"))
    if not pts:
        return None
    lons, lats = zip(*pts)
    return sum(lons) / len(lons), sum(lats) / len(lats)


_WKT_PAIR_RE = re.compile(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")


def _centroid_wkt(wkt: str):
    """Mean (lon, lat) over every coordinate pair in a WKT (MULTI)POLYGON/POINT
    string, e.g. ``"SRID=4326;MULTIPOLYGON(((-88.85 30.33,...)))"``."""
    if not wkt:
        return None
    body = wkt.split(";", 1)[-1]
    pairs = _WKT_PAIR_RE.findall(body)
    if not pairs:
        return None
    lons = [float(a) for a, _ in pairs]
    lats = [float(b) for _, b in pairs]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def fetch_sources_for_slick(slick_id: int, limit: int = 50) -> pd.DataFrame:
    """Ranked sources for one slick -- the per-case call the benchmark needs."""
    data = _get(f"{BASE}/{SOURCE_COLLECTION}/items", {"slick_id": int(slick_id),
                                                       "limit": limit, "f": "json"})
    df = pd.DataFrame(_extract_rows(data))
    if not df.empty and "source_rank" in df:
        df = df.sort_values("source_rank")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    help="lon_min lat_min lon_max lat_max")
    ap.add_argument("--start", help="ISO date, e.g. 2023-01-01")
    ap.add_argument("--end", help="ISO date")
    ap.add_argument("--out-dir", default="data/cerulean")
    ap.add_argument("--max-items", type=int, default=20000)
    a = ap.parse_args()

    dt = None
    if a.start and a.end:
        dt = f"{a.start}T00:00:00Z/{a.end}T23:59:59Z"

    os.makedirs(a.out_dir, exist_ok=True)
    print("Fetching slicks...", file=sys.stderr)
    slicks = fetch_items(SLICK_COLLECTION, a.bbox, dt, max_items=a.max_items)
    print("Fetching sources...", file=sys.stderr)
    sources = fetch_items(SOURCE_COLLECTION, a.bbox, dt, max_items=a.max_items)

    sp = os.path.join(a.out_dir, "slicks.csv")
    op = os.path.join(a.out_dir, "sources.csv")
    slicks.to_csv(sp, index=False)
    sources.to_csv(op, index=False)
    print(f"\n{len(slicks)} slicks -> {sp}")
    print(f"{len(sources)} sources -> {op}")

    if not sources.empty and "source_type" in sources:
        print("\nSource type breakdown:")
        print(sources.source_type.value_counts().to_string())
    print("\nNOTE: Cerulean's ranker also uses AIS. Report AGREEMENT with it, "
          "not accuracy against it.")


if __name__ == "__main__":
    main()
