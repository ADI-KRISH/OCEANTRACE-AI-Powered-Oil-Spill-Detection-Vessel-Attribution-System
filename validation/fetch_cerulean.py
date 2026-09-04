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
                limit: int = 1000, max_items: int = 20000) -> pd.DataFrame:
    """Page through an OGC Features collection into a flat DataFrame."""
    url = f"{BASE}/{collection}/items"
    params = {"limit": limit, "f": "json"}
    if bbox:
        params["bbox"] = ",".join(str(b) for b in bbox)
    if datetime_range:
        params["datetime"] = datetime_range

    rows, offset = [], 0
    while len(rows) < max_items:
        params["offset"] = offset
        data = _get(url, params)
        feats = data.get("features", [])
        if not feats:
            break
        for f in feats:
            rec = dict(f.get("properties", {}))
            rec["id"] = f.get("id")
            geom = f.get("geometry") or {}
            rec["geometry_type"] = geom.get("type")
            # Keep a representative point; the full polygon stays in the API.
            coords = geom.get("coordinates")
            pt = _first_point(coords)
            if pt:
                rec["lon"], rec["lat"] = pt
            rows.append(rec)
        print(f"  {collection}: {len(rows)} items", file=sys.stderr)
        if len(feats) < limit:
            break
        offset += limit
    return pd.DataFrame(rows)


def _first_point(coords):
    """Descend a GeoJSON coordinate nest to its first (lon, lat) pair."""
    while isinstance(coords, (list, tuple)) and coords:
        if len(coords) >= 2 and all(isinstance(c, (int, float)) for c in coords[:2]):
            return float(coords[0]), float(coords[1])
        coords = coords[0]
    return None


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
