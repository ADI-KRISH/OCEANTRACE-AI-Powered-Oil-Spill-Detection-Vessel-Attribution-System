"""Validate attribution against real, documented spills with a known vessel.

    python -m validation.real_cases --list
    python -m validation.real_cases --case nyk_delphinus

This is the only part of the system measured on **real** data: a real NOAA
incident, real MarineCadastre AIS, and a vessel that was actually identified as
responsible. The system is given the reported location and a deliberately wide
time window, then has to pick the right vessel out of everything else at sea
nearby. Nothing about the answer is fed to it.

What this does and does not prove
---------------------------------
It validates **Module 3 (attribution)** given a reasonable origin. It does *not*
validate Module 1 on real SAR (the detector is trained on synthetic data) and it
does not validate Module 2 against a real drift, because the origin here is the
reported incident position rather than a hindcast from a real satellite
detection. Those need the Zenodo dataset and a matched Sentinel-1 scene.

Case difficulty is recorded per case and should be reported with the result: a
vessel that caught fire and stayed put is a far easier target than an operational
discharge from a ship under way.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from attribution.features import OriginHypothesis
from attribution.pipeline import attribute

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "validation")

#: Each case: a documented incident, the vessel found responsible, and the AIS
#: day it needs. `difficulty` is an honest label, not a score.
CASES = {
    "nyk_delphinus": {
        "title": "M/V NYK DELPHINUS fire, offshore Monterey Bay, CA",
        "date": "2021-05-14",
        "lat": 36.3767, "lon": -122.9000,
        "truth_mmsi": 636018222,
        "truth_name": "NYK DELPHINUS",
        "ais": "ais-2021-05-14.csv",
        "bbox": (35.3, 37.6, -124.5, -121.3),
        # No hour in the NOAA record, so a wide window is assumed rather than
        # fitted -- fitting the time to the vessel would make this circular.
        "time_utc": "12:00:00", "sigma_min": 180.0, "sigma_km": 6.0,
        "difficulty": ("EASY — the vessel caught fire and remained on scene, so it "
                       "sits inside the search radius for the whole window. An "
                       "operational discharge from a transiting ship is much harder."),
        "source": "NOAA IncidentNews",
    },
}


def run_case(key: str, radius_km: float = 30.0, top_n: int = 5, quiet: bool = False):
    """Run one case. Returns a dict with the rank the true vessel received."""
    if key not in CASES:
        raise SystemExit(f"unknown case {key!r}; try --list")
    c = CASES[key]
    ais_path = os.path.join(DATA, c["ais"])
    if not os.path.exists(ais_path):
        la0, la1, lo0, lo1 = c["bbox"]
        raise SystemExit(
            f"Missing {ais_path}.\nDownload it with:\n"
            f"  python -m validation.ais_download --date {c['date']} "
            f"--out-dir data/validation --bbox {la0} {la1} {lo0} {lo1}")

    t = pd.Timestamp(f"{c['date']}T{c['time_utc']}Z").timestamp()
    origin = OriginHypothesis.from_point(
        c["lat"], c["lon"], t, n=400,
        sigma_km=c["sigma_km"], sigma_min=c["sigma_min"])

    res = attribute(ais_path, origin=origin, radius_km=radius_km)
    hit = res.suspects[res.suspects.mmsi == c["truth_mmsi"]]
    rank = int(hit["rank"].iloc[0]) if not hit.empty else None
    pct = float(hit.attribution_pct.iloc[0]) if not hit.empty else 0.0

    if not quiet:
        print(f"\n{c['title']}")
        print(f"  {c['date']}  {c['lat']:.4f}, {c['lon']:.4f}   ({c['source']})")
        print(f"  Known responsible vessel: {c['truth_name']} (MMSI {c['truth_mmsi']})")
        print(f"\n  Difficulty: {c['difficulty']}\n")
        print(res.explain(top_n))
        print()
        if rank is None:
            print(f"  >>> MISS — {c['truth_name']} was filtered out before scoring.")
        else:
            print(f"  >>> {c['truth_name']} ranked #{rank} of {len(res.suspects)}"
                  f"  ({pct:.1f}% attribution)")

    return {"case": key, "rank": rank, "attribution_pct": pct,
            "n_candidates": res.n_candidates,
            "n_screened": res.n_vessels_screened,
            "truth_mmsi": c["truth_mmsi"], "truth_name": c["truth_name"],
            "difficulty": c["difficulty"], "result": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="nyk_delphinus")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--radius-km", type=float, default=30.0)
    a = ap.parse_args()

    if a.list:
        for k, c in CASES.items():
            print(f"{k:18s} {c['date']}  {c['truth_name']:16s} {c['title']}")
        return

    keys = list(CASES) if a.all else [a.case]
    results = [run_case(k, radius_km=a.radius_km) for k in keys]

    if len(results) > 1:
        print("\n" + "=" * 62)
        hits = [r for r in results if r["rank"] is not None]
        top1 = sum(r["rank"] == 1 for r in hits)
        print(f"{len(results)} real cases | Top-1 {top1}/{len(results)} | "
              f"recalled {len(hits)}/{len(results)}")


if __name__ == "__main__":
    main()
