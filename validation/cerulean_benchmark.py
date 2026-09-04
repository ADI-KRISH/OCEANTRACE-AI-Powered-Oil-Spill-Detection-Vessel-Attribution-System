"""Broad real-data validation: many SkyTruth Cerulean cases, not just one.

`validation/real_cases.py` proves the ranker on a single hand-picked, easy
incident (a vessel that caught fire and sat still). This module complements it
with statistical breadth: real Sentinel-1 detections, real MarineCadastre AIS,
across many independent cases, comparing our ranking to an independent
operational system (SkyTruth Cerulean) rather than to our own simulator.

    python -m validation.cerulean_benchmark --build --n-cases 25 --max-days 12
    python -m validation.cerulean_benchmark --run

Cerulean also uses AIS to pick its probable source, so agreement with it is
**not** ground truth -- it measures whether two independently-built AIS-based
methods concur. Report agreement, never accuracy. See `real_cases.py`'s
docstring for the one case validated against an actual documented responsible
vessel.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from .ais_download import download_day
from .fetch_cerulean import fetch_items, fetch_sources_for_slick

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "validation")
CASES_CSV = os.path.join(DATA, "cerulean_cases.csv")
RESULTS_CSV = os.path.join(DATA, "cerulean_results.csv")
REPORT_MD = os.path.join(DATA, "CERULEAN_VALIDATION.md")

#: US-shelf boxes with dense MarineCadastre coverage (lon_min, lat_min, lon_max,
#: lat_max). The deep Gulf / Bay of Campeche has Cerulean slicks too, but no
#: MarineCadastre AIS at all, so it is deliberately excluded -- including it
#: would just relabel "no US AIS" as "attribution failure".
US_BOXES = {
    "gulf": (-96.5, 27.2, -88.0, 30.6),
    "east": (-81.5, 24.0, -69.0, 42.0),
    "west": (-125.5, 32.0, -117.0, 49.0),
}


# ---------------------------------------------------------------------------
# Build: pick real slicks whose #1 probable source is a vessel
# ---------------------------------------------------------------------------

def build_cases(start="2022-01-01", end="2023-12-31", min_conf=0.90,
                n_cases=25, max_days=12) -> pd.DataFrame:
    dt = f"{start}T00:00:00Z/{end}T23:59:59Z"
    candidates = []
    for region, bbox in US_BOXES.items():
        df = fetch_items("public.slick_plus", bbox=bbox, datetime_range=dt,
                         max_items=1500)
        if df.empty:
            continue
        df = df[pd.to_numeric(df.get("machine_confidence"), errors="coerce")
                >= min_conf].copy()
        df["region"] = region
        candidates.append(df)
        print(f"[build] {region}: {len(df)} slicks >= {min_conf} confidence")
    if not candidates:
        raise SystemExit("Cerulean returned no candidate slicks for these boxes.")
    slicks = pd.concat(candidates, ignore_index=True)
    slicks = slicks.sort_values("machine_confidence", ascending=False)

    kept = []
    for _, s in slicks.iterrows():
        src = fetch_sources_for_slick(s["id"])
        if src.empty or src.iloc[0].get("source_type") != "VESSEL":
            continue
        vessels = src[src.source_type == "VESSEL"]
        kept.append({
            "slick_id": s["id"], "date": str(s["slick_timestamp"])[:10],
            "slick_timestamp": s["slick_timestamp"], "region": s["region"],
            "lat": s["lat"], "lon": s["lon"],
            "area_m2": s.get("area"), "length_m": s.get("length"),
            "machine_confidence": s["machine_confidence"],
            "cerulean_mmsi_rank1": str(vessels.iloc[0]["mmsi_or_structure_id"]),
            "cerulean_vessel_topk": "|".join(
                str(x) for x in vessels.mmsi_or_structure_id.head(5)),
            "slick_url": s.get("slick_url"),
        })
        time.sleep(0.05)
        if len(kept) >= n_cases * 6:
            break
    print(f"[build] {len(kept)} slicks with a vessel #1 source")
    if not kept:
        raise SystemExit("No slick in range had a vessel as its #1 source.")

    by_date = defaultdict(list)
    for x in kept:
        by_date[x["date"]].append(x)
    dates = sorted(by_date, key=lambda d: -max(x["machine_confidence"] for x in by_date[d]))

    picked = []
    for d in dates[:max_days]:
        for x in sorted(by_date[d], key=lambda x: -x["machine_confidence"])[:3]:
            picked.append(x)
            if len(picked) >= n_cases:
                break
        if len(picked) >= n_cases:
            break

    out = pd.DataFrame(picked)
    os.makedirs(DATA, exist_ok=True)
    out.to_csv(CASES_CSV, index=False)
    print(f"[build] {len(out)} cases over {out.date.nunique()} days -> {CASES_CSV}")
    print(out[["slick_id", "date", "region", "lat", "lon",
              "machine_confidence", "cerulean_mmsi_rank1"]].to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Run: attribute() on each case, compare to Cerulean's ranking
# ---------------------------------------------------------------------------

def run_cases(cases: pd.DataFrame, age_max_h=24.0, radius_km=75.0) -> pd.DataFrame:
    from attribution.features import OriginHypothesis
    from attribution.pipeline import attribute

    rows = []
    for _, c in cases.sort_values("date").iterrows():
        print(f"\n[case {c.slick_id}] {c.date} ({c.region})  "
              f"Cerulean#1={c.cerulean_mmsi_rank1}")
        try:
            # Whole day, no bbox filter at download time: several cases on the
            # same date can sit in different regions, and `attribute()` already
            # applies its own bbox to whatever CSV it is handed, so pre-filtering
            # here would only risk silently caching the wrong crop under a
            # filename the next case reuses without noticing.
            ais_path = download_day(c.date, out_dir=DATA)
        except Exception as exc:                       # noqa: BLE001
            print(f"  !! AIS unavailable: {exc}")
            rows.append({"slick_id": c.slick_id, "date": c.date, "n_candidates": 0})
            continue

        length_km = (c.get("length_m") or 6000) / 1000.0
        sigma_km = float(np.clip(length_km / 2.0, 4.0, 18.0))
        t = pd.Timestamp(c.slick_timestamp).timestamp()
        origin = OriginHypothesis.from_point(
            c.lat, c.lon, t, n=400, sigma_km=sigma_km, sigma_min=age_max_h * 30.0)

        res = attribute(ais_path, origin=origin, radius_km=radius_km)
        if res.suspects.empty:
            print("  no candidates")
            rows.append({"slick_id": c.slick_id, "date": c.date, "n_candidates": 0})
            continue

        sus = res.suspects.copy()
        sus["mmsi"] = sus.mmsi.astype(str)
        topk = str(c.cerulean_vessel_topk).split("|")
        cer1 = c.cerulean_mmsi_rank1
        hit = sus[sus.mmsi == cer1]
        in_cand = not hit.empty
        our_rank = int(hit["rank"].iloc[0]) if in_cand else None
        our1 = sus.mmsi.iloc[0]

        rows.append({
            "slick_id": c.slick_id, "date": c.date, "region": c.region,
            "n_candidates": len(sus), "n_screened": res.n_vessels_screened,
            "cerulean_r1": cer1, "our_r1": our1,
            "our_r1_name": sus.name.iloc[0],
            "cerulean_r1_in_candidates": in_cand,
            "our_rank_of_cerulean_r1": our_rank,
            "agree_at_1": our1 == cer1,
            "cerulean_r1_in_our_top3": bool(our_rank and our_rank <= 3),
            "our_r1_in_cerulean_topk": our1 in topk,
        })
        print(f"  our #1 = {our1} ({sus.name.iloc[0]}); "
              f"Cerulean #1 rank with us = {our_rank}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_CSV, index=False)
    return out


def summarise(out: pd.DataFrame) -> str:
    ok = out[out.n_candidates > 0] if "n_candidates" in out else out.iloc[0:0]
    lines = ["===== Cerulean benchmark summary =====",
            f"cases run                     : {len(out)}",
            f"cases with >=1 candidate      : {len(ok)}"]
    if len(ok):
        cov = ok.cerulean_r1_in_candidates.mean()
        lines.append(f"Cerulean #1 vessel seen by us : {cov:.0%}")
        seen = ok[ok.cerulean_r1_in_candidates]
        if len(seen):
            lines.append(f"  agree@1                    : {seen.agree_at_1.mean():.0%}")
            lines.append(f"  Cerulean #1 in our top-3   : {seen.cerulean_r1_in_our_top3.mean():.0%}")
            lines.append(f"  median rank we give it     : {seen.our_rank_of_cerulean_r1.median():.1f}")
        lines.append(f"our #1 within Cerulean top-5  : {ok.our_r1_in_cerulean_topk.mean():.0%}")
    return "\n".join(lines)


def write_report(out: pd.DataFrame):
    body = f"""# Broad real-data validation — attribution vs. SkyTruth Cerulean

Complements `validation/real_cases.py` (one hand-picked, easy real incident)
with statistical breadth: {len(out)} real Sentinel-1 slicks (US shelf,
2022-2023) whose #1 probable source Cerulean labels as a vessel, compared to
this module's own ranking on the same slick + real MarineCadastre AIS.

**Not ground truth.** Cerulean's own ranker also uses AIS. This measures
*agreement with an independent operational system*, which is the best real
check available without enforcement records.

Reproduce:
```
python -m validation.cerulean_benchmark --build --n-cases 25 --max-days 12
python -m validation.cerulean_benchmark --run
```

```
{summarise(out)}
```

## Cases

```
{out.to_string(index=False)}
```

## Reading this

- **Coverage** below 100% is the dominant limiter, not the scoring model:
  Cerulean draws on global satellite AIS; MarineCadastre is US-only, so a
  vessel it names may simply not be in our feed.
- When the vessel *is* in our feed, `track_match` and the behavioural features
  (`gap_over_origin`, `slow_steaming`, `loiter_score`) are what should be
  pulling it up the ranking -- if agreement stays low even there, that points
  at the origin cloud (sigma/age window) rather than the scorer.
"""
    with open(REPORT_MD, "w") as fh:
        fh.write(body)
    print(f"\nwrote {REPORT_MD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--min-conf", type=float, default=0.90)
    ap.add_argument("--n-cases", type=int, default=25)
    ap.add_argument("--max-days", type=int, default=12)
    ap.add_argument("--age-max-h", type=float, default=24.0)
    ap.add_argument("--radius-km", type=float, default=75.0)
    a = ap.parse_args()

    if not a.build and not a.run:
        a.build = a.run = True

    if a.build:
        build_cases(a.start, a.end, a.min_conf, a.n_cases, a.max_days)

    if a.run:
        if not os.path.exists(CASES_CSV):
            raise SystemExit(f"No {CASES_CSV}; run with --build first.")
        cases = pd.read_csv(CASES_CSV, dtype={"cerulean_mmsi_rank1": str})
        out = run_cases(cases, a.age_max_h, a.radius_km)
        print("\n" + summarise(out))
        write_report(out)


if __name__ == "__main__":
    main()
