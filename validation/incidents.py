"""NOAA IncidentNews curation -> a shortlist of real, checkable spill cases.

Synthetic scenarios prove the ranker works against its own simulator, which is a
weak claim. This module narrows NOAA's incident export down to the cases that
could actually falsify it: an oil spill, from a vessel, with coordinates, in the
AIS era, ideally with a vessel name recoverable from the title.

The name extraction is best-effort and is explicitly a *guess* -- the column is
named `vessel_guess` for that reason. Resolving a guess to an MMSI is a manual
step through Equasis or MarineTraffic; nothing here should be treated as an
identification.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

#: Substrings implying the incident involved a vessel rather than a pipeline,
#: refinery, storage tank or unknown "mystery" source.
VESSEL_HINTS = [
    "vessel", "ship", "barge", "tanker", "tug", "boat", "freighter", "trawler",
    "fishing", "cargo", "ferry", "yacht", "sunken", "sank", "sinking", "capsiz",
    "grounding", "aground", "collision", "allision", "t/v", "m/v", "f/v", "s/s",
]

#: Substrings implying oil specifically (as opposed to chemical-only releases).
OIL_HINTS = [
    "oil", "diesel", "fuel", "crude", "bunker", "gasoline", "petrol", "hydraulic",
    "lube", "ifo", "hfo", "slick", "sheen", "petroleum", "condensate",
]

#: Vessel-name prefixes used in NOAA titles, e.g. "T/V EXXON VALDEZ".
_PREFIX_RE = re.compile(
    r"\b(?:M/V|MV|T/V|TV|F/V|FV|S/S|SS|M/T|MT|USCGC|USNS|R/V)[\s.]+"
    r"([A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,3})"
)

#: Words that look like names but are not, so a match on them is discarded.
_STOPWORDS = {
    "OIL", "SPILL", "BARGE", "TANKER", "VESSEL", "SHIP", "TUG", "BOAT", "FISHING",
    "RIVER", "BAY", "PORT", "HARBOR", "HARBOUR", "CREEK", "LAKE", "ISLAND",
    "MYSTERY", "UNKNOWN", "PIPELINE", "PLATFORM", "TERMINAL", "REFINERY",
}

#: AIS coverage is only usable from roughly 2009 onward.
AIS_ERA_YEAR = 2009


def _find_col(df: pd.DataFrame, *names) -> str | None:
    lower = {str(c).lower().strip().replace(" ", "_"): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    for n in names:
        for k, v in lower.items():
            if n in k:
                return v
    return None


def extract_vessel_name(title: str) -> str | None:
    """Best-effort vessel name from an incident title. None when unsure.

    Prefers an explicit prefix (``T/V EXXON VALDEZ``) and falls back to a run of
    capitalised words before a separator. Returns None rather than a bad guess --
    a wrong name costs an analyst a wasted Equasis lookup, and worse, invites a
    confident but false attribution.
    """
    if not isinstance(title, str) or not title.strip():
        return None

    m = _PREFIX_RE.search(title)
    if m:
        cand = m.group(1).strip(" ,.-")
        if cand.upper() not in _STOPWORDS and len(cand) > 2:
            return cand

    # Fall back: an ALL-CAPS run of 1-3 words, which is how NOAA writes most names.
    head = re.split(r"[,:;()\-]| spill| incident| release", title, maxsplit=1)[0]
    # Hull numbers and roman numerals ("DBL 152", "ATHOS I") are part of the name
    # and are what distinguishes sisters in a series, so they must survive.
    toks = re.findall(r"\b[A-Z][A-Z0-9'\-]{2,}\b|\b\d{1,4}\b|\b(?:I{1,3}|IV|VI{0,3}|IX|XI{0,2})\b",
                      head)
    toks = [t for t in toks if t not in _STOPWORDS]
    # Trailing numerals only count as a suffix, never as the name itself.
    while toks and re.fullmatch(r"\d{1,4}|I{1,3}|IV|VI{0,3}|IX|XI{0,2}", toks[0]):
        toks.pop(0)
    words = [t for t in toks if not re.fullmatch(r"\d{1,4}|I{1,3}|IV|VI{0,3}|IX|XI{0,2}", t)]
    if 1 <= len(words) <= 3:
        return " ".join(toks[:len(words) + 1]) if len(toks) > len(words) else " ".join(toks)
    return None


def curate(
    path: str = "incidents.csv",
    out_path: str | None = "outputs/real_cases.csv",
    ais_era_only: bool = False,
) -> pd.DataFrame:
    """Filter NOAA IncidentNews to oil + vessel + coordinates.

    Returns a frame with `name`, `date`, `lat`, `lon`, `year`, `vessel_guess`,
    `is_oil`, `is_vessel`, `ais_era`, plus a `priority` flag marking the cases
    worth resolving to an MMSI first.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download the IncidentNews export from "
            "https://incidentnews.noaa.gov/raw/index and save it there."
        )
    df = pd.read_csv(path, low_memory=False)

    c_name = _find_col(df, "name", "title", "incident")
    c_date = _find_col(df, "date", "open_date", "incident_date")
    c_lat = _find_col(df, "lat", "latitude")
    c_lon = _find_col(df, "lon", "long", "longitude")
    c_type = _find_col(df, "commodity", "material", "product", "threat")
    if not all([c_name, c_lat, c_lon]):
        raise ValueError(f"unexpected IncidentNews schema: {list(df.columns)[:20]}")

    out = pd.DataFrame({
        "name": df[c_name].astype(str),
        "lat": pd.to_numeric(df[c_lat], errors="coerce"),
        "lon": pd.to_numeric(df[c_lon], errors="coerce"),
    })
    out["date"] = (pd.to_datetime(df[c_date], errors="coerce")
                   if c_date else pd.NaT)
    out["year"] = out.date.dt.year
    # `.fillna("")` before concatenation, not after: pandas' native "str" dtype
    # (default since 3.0) represents a missing `astype(str)` value as a bare
    # float NaN rather than the string "nan", so `Series + Series` silently
    # propagates NaN through the whole row instead of raising -- and the
    # subsequent `.apply(lambda s: ... in s)` then crashes on a float, not a
    # missing string, for every incident whose commodity/threat column was
    # blank (the common case: 1,093 of 4,929 rows here have no `commodity`).
    type_txt = df[c_type].astype(str).str.lower().fillna("") if c_type else ""
    haystack = out.name.str.lower().fillna("") + " " + type_txt

    out["is_oil"] = haystack.apply(lambda s: any(h in s for h in OIL_HINTS))
    out["is_vessel"] = haystack.apply(lambda s: any(h in s for h in VESSEL_HINTS))
    out["vessel_guess"] = out.name.apply(extract_vessel_name)
    out["ais_era"] = out.year >= AIS_ERA_YEAR

    keep = out.lat.notna() & out.lon.notna() & out.is_oil & out.is_vessel
    if ais_era_only:
        keep &= out.ais_era
    res = out[keep].copy()

    # Cases worth an analyst's time first: named, in the AIS era, at sea.
    res["priority"] = res.vessel_guess.notna() & res.ais_era
    res = res.sort_values(["priority", "year"], ascending=[False, False])

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        res.to_csv(out_path, index=False)
    return res


def summarise(res: pd.DataFrame) -> str:
    return (
        f"{len(res)} oil + vessel incidents with coordinates\n"
        f"  {int(res.vessel_guess.notna().sum())} with a vessel-name guess\n"
        f"  {int(res.ais_era.sum())} in the AIS era (>= {AIS_ERA_YEAR})\n"
        f"  {int(res.priority.sum())} priority cases (named AND AIS-era)"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Curate NOAA IncidentNews.")
    ap.add_argument("--input", default="incidents.csv")
    ap.add_argument("--output", default="outputs/real_cases.csv")
    a = ap.parse_args()
    r = curate(a.input, a.output)
    print(summarise(r))
    print(f"\n-> {a.output}")
    print(r[r.priority].head(20)[["name", "year", "lat", "lon", "vessel_guess"]]
          .to_string(index=False))
