"""Download a NOAA MarineCadastre daily AIS file for a given date.

    python -m oilspill_attribution.tools.ais_download --date 2022-10-11 \
        --bbox 38.5 40.5 -74.5 -71.5

Files are ~100-300 MB zipped per day. Passing a `--bbox` extracts and keeps only
rows inside it, which is what makes a validation set of 15-20 incident days
practical to store: a full year of raw dailies is well over 50 GB.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import pandas as pd
import requests

#: MarineCadastre reorganised its bucket layout in 2022; both forms are tried.
URL_TEMPLATES = [
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip",
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.csv.zip",
]


def download_day(date: str, out_dir: str = "ais-dataset",
                 bbox=None, keep_zip: bool = False) -> str:
    """Fetch one UTC day of US AIS. Returns the written CSV path."""
    ts = pd.Timestamp(date)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"ais-{ts:%Y-%m-%d}.csv")
    if os.path.exists(out_csv):
        print(f"{out_csv} already present; skipping download.", file=sys.stderr)
        return out_csv

    last_err = None
    for tmpl in URL_TEMPLATES:
        url = tmpl.format(year=ts.year, month=ts.month, day=ts.day)
        print(f"Trying {url}", file=sys.stderr)
        try:
            r = requests.get(url, stream=True, timeout=120)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue
            blob = io.BytesIO()
            total = 0
            for chunk in r.iter_content(1 << 20):
                blob.write(chunk)
                total += len(chunk)
                print(f"\r  {total/1e6:.0f} MB", end="", file=sys.stderr)
            print(file=sys.stderr)
            blob.seek(0)

            with zipfile.ZipFile(blob) as z:
                inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not inner:
                    last_err = "no CSV inside the archive"
                    continue
                with z.open(inner[0]) as fh:
                    _write_filtered(fh, out_csv, bbox)
            if keep_zip:
                with open(out_csv.replace(".csv", ".zip"), "wb") as fh:
                    fh.write(blob.getvalue())
            return out_csv
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            last_err = str(exc)

    raise RuntimeError(
        f"Could not download AIS for {date} -- {last_err}. "
        "Check the date is available at https://coast.noaa.gov/htdata/CMSP/AISDataHandler/"
    )


def _write_filtered(fh, out_csv: str, bbox):
    """Stream the CSV out, optionally keeping only rows inside `bbox`."""
    if bbox is None:
        pd.read_csv(fh).to_csv(out_csv, index=False)
        return
    la0, la1, lo0, lo1 = bbox
    first, kept, seen = True, 0, 0
    for chunk in pd.read_csv(fh, chunksize=500_000, low_memory=False):
        seen += len(chunk)
        lat = pd.to_numeric(chunk.get("LAT"), errors="coerce")
        lon = pd.to_numeric(chunk.get("LON"), errors="coerce")
        sel = chunk[lat.between(la0, la1) & lon.between(lo0, lo1)]
        kept += len(sel)
        sel.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        print(f"\r  filtered {kept:,}/{seen:,} rows", end="", file=sys.stderr)
    print(file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--out-dir", default="ais-dataset")
    ap.add_argument("--bbox", nargs=4, type=float,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    help="keep only rows inside this box")
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args()
    path = download_day(a.date, a.out_dir, a.bbox, a.keep_zip)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
