"""Download a NOAA daily AIS file for a given date.

    python -m validation.ais_download --date 2022-10-11 \
        --bbox 38.5 40.5 -74.5 -71.5

Files are ~250 MB-1 GB per day, so a `--bbox` extracts and keeps only rows
inside it -- what makes a validation set of many incident days practical to
store.

Two sources, tried in this order:

1. **NOAA OCM Azure blob mirror** (`noaaocm.blob.core.windows.net`), zstd-
   compressed CSVs. This is the default and should be preferred: it streams at
   several MB/s and completes reliably.
2. **MarineCadastre direct** (`coast.noaa.gov`) as a fallback. In practice this
   server frequently drops connections partway through files this large --
   `requests.get(..., stream=True)` returns HTTP 200 and then the socket stalls
   or resets after a few hundred MB with no exception raised by `iter_content`
   until the connection is force-closed, which silently produces a truncated,
   unreadable zip. It is kept only as a fallback for dates the Azure mirror
   does not carry.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import zipfile

import pandas as pd
import requests

AZURE_TEMPLATE = ("https://noaaocm.blob.core.windows.net/ais/csv2/csv{year}/"
                  "ais-{year}-{month:02d}-{day:02d}.csv.zst")
#: MarineCadastre reorganised its bucket layout in 2022; both forms are tried.
MC_URL_TEMPLATES = [
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip",
    "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.csv.zip",
]


def download_day(date: str, out_dir: str = "ais-dataset",
                 bbox=None, keep_zip: bool = False, source: str = "auto") -> str:
    """Fetch one UTC day of US AIS. Returns the written CSV path."""
    ts = pd.Timestamp(date)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"ais-{ts:%Y-%m-%d}.csv")
    if os.path.exists(out_csv):
        print(f"{out_csv} already present; skipping download.", file=sys.stderr)
        return out_csv

    if source in ("auto", "azure"):
        try:
            return _download_azure(ts, out_csv, bbox)
        except Exception as exc:                      # noqa: BLE001
            if source == "azure":
                raise
            print(f"Azure mirror failed ({exc}); falling back to MarineCadastre.",
                 file=sys.stderr)

    return _download_marinecadastre(ts, out_csv, bbox, keep_zip)


def _download_azure(ts: pd.Timestamp, out_csv: str, bbox) -> str:
    """curl (resumable, retried) + zstd -- reliably fast for the ~1 GB dailies."""
    url = AZURE_TEMPLATE.format(year=ts.year, month=ts.month, day=ts.day)
    print(f"Fetching {url}", file=sys.stderr)
    zst_path = out_csv + ".zst"
    subprocess.run(
        ["curl", "-sL", "--fail", "--retry", "6", "--retry-all-errors",
         "--retry-delay", "3", "-C", "-", "-o", zst_path, url], check=True)

    raw_csv = out_csv if bbox is None else out_csv + ".raw"
    subprocess.run(["zstd", "-dfq", zst_path, "-o", raw_csv], check=True)
    os.remove(zst_path)
    if bbox is not None:
        with open(raw_csv, "rb") as fh:
            _write_filtered(fh, out_csv, bbox, lat_col="latitude", lon_col="longitude")
        os.remove(raw_csv)
    return out_csv


def _download_marinecadastre(ts: pd.Timestamp, out_csv: str, bbox, keep_zip: bool) -> str:
    last_err = None
    for tmpl in MC_URL_TEMPLATES:
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
        f"Could not download AIS for {ts:%Y-%m-%d} -- {last_err}. "
        "Check the date is available at https://coast.noaa.gov/htdata/CMSP/AISDataHandler/"
    )


def _write_filtered(fh, out_csv: str, bbox, lat_col: str = "LAT", lon_col: str = "LON"):
    """Stream the CSV out, optionally keeping only rows inside `bbox`."""
    if bbox is None:
        pd.read_csv(fh).to_csv(out_csv, index=False)
        return
    la0, la1, lo0, lo1 = bbox
    first, kept, seen = True, 0, 0
    for chunk in pd.read_csv(fh, chunksize=500_000, low_memory=False):
        seen += len(chunk)
        lat = pd.to_numeric(chunk.get(lat_col, chunk.get(lat_col.lower())), errors="coerce")
        lon = pd.to_numeric(chunk.get(lon_col, chunk.get(lon_col.lower())), errors="coerce")
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
    ap.add_argument("--source", choices=["auto", "azure", "marinecadastre"],
                    default="auto")
    a = ap.parse_args()
    path = download_day(a.date, a.out_dir, a.bbox, a.keep_zip, a.source)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
