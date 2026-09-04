# Datasets

Large data files are **not committed** (see `.gitignore`). Download them here.

## 1. Oil-spill SAR segmentation — Krestenitis et al. (Module 1)

The base dataset for the detection model. Sentinel-1 GRD patches with 5-class
masks: `sea`, `oil_spill`, `look_alike`, `ship`, `land`.

- Paper: Krestenitis et al. (2019), *Oil Spill Identification from Satellite
  Images Using Deep Neural Networks*, Remote Sensing 11(15):1762.
- Download: https://m4d.iti.gr/oil-spill-detection-dataset/
  (Zenodo mirror: search "Oil Spill Detection Dataset" — access is granted on
  request via a short form.)

Extract so the layout is:

```
data/oil_spill/
  train/images/*.jpg
  train/labels/*.png
  val/images/*.jpg
  val/labels/*.png
```

`detection/data.py` also accepts `image/`+`label/`, `masks/`, or `annotations/`
as directory names, and handles both indexed and RGB masks.

**Until this is downloaded**, every command still runs — `--synthetic` generates
patches with the same five classes. Those numbers describe the generator, not
Sentinel-1, and must never be quoted as results.

## 2. AIS — MarineCadastre (Module 3)

- Access: https://marinecadastre.gov/accessais/
- Daily US files: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{YYYY}/AIS_{YYYY}_{MM}_{DD}.zip`
- Schema: `MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName,
  VesselType, Length, ...`

Files are 100–300 MB/day zipped, so crop to the region of interest on download.

## 3. Met-ocean forcing — for OpenDrift (Module 2)

- Currents: Copernicus Marine (CMEMS) — https://marine.copernicus.eu/
  (free account required; `copernicusmarine` CLI)
- Wind: ERA5 via the Copernicus Climate Data Store, or NOAA GFS.
- Stokes drift: included in CMEMS wave products.

Keep one small canned regional subset in `data/forcing/` for the demo, so the
pipeline runs without live API access.

## 4. Validation incidents

- NOAA IncidentNews: https://incidentnews.noaa.gov/raw/index — save the export
  as `incidents.csv` at the repo root, then:
  ```bash
  python -m validation.incidents --input incidents.csv \
      --output data/validation/demo_candidates.csv
  ```
  Produces `data/validation/demo_candidates.csv` (committed: 1,320 oil+vessel
  incidents with coordinates, 337 "priority" — named vessel, AIS era).
- SkyTruth Cerulean (slicks with AIS-correlated sources):
  https://api.cerulean.skytruth.org
  ```bash
  python -m validation.cerulean_benchmark --build --n-cases 25 --max-days 12
  python -m validation.cerulean_benchmark --run
  ```
  Produces `data/validation/cerulean_cases.csv`, `cerulean_results.csv` and
  `CERULEAN_VALIDATION.md` (all committed — small; the multi-GB daily AIS files
  it downloads into `data/validation/ais-*.csv` along the way are not).

> Cerulean's ranker also consumes AIS, so agreement with it measures whether two
> AIS-based methods concur — **not** whether either is correct. Report agreement,
> never accuracy.
