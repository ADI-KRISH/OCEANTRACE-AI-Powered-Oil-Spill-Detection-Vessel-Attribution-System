# CLAUDE.md

> Project guide for Claude Code. Read this before doing any work in this repo.

## What we are building

An automated pipeline that:
1. **Detects** oil spills at sea from satellite imagery (Sentinel-1 SAR).
2. **Characterizes** each spill (area, shape, orientation, rough age).
3. **Hindcasts** the slick backward in time to estimate its origin point + time window, and forecasts forward spread.
4. **Attributes** the spill to a responsible vessel by correlating the origin window with historic AIS (vessel tracking) data, and produces an explainable ranked suspect list.
5. **Visualizes** everything on an interactive web map.

Problem owner: **National Technical Research Organisation (NTRO)** — Disaster Management theme.
Two things judges care most about: **explainable attribution** (defensible evidence, not a black-box score) and **honest uncertainty** (no false precision on age/origin).

## System modules

### Module 1 — Detection & Characterization (`/detection`)
- Input: Sentinel-1 SAR scenes (oil dampens capillary waves → dark slicks).
- Model: semantic segmentation (U-Net or DeepLabv3+) into 5 classes: `sea`, `oil_spill`, `look_alike`, `land`, `ship`.
- Base dataset: Zenodo Sentinel-1 SAR Oil Spill Dataset (Krestenitis et al.) — same 5 classes.
- Hard problem: suppress **look-alikes** (low-wind zones, algae, rain cells) that mimic oil.
- Characterization: area (pixel count × resolution²), perimeter, centroid, orientation, shape descriptors. Age = rough estimate from spreading/weathering models, always tagged low-confidence.

### Module 2 — Drift Hindcast/Forecast (`/drift`)
- Use **OpenDrift** (OpenOil model) — do NOT reimplement Lagrangian tracking.
- Forcing data: ocean currents (Copernicus Marine / CMEMS), wind (ERA5 or GFS), Stokes drift.
- Backward run (negative timestep) → origin **probability heatmap** + time window (not a single point).
- Forward run → predicted spread.
- Seed particles across the detected slick polygon; run an ensemble.

### Module 3 — Vessel Attribution (`/attribution`)
- AIS format: MarineCadastre (MMSI, timestamp, lat/lon, SOG, COG). Real if available, else synthetic for the region.
- Reconstruct + interpolate tracks within the origin space-time window; filter out irrelevant traffic.
- **Explainable weighted scoring** per vessel:
  - proximity to origin probability cloud
  - trajectory intersection with backward-drift particles
  - behavioral anomalies: **AIS gaps** (transmitter off during slick creation), speed drops, loitering, course deviation
  - vessel-type prior (tankers > small craft)
- Output: ranked suspects with per-vessel confidence + human-readable reasoning.

### Module 4 — Visual Interface (`/frontend`, `/api`)
- Backend: FastAPI. Frontend: React + Leaflet/Mapbox.
- Layers: SAR image, spill polygon, drift animation (time slider), origin heatmap, AIS tracks colored by suspicion score, suspect table.

## Tech stack
Python 3.11+, PyTorch, rasterio / ESA SNAP (snappy) for SAR preprocessing, OpenDrift, GeoPandas, MovingPandas (AIS trajectories), FastAPI, React + Leaflet.

## Repo layout (target)
```
/detection      SAR preprocessing, segmentation model, characterization
/drift          OpenDrift wrappers, forcing-data loaders, hindcast/forecast
/attribution    AIS ingest, filtering, scoring engine
/api            FastAPI service tying modules together
/frontend       React + Leaflet map UI
/data           datasets (gitignored; see data/README for download links)
/notebooks      exploration / evaluation
```

## Build order (hackathon priority)
1. **Detection** on the Zenodo dataset — highest value, most demoable. Get it working first.
2. **Drift** with one region's canned CMEMS/ERA5 sample.
3. **Attribution** with synthetic AIS around a known documented spill.
4. **UI** last — stitch the outputs onto the map.

## Conventions for Claude Code
- Ask before adding a heavy new dependency; prefer the stack above.
- Keep the attribution scoring **transparent and inspectable** — every score must be explainable in plain language. No opaque end-to-end "guilt classifier".
- Never present age/origin as exact; always carry and surface uncertainty.
- Large data files stay out of git; document download steps in `data/README.md`.
- Write small, testable functions; add a quick eval/notebook when touching the model.
- Datasets:
  - AIS format & samples: https://marinecadastre.gov/accessais/
  - SAR oil spill data: Zenodo Sentinel-1 SAR Oil Spill Dataset