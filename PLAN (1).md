# SIH 26143 — Oil-spill detection + vessel attribution — PLAN

**Problem (NTRO):** from satellite imagery (SAR/EO) + AIS, (a) detect & characterise
an oil slick, (b) hindcast its origin and forecast its drift using met-ocean data,
(c) attribute it to a vessel using historic AIS — filter irrelevant traffic, score
suspects by proximity / trajectory / behavioural anomaly. Plus a visual interface.

## 1. System decomposition

```
        ┌──────────────────────┐   ┌───────────────────────┐   ┌──────────────────────────┐
 SAR/EO │ A. Slick detection   │   │ B. Drift hindcast /   │   │ C. AIS attribution       │
 scene ─▶  + characterisation  │──▶│    forecast (met-ocean)│──▶│    (THIS REPO / my part) │──▶ ranked
        │  polygon, area, age, │   │  origin (lat,lon,t)   │   │  filter traffic + score  │    suspects
        │  linearity, centerln │   │  + uncertainty cloud  │   │  suspects + evidence     │    + map
        └──────────────────────┘   └───────────────────────┘   └──────────────────────────┘
```

Interface between B and C is the single object `OriginHypothesis`
(`oilspill_attribution/src/features.py`): Monte-Carlo samples of `(lat, lon, time)`
for the discharge, plus optional `slick_bearing_deg` (slick long-axis azimuth from A).
This keeps my module runnable independently of the detection/drift work.

## 2. My part — AIS vessel attribution  (`oilspill_attribution/`)

### Framing
No public "spill-polygon → guilty-MMSI" dataset exists at scale, so attribution is
**spatio-temporal correlation + anomaly scoring**, with two interchangeable scorers:

| scorer | needs labels | role |
|---|---|---|
| `baseline` — transparent weighted log-odds | no | always available, fully explainable |
| `learned ranker` — LightGBM LambdaMART | yes (from simulator) | higher accuracy |

### Pipeline (`src/pipeline.py :: attribute(csv, lat, lon, time)`)
1. **Spatiotemporal query** — AIS rows in a padded bbox over
   `[t − age_max − lookback, t + 2h]`.
2. **Trajectory reconstruction** (`src/ais.py`) — per-MMSI clean (dup/teleport
   removal), time-indexed interpolation, AIS-gap preservation.
3. **Candidate filter** — keep vessels with ≥1 raw fix within `radius_km` of the
   origin cloud in the window; drop the rest ("filter irrelevant traffic").
4. **Feature extraction** (`src/features.py`, 14 features):
   - proximity: `prox_score`, `min_dist_km`, `mean_dist_km`, `dwell_frac`
   - temporal: `time_gap_min`
   - behavioural anomaly: `slow_steaming` (2–9 kn), `loiter_score` (heading
     variance at low SOG), `ais_gap_max_min`, `gap_over_origin` (dark interval
     covering the origin in space *and* time)
   - trajectory shape: `course_align` (vessel COG vs slick long-axis)
   - priors: `vtype_prior` (tanker/cargo > fishing > pleasure), `size_score`
5. **Scoring** (`src/scoring.py`) — probability per vessel, normalised to an
   `attribution_pct`, with a plain-language `evidence` list per suspect.
6. **Visualisation** (`src/viz.py`) — Folium map: origin uncertainty cloud +
   colour-coded suspect tracks + evidence popups. (`outputs/demo_map.html`)

### Training data — synthetic simulator (`src/simulate.py`)
Take a real AIS day → pick a moving vessel as culprit → extrude a slick along its
track during a discharge window → drift it forward with a toy advection–diffusion
ocean model (`src/drift.py`) → hand the attribution stage an *imperfect* backward
origin estimate. Ground truth known ⇒ train + measure. Problem statement explicitly
permits synthetic data.

### Evaluation
`train_eval.py` — generate N scenarios, 70/30 split by scenario, report
**Top-1 / Recall@3 / MRR / median-rank**. Current (140 scenarios, US East Coast,
~71 candidates/scenario):

| scorer | Top-1 | Recall@3 | MRR |
|---|---|---|---|
| baseline | 42.9% | 73.8% | 0.60 |
| learned ranker | 54.8% | 76.2% | 0.67 |

## 3. Real-world validation set

Synthetic ≠ proof. Two real sources are wired in:

### 3a. NOAA IncidentNews  (`src/incidents.py`, input `incidents.csv`)
4,929 incidents → filter to oil + vessel-related with coordinates → best-effort
vessel-name extraction from the title → `outputs/real_cases.csv`
(~1,656 incidents, ~770 with a name guess, ~1,160 in the AIS era ≥2009).
Manual step: resolve vessel name → MMSI via Equasis / MarineTraffic for ~15–20
clean cases, download the matching AIS day (`tools/ais_download.py`), run the
pipeline, check the known vessel's rank.

### 3b. SkyTruth Cerulean  (`tools/fetch_cerulean.py`)
Cerulean runs a CNN over every Sentinel-1 scene and cross-correlates slicks with
AIS + infrastructure. Its OGC API gives, per slick, a **ranked source list with
MMSI, `source_type` (VESSEL/DARK/INFRA) and `source_collated_score`**
(`public.slick_plus` + `public.source_plus`). Pull a regional benchmark:
`data/cerulean/{slicks,sources}.csv`. Use Cerulean rank-1 vessel as a strong
prior label (their ranker also uses AIS, so it is not independent ground truth —
report agreement, not accuracy, against it).

## 4. Datasets in use

| data | source | coverage |
|---|---|---|
| `ais-dataset/ais-2022-10-11.csv` | NOAA MarineCadastre daily AIS | US waters, 1 day |
| `ais-dataset/data.csv` | Kaggle AIS (engineered features) | sample |
| `incidents.csv` | NOAA IncidentNews export | 1957–2026, US |
| Cerulean API | api.cerulean.skytruth.org | global Sentinel-1, 2020– |
| (detection team) Zenodo Sentinel-1 SAR oil-spill dataset | per problem statement | — |

## 5. Integration contract (for teammates)

```python
from oilspill_attribution.src.features import OriginHypothesis
from oilspill_attribution.src.pipeline import attribute

# drift team produces this:
origin = OriginHypothesis(lat=<Nx>, lon=<Nx>, t_unix=<Nx>, slick_bearing_deg=<deg>)
# attribution returns ranked suspects + evidence + a folium map
result = attribute("AIS_2024_03_11.csv", lat, lon, "2024-03-11T14:00:00",
                   slick_bearing_deg=55)
```

## 6. Status & next steps

- [x] AIS loader / trajectory reconstruction (handles both NOAA schemas)
- [x] Feature extraction (14 features) + baseline + LightGBM ranker
- [x] Synthetic simulator + train/eval harness + metrics
- [x] Folium map viz
- [x] IncidentNews curation → `real_cases.csv`
- [x] Cerulean benchmark fetcher
- [ ] Resolve 15–20 IncidentNews cases to MMSI; download AIS days; validate
- [ ] Fold Cerulean regional benchmark into `train_eval.py` as a real test split
- [ ] Replace toy drift with real currents (HYCOM/CMEMS) + wind (ERA5) — B's job
- [ ] MMSI-spoofing / identity-switch detection
- [ ] Web UI: map + timeline slider + suspect table (wrap `attribute()` in FastAPI)
- [ ] Slide deck: methodology, metrics, one real-case walkthrough
