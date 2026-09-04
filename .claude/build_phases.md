# BUILD_PHASES.md — Claude Code build order

Full platform: **live-ish Sentinel-1 SAR → MobileNetV3 oil detection → spill time-window
→ existing AIS attribution (PLAN.md) → 3D globe UI with ship routes, spill pins, suspect
rankings.**

> Rule for Claude Code: do **ONE phase per session**. Finish, test, commit, then move on.
> Do not start a later phase until the earlier one runs. Keep the existing
> `oilspill_attribution/` module and its `attribute()` contract intact.

---

## Phase 0 — Skeleton & contracts (do first, tiny)
- Create repo layout:
  ```
  /ingest      Sentinel fetch + tiling
  /detection   MobileNetV3 inference wrapper + time-window estimator
  /attribution -> existing oilspill_attribution/ (DO NOT rewrite)
  /api         FastAPI service
  /frontend    React + globe UI
  /data        gitignored
  ```
- Define the data objects that pass between stages (Pydantic models in `/api/schemas.py`):
  `SarScene`, `Detection{has_oil, mask_geojson, bbox, scene_time}`,
  `SpillWindow{t_start, t_end, centroid_lat, centroid_lon}`,
  then reuse `OriginHypothesis` + `attribute()` from PLAN.md.
- No logic yet — just stubs that return fake data, so the whole chain runs end-to-end.

## Phase 1 — Sentinel-1 ingest (`/ingest`)
- Use **Copernicus Data Space Ecosystem** (free account → OAuth2 token).
- Two functions:
  - `search_latest(bbox, since_days)` → newest S1 GRD product over the AOI via the
    OData catalogue (`catalogue.dataspace.copernicus.eu/odata/v1`).
  - `fetch_tile(product_id, bbox)` → pull just the AOI subset (Sentinel Hub Process API
    with an evalscript) as a GeoTIFF/PNG. This avoids downloading whole 100km granules.
- Store token in `.env` (`CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`). Never hardcode.
- IMPORTANT: "live" = latest available pass (S1 revisit ~1–3 days), NOT a real-time stream.
  Label it "latest scene" in the UI.
- Deliver a CLI: `python -m ingest.cli --bbox ... --since 5` that saves a scene to `/data`.

## Phase 2 — MobileNetV3 detection wrapper (`/detection`)
- I already HAVE the trained MobileNetV3 model — load it, do not retrain.
- `detect(scene_path) -> Detection`:
  - preprocess SAR tile to the model's expected input (document the transform).
  - run inference → oil / no-oil; if the model is segmentation-capable, emit a mask →
    vectorize to `mask_geojson` (use rasterio/shapely). If it's classification-only,
    tile the scene into a grid, classify each tile, and merge positive tiles into a
    coarse polygon.
  - compute characterisation: area_km2, centroid, long-axis bearing (feeds
    `slick_bearing_deg` in attribution).
- Confirm with me: model file path, input size, normalization, and whether it outputs a
  class or a mask — before writing preprocessing.

## Phase 3 — Spill time-window estimator (`/detection/window.py`)
- We know when the scene was captured (`scene_time`). The spill happened at or before that.
- Estimate `SpillWindow`:
  - upper bound = scene_time.
  - lower bound = scene_time − age_max (default 48h; expose as a param).
  - if two consecutive passes exist (oil absent in the earlier, present in the later),
    tighten the window to between those two pass times — this is the strong case; support it.
- Output feeds the AIS query window in `attribute()`. Keep uncertainty explicit; never
  output a single exact instant.

## Phase 4 — Wire the chain into FastAPI (`/api`)
- Endpoint `POST /analyze {bbox, since_days}`:
  ingest → detect → if has_oil: build OriginHypothesis (Monte-Carlo around centroid +
  window) → call existing `attribute()` → return detection + ranked suspects + evidence
  as JSON (+ the folium/geojson layers the UI needs).
- Endpoint `GET /scenes`, `GET /result/{id}` for the UI to poll.
- Long jobs run async (background task + job id); the UI polls. This avoids the timeouts.
- Return GeoJSON for: spill polygon, AIS tracks (colored by score), origin cloud.

## Phase 5 — Globe UI (`/frontend`) — the flashy part, do LAST
- React + a WebGL globe: use **CesiumJS** (best for real geospatial globe + camera fly-to)
  or react-globe.gl if you want lighter. Prefer Cesium for accurate lat/lon + terrain.
- Layers on the globe:
  - spill polygon pinned at its coordinates, pulsing marker, fly-to on detection.
  - AIS ship routes as animated polylines; color = suspicion score (green→red).
  - origin uncertainty cloud as a translucent heat blob.
  - click a ship → side panel with rank, attribution_pct, and the plain-language
    evidence list from the attribution module.
- Add: region picker (draw a bbox), "Analyze latest scene" button, timeline slider
  scrubbing the spill window, and a ranked suspect table synced with the globe.
- Keep it a thin client: it only renders JSON from `/api`. No ML in the browser.

---

## Guardrails for every phase
- Don't break `attribute()` or the PLAN.md metrics.
- Ask me before adding heavy deps or when the model I/O is unclear.
- Secrets in `.env`, data in `/data` (gitignored).
- After each phase: a runnable command + a 1-line "how to test" note.
- Prefer small, testable functions over one mega-script.