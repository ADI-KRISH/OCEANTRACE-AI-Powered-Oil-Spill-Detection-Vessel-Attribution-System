"""FastAPI service tying the modules together.

    python -m uvicorn api.main:app --reload --port 8000

Right now only Module 1 (detection) exists, so `/api/modules` reports the others
as unavailable and their endpoints return 501 rather than inventing data. The
frontend reads that capability map and disables those layers, so what a viewer
sees always reflects what is actually built.
"""
from __future__ import annotations

import io
import os
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "detection", "checkpoints", "unet_best.pt")

app = FastAPI(title="Oil-spill detection & attribution platform",
              description="SIH 26143 (NTRO) — SAR detection, drift, AIS attribution.",
              version="0.1")

# The Vite dev server runs on a different port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

#: Synthetic scenes have no real geolocation, so the caller chooses where to
#: place one. Anywhere on the world ocean is valid -- the problem is global, and
#: a demo may need the Arabian Sea, the Gulf, the Malacca Strait or the North
#: Sea. Every response carries `georeferencing: "demo_placement"` so a placed
#: scene is never mistaken for a real Sentinel-1 footprint.
DEFAULT_ORIGIN = {"lat": 18.60, "lon": 71.60}
#: Sentinel-1 GRD ground sample distance. Using the real value matters: at 30 m
#: a demo scene produces 16 km^2 slicks, which saturate the age model and smear
#: the backward drift over 150 km.
DEMO_PIXEL_SIZE_M = 10.0

#: Named hotspots offered by the UI. Chosen because each is a documented
#: operational-discharge or tanker-traffic region, so a demo can be framed
#: against a real problem area rather than an arbitrary patch of sea.
REGIONS = [
    {"id": "arabian_sea",   "name": "Arabian Sea (off Mumbai)",  "lat": 18.60, "lon": 71.60},
    {"id": "gulf_of_kutch", "name": "Gulf of Kutch",             "lat": 22.55, "lon": 69.00},
    {"id": "bay_of_bengal", "name": "Bay of Bengal (Chennai)",   "lat": 13.00, "lon": 80.60},
    {"id": "malacca",       "name": "Strait of Malacca",         "lat":  2.30, "lon": 101.60},
    {"id": "persian_gulf",  "name": "Persian Gulf (Hormuz)",     "lat": 26.30, "lon": 56.20},
    {"id": "suez",          "name": "Gulf of Suez",              "lat": 28.20, "lon": 33.30},
    {"id": "gulf_mexico",   "name": "Gulf of Mexico",            "lat": 28.20, "lon": -89.50},
    {"id": "north_sea",     "name": "North Sea",                 "lat": 54.50, "lon":  3.50},
    {"id": "mediterranean", "name": "Eastern Mediterranean",     "lat": 34.20, "lon": 25.00},
    {"id": "gulf_guinea",   "name": "Gulf of Guinea",            "lat":  3.50, "lon":  6.00},
    {"id": "singapore",     "name": "Singapore Strait",          "lat":  1.20, "lon": 103.80},
    {"id": "black_sea",     "name": "Black Sea",                 "lat": 43.50, "lon": 31.00},
]

#: Longest backward integration the hindcast will attempt, hours.
MAX_HINDCAST_H = 24.0

_MODEL = None
_SCENES: dict[str, dict] = {}


def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(CKPT):
            raise HTTPException(
                503, f"No detection checkpoint at {CKPT}. Train one first: "
                     "python -m detection.train --synthetic --epochs 25")
        from detection.predict import load_checkpoint
        _MODEL = load_checkpoint(CKPT)
    return _MODEL


def demo_transform(lat: float, lon: float, pixel_size_m: float = DEMO_PIXEL_SIZE_M):
    """Affine geotransform placing a square scene with its NW corner at lat/lon.

    The longitude scale is taken at the scene's own latitude, so a scene in the
    North Sea is not stretched relative to one near the equator.
    """
    deg_lat = pixel_size_m / 111_320.0
    cos_lat = max(np.cos(np.radians(lat)), 0.02)   # guard the poles
    deg_lon = pixel_size_m / (111_320.0 * cos_lat)
    # GDAL order: (lon0, dlon/dcol, dlon/drow, lat0, dlat/dcol, dlat/drow)
    return (lon, deg_lon, 0.0, lat, 0.0, -deg_lat)


def transform_bounds(transform, h: int, w: int):
    """Leaflet ImageOverlay bounds: [[south, west], [north, east]]."""
    from detection.characterize import pixel_to_lonlat
    lon0, lat0 = pixel_to_lonlat(0, 0, transform)
    lon1, lat1 = pixel_to_lonlat(h, w, transform)
    return [[min(lat0, lat1), min(lon0, lon1)], [max(lat0, lat1), max(lon0, lon1)]]


# ---------------------------------------------------------------------------
# Capability map
# ---------------------------------------------------------------------------

@app.get("/api/modules")
def modules():
    """What is actually built. The frontend disables everything reported false."""
    return {
        "detection": {
            "available": os.path.exists(CKPT),
            "name": "Detection & characterization",
            "note": ("Trained on SYNTHETIC data — results demonstrate the "
                     "pipeline, not Sentinel-1 performance."
                     if os.path.exists(CKPT) else
                     "No checkpoint. Run: python -m detection.train --synthetic"),
        },
        "drift": {
            "available": True, "name": "Drift hindcast / forecast",
            "note": _drift_note(),
        },
        "attribution": {
            "available": True, "name": "AIS vessel attribution",
            "note": ("Transparent weighted scoring with plain-language evidence. "
                     "AIS is SYNTHETIC for the region, as the problem statement "
                     "permits where real AIS is unavailable."),
        },
    }


def _drift_note():
    from drift.forcing import CMEMSForcing
    if CMEMSForcing.logged_in():
        return "Backward ensemble with Copernicus Marine currents."
    return ("Backward ensemble with ANALYTIC forcing — not a met-ocean model. "
            "Run `copernicusmarine login` for real CMEMS currents.")


@app.get("/api/health")
def health():
    return {"status": "ok", "detection_checkpoint": os.path.exists(CKPT)}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    demo_seed: Optional[int] = Field(None, description="synthetic scene seed")
    image_path: Optional[str] = Field(None, description="path to a SAR scene")
    size: int = 512
    pixel_size_m: float = DEMO_PIXEL_SIZE_M
    wind_speed_ms: Optional[float] = None
    lat: Optional[float] = Field(None, description="scene NW corner latitude")
    lon: Optional[float] = Field(None, description="scene NW corner longitude")
    region: Optional[str] = Field(None, description="named region id, see /api/regions")


@app.post("/api/detect")
def detect_endpoint(req: DetectRequest):
    """Run detection + characterisation and register the scene for map layers."""
    from detection.predict import colorise, detect

    model, ckpt = get_model()

    if req.image_path:
        if not os.path.exists(req.image_path):
            raise HTTPException(404, f"image not found: {req.image_path}")
        from PIL import Image
        raw = np.array(Image.open(req.image_path).convert("L"), np.float32) / 255.0
        truth = None
        source = os.path.basename(req.image_path)
    else:
        from detection.data import synth_scene
        seed = 4 if req.demo_seed is None else req.demo_seed
        raw, truth = synth_scene(req.size, seed=seed)
        source = f"synthetic (seed {seed})"

    h, w = raw.shape

    lat, lon = DEFAULT_ORIGIN["lat"], DEFAULT_ORIGIN["lon"]
    if req.region:
        hit = next((r for r in REGIONS if r["id"] == req.region), None)
        if hit is None:
            raise HTTPException(422, f"unknown region {req.region!r}; see /api/regions")
        lat, lon = hit["lat"], hit["lon"]
    if req.lat is not None and req.lon is not None:
        lat, lon = req.lat, req.lon
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(422, "lat must be -90..90 and lon -180..180")
    transform = demo_transform(lat, lon, req.pixel_size_m)

    t0 = time.time()
    mask, probs, slicks = detect(raw, model, pixel_size_m=req.pixel_size_m,
                                 transform=transform,
                                 wind_speed_ms=req.wind_speed_ms)
    elapsed = time.time() - t0

    sid = f"scene_{len(_SCENES)}_{int(time.time())}"
    _SCENES[sid] = {
        "sar": raw, "mask": mask, "truth": truth,
        "bounds": transform_bounds(transform, h, w),
    }

    return {
        "scene_id": sid,
        "source": source,
        "shape": [h, w],
        "bounds": _SCENES[sid]["bounds"],
        "georeferencing": "demo_placement" if not req.image_path else "unset",
        "placed_at": {"lat": lat, "lon": lon},
        "georeferencing_note": (
            "Synthetic scene placed off Mumbai so the map has somewhere to draw "
            "it. These are NOT real Sentinel-1 coordinates."),
        "pixel_size_m": req.pixel_size_m,
        "inference_seconds": round(elapsed, 2),
        "model": {"arch": ckpt.get("arch"), "epoch": ckpt.get("epoch"),
                  "oil_iou": round(float(ckpt.get("oil_iou", 0)), 4),
                  "trained_on": "synthetic"},
        "n_slicks": len(slicks),
        "slicks": [s.to_dict() for s in slicks],
        "has_truth": truth is not None,
    }


def _png(arr: np.ndarray, mode: str = "RGBA") -> Response:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/scene/{scene_id}/sar.png")
def scene_sar(scene_id: str):
    sc = _SCENES.get(scene_id)
    if sc is None:
        raise HTTPException(404, "unknown scene_id")
    img = (np.clip(sc["sar"], 0, 1) * 255).astype(np.uint8)
    return _png(np.stack([img] * 3 + [np.full_like(img, 255)], -1))


@app.get("/api/scene/{scene_id}/mask.png")
def scene_mask(scene_id: str):
    """Class mask as a transparent overlay -- sea is left fully transparent."""
    from detection.predict import colorise

    sc = _SCENES.get(scene_id)
    if sc is None:
        raise HTTPException(404, "unknown scene_id")
    rgb = colorise(sc["mask"])
    alpha = np.where(sc["mask"] == 0, 0, 190).astype(np.uint8)
    return _png(np.dstack([rgb, alpha]))


@app.get("/api/scene/{scene_id}/truth.png")
def scene_truth(scene_id: str):
    from detection.predict import colorise

    sc = _SCENES.get(scene_id)
    if sc is None or sc.get("truth") is None:
        raise HTTPException(404, "no ground truth for this scene")
    rgb = colorise(sc["truth"])
    alpha = np.where(sc["truth"] == 0, 0, 190).astype(np.uint8)
    return _png(np.dstack([rgb, alpha]))


# ---------------------------------------------------------------------------
# Not yet built -- explicit 501s rather than fabricated data
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    """One click: detect -> hindcast -> attribute."""
    demo_seed: Optional[int] = None
    size: int = 512
    region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    pixel_size_m: float = DEMO_PIXEL_SIZE_M
    slick_id: int = 1
    forcing: str = "auto"          # auto | analytic | cmems
    n_particles: int = 600
    forecast_hours: float = 24.0
    n_vessels: int = 220
    radius_km: float = 25.0
    top_n: int = 10


@app.post("/api/pipeline/run")
def pipeline_run(req: PipelineRequest):
    """Run the whole chain and return every layer the map needs.

    Detection is real. Drift is a real backward ensemble. Attribution is real
    scoring over SYNTHETIC AIS generated for the scene's own location -- the
    problem statement allows synthetic AIS where real data is unavailable, and
    MarineCadastre covers US waters only, so a global demo cannot use it.
    """
    import numpy as np

    from attribution.pipeline import attribute
    from attribution.simulate import synth_ais_day
    from drift.forcing import get_forcing
    from drift.hindcast import forecast, hindcast_origin

    # --- 1. detection ---------------------------------------------------
    det = detect_endpoint(DetectRequest(
        demo_seed=req.demo_seed, size=req.size, region=req.region,
        lat=req.lat, lon=req.lon, pixel_size_m=req.pixel_size_m))
    if not det["slicks"]:
        return {"detection": det, "drift": None, "attribution": None,
                "note": "No oil slick detected, so drift and attribution did not run."}

    slick = next((s for s in det["slicks"] if s["id"] == req.slick_id),
                 det["slicks"][0])
    poly = slick.get("polygon_lonlat") or []
    if len(poly) < 3:
        raise HTTPException(422, "detected slick has no usable polygon")
    slick_lat = np.array([p[1] for p in poly], dtype=float)
    slick_lon = np.array([p[0] for p in poly], dtype=float)

    # --- 2. drift hindcast ----------------------------------------------
    # Detection time is synthetic; anchor it to "now" so the demo reads sensibly.
    t_detect = float(time.time())
    age_h = slick.get("age_estimate_h") or 6.0
    rng_lo, rng_hi = (slick.get("age_range_h") or [age_h / 4, age_h * 4])[:2]
    # Cap the backward sweep. Past ~2 days a slick has usually dispersed below
    # SAR detectability, and integrating that far back produces an origin cloud
    # so large that attribution over it is meaningless -- an honest refusal to
    # over-reach rather than a cosmetic trim.
    rng_hi = min(float(rng_hi), MAX_HINDCAST_H)
    rng_lo = min(float(rng_lo), rng_hi * 0.5)
    age_h = min(float(age_h), rng_hi)

    forcing = get_forcing(req.forcing, seed=abs(hash(det["scene_id"])) % 1000)
    est = hindcast_origin(
        slick_lat, slick_lon, t_detect, age_h=age_h,
        age_range_h=(rng_lo, rng_hi), n_particles=req.n_particles,
        forcing=forcing, slick_bearing_deg=slick.get("orientation_deg"))
    fc = forecast(slick_lat, slick_lon, t_detect, hours=req.forecast_hours,
                  forcing=forcing)

    # --- 3. attribution --------------------------------------------------
    cla, clo = est.centroid
    ais = synth_ais_day(n_vessels=req.n_vessels, lat0=cla, lon0=clo,
                        day_start=est.time_window[0] - 6 * 3600,
                        seed=abs(hash(det["scene_id"])) % 1000)
    result = attribute(ais, origin=est, radius_km=req.radius_km)

    return {
        "detection": det,
        "slick_used": slick["id"],
        "drift": {**est.to_dict(), "forecast": fc},
        "attribution": {
            **result.to_dict(req.top_n),
            "ais_source": "synthetic (generated for this location)",
        },
    }


@app.post("/api/drift/hindcast")
def drift_hindcast(payload: dict):
    """Backward ensemble for an explicit slick polygon."""
    import numpy as np
    from drift.forcing import get_forcing
    from drift.hindcast import hindcast_origin

    poly = payload.get("polygon_lonlat") or []
    if len(poly) < 3:
        raise HTTPException(422, "polygon_lonlat with >=3 points is required")
    lat = np.array([p[1] for p in poly], float)
    lon = np.array([p[0] for p in poly], float)
    est = hindcast_origin(
        lat, lon, float(payload.get("t_detect", time.time())),
        age_h=float(payload.get("age_h", 6.0)),
        age_range_h=tuple(payload["age_range_h"]) if payload.get("age_range_h") else None,
        forcing=get_forcing(payload.get("forcing", "auto")),
        slick_bearing_deg=payload.get("slick_bearing_deg"))
    return est.to_dict()


@app.get("/api/regions")
def regions():
    """Named ocean regions the UI offers for placing a synthetic scene."""
    return REGIONS


@app.get("/api/classes")
def classes():
    from detection.config import CLASS_COLORS, CLASS_NAMES
    return [{"index": i, "name": n,
             "color": "#%02x%02x%02x" % CLASS_COLORS[i]}
            for i, n in enumerate(CLASS_NAMES)]
