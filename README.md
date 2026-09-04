# SIH 26143 — Oil-spill detection & vessel attribution

NTRO / Disaster Management. Detect oil spills in Sentinel-1 SAR, hindcast their
origin, attribute them to a vessel via AIS, and show it all on a map.

See [`.claude/claude.md`](.claude/claude.md) for the full spec and
[`plan.md`](plan.md) for the attribution-stage design notes.

## Run it

Two processes. Backend first:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Then the frontend:

```bash
cd frontend
npm install          # first time only
npm run dev          # -> http://localhost:5173
```

Open <http://localhost:5173>, pick a scene seed, hit **Detect slicks**.

If you have no trained checkpoint yet:

```bash
python -m detection.train --synthetic --epochs 25    # ~11 min on an RTX 4050
```

Tests: `python -m pytest detection/tests -q`

## Module status

| # | Module | Path | Status |
|---|---|---|---|
| 1 | Detection & characterization | `/detection` | **Built** — 31 tests |
| 2 | Drift hindcast/forecast | `/drift` | Not built — **OpenDrift 1.14.11 installed**, needs CMEMS login |
| 3 | AIS vessel attribution | `/attribution` | Not built — needs an origin estimate from Module 2 |
| 4 | API | `/api` | **Built** — serves Module 1; returns 501 for 2 and 3 |
| 5 | Frontend | `/frontend` | **Built** — React + Leaflet |

The UI reads `/api/modules` and **disables the layers whose module does not
exist**, saying why on each one. Nothing in the interface is mocked: an unbuilt
module shows as unbuilt rather than as an empty result, so a demo never implies
capability that is not there.

## What works end to end today

SAR scene → segmentation → slick polygons + characterization → map, with:

- SAR scene and predicted class mask as georeferenced overlays
- Slick polygons, clickable, with per-slick measurements
- Ground-truth overlay for synthetic scenes (so you can see the model's errors)
- Long-axis orientation, area, aspect, contrast, solidity per slick
- Age shown as an **interval with low confidence**, never a bare number

Disabled, because their modules do not exist: origin heatmap, drift particles,
timeline scrubber, AIS tracks, suspect table.

## Honest state of the numbers

**Everything is trained on synthetic SAR.** The Krestenitis Zenodo dataset needs
a manual request (see [`data/README.md`](data/README.md)). Current results
(oil IoU 0.941, mIoU 0.791, ship F1 0.59) describe the generator, not Sentinel-1 — they show
the pipeline converges and the metrics are wired correctly, nothing more.

Demo scenes can be placed **anywhere on the world ocean** — 12 preset regions
(Arabian Sea, Gulf of Kutch, Malacca, Hormuz, Suez, Gulf of Mexico, North Sea,
Mediterranean, Gulf of Guinea, Singapore, Black Sea, Bay of Bengal) or any
lat/lon you type.
The API labels this `georeferencing: "demo_placement"` and the UI shows a banner.
They are not real Sentinel-1 coordinates.

Two known model failures, documented in [`detection/README.md`](detection/README.md):
ship detection is mediocre (F1 0.59 at instance level — a supporting cue, not
evidence), and land is unreliable and should come from a coastline mask rather
than the segmenter.

## Next

1. Get the Zenodo dataset, retrain, replace every number above.
2. Module 2 (drift) with OpenDrift — unblocks the origin heatmap and timeline.
3. Module 3 (attribution) — unblocks AIS tracks and the suspect table.
