# OCEANTRACE — AI-Powered Oil-Spill Detection & Vessel Attribution System

**SIH 26143 · NTRO · Disaster Management**

NTRO / Disaster Management. Detect oil spills in Sentinel-1 SAR, hindcast their
origin, attribute them to a vessel via AIS, and show it all on a map.

See [`.claude/claude.md`](.claude/claude.md) for the spec.

## Run it

```bash
python -m uvicorn api.main:app --reload --port 8000    # terminal 1
cd frontend && npm install && npm run dev              # terminal 2
```

Open <http://localhost:5173> and press **Run full pipeline**.
Deep links work: `?seed=4&region=arabian_sea` runs a specific case on load.

No trained checkpoint yet?

```bash
python -m detection.train --synthetic --epochs 30      # ~11 min on an RTX 4050
```

Tests: `python -m pytest detection/tests drift/tests attribution/tests -q` (64)

## All five modules are built

| # | Module | Path | What it does |
|---|---|---|---|
| 1 | Detection & characterization | `/detection` | U-Net / DeepLabv3+ / MobileNet, 5-class SAR segmentation → slick polygons, orientation, area, age |
| 2 | Drift hindcast / forecast | `/drift` | Backward ensemble → origin heatmap, discharge **track**, time window; forward forecast |
| 3 | AIS vessel attribution | `/attribution` | Traffic filtering, 14 features + track matching, transparent scoring with evidence |
| 4 | API | `/api` | FastAPI; `/api/pipeline/run` does the whole chain in one call |
| 5 | Frontend | `/frontend` | React + Leaflet; all layers, timeline scrubber, suspect table |

## The end-to-end story

1. **Detect** — segment a SAR scene into sea / oil / look-alike / ship / land,
   trace slick polygons and measure each one.
2. **Characterise** — area, long-axis orientation, aspect, contrast, solidity,
   and a rough age *interval*.
3. **Hindcast** — run a backward ensemble across a **range** of ages, producing an
   origin probability heatmap, a discharge track and a time window.
4. **Attribute** — filter AIS traffic, score every candidate vessel on proximity,
   behaviour, trajectory shape and track match, and rank them with evidence.
5. **Show** — every stage on one map, with a timeline for the forward forecast.

## Three design decisions worth defending

**The origin is a heatmap and a track, never a point.** Backward advection
reverses cleanly; diffusion does not — you cannot un-mix — so a backward run
legitimately produces a spreading cloud. And because a slick laid by a moving
vessel back-tracks to a **line**, `/drift` returns a discharge *track*. That is
what makes attribution strong: matching a vessel's AIS against a curve is far
more discriminating than proximity to a point.

**Age is swept, not assumed.** Age-from-area inverts as `t ~ r⁴`, so a 20% error
in extent is a factor-of-two error in age — it is the dominant error term, ahead
of currents. The ensemble therefore samples a *range* of ages log-uniformly and
reports the window as a distribution.

**The transparent score is always the primary answer.** Every suspect carries
plain-language evidence. A learned re-ranker can run alongside it and is shown
side by side when it disagrees, but it never replaces the explanation and never
appears without it. The spec rules out an opaque guilt classifier; this keeps the
accuracy without losing the audit trail.

## Honest state of the numbers

**Everything is trained and driven by synthetic data.**

- **SAR**: the Krestenitis Zenodo dataset needs a manual request (see
  [`data/README.md`](data/README.md)). Current detection numbers describe the
  generator, not Sentinel-1.
- **Drift**: forcing is an **analytic field, not a met-ocean model**. Run
  `copernicusmarine login` and `/drift` switches to real CMEMS currents
  automatically. The UI says which is in use, and the API returns
  `forcing.realistic: false` for the analytic one.
- **AIS**: synthetic, generated for whatever location the scene is placed at.
  The problem statement permits this where real AIS is unavailable, and
  MarineCadastre covers US waters only, so a global demo cannot use it.
- **Placement**: demo scenes can be put anywhere on the world ocean (12 presets
  or any lat/lon). They are labelled `georeferencing: "demo_placement"` and are
  not real Sentinel-1 coordinates.

### Measured

Detection (synthetic, U-Net, 30 epochs): oil IoU **0.941**, mIoU **0.791**,
look-alike **0.855**; false-alarm rate 5.3%. Ships at instance level:
precision 0.56 / recall 0.62 / **F1 0.59** — a supporting cue, not evidence.

Attribution over the full chain (8 scenarios, real drift ensemble in the loop):
**Top-1 38%, Recall@3 88%, median rank 2**.

### Validated on a real incident

`validation/real_cases.py` runs the attribution stage against a **real documented
spill with a known responsible vessel**, using **real MarineCadastre AIS**:

| case | result |
|---|---|
| M/V NYK DELPHINUS fire, offshore Monterey Bay, 2021-05-14 | ranked **#1 of 19** candidates, 45.2% attribution |

The system was given only the reported position and a ±3 h window — the time was
assumed, not fitted — and had to pick the right ship out of 19 candidates
including four other cargo vessels and a tanker.

```bash
python -m validation.real_cases --case nyk_delphinus
```

**What this proves and what it does not.** It validates Module 3 given a
reasonable origin. It does *not* validate Module 1 on real SAR (the detector is
synthetic-trained) or Module 2 against a real drift, since the origin here is the
reported position rather than a hindcast from a satellite detection. And this
case is an **easy** one: the vessel caught fire and stayed on scene, so it sits in
the search radius throughout. An operational discharge from a transiting ship is
substantially harder. One case is a demonstration, not a benchmark.

`data/validation/demo_candidates.csv` holds **201 more** real US incidents with a
named vessel in the Sentinel-1 era, ready to run the same way.

### Known weaknesses

- Ship detection is mediocre; needs a small-object head to improve further.
- Land is unreliable and should come from a GSHHG/OSM coastline mask, not the
  segmenter.
- `track_match` is weak under analytic forcing, because the back-tracked path is
  displaced from the truth. Real currents should improve it materially.
- Attribution is **correlation, not proof**. It ranks who *could* have done it
  and says why.

## Next

1. Zenodo dataset → retrain → replace every detection number.
2. `copernicusmarine login` → real currents → re-measure attribution.
3. Ship small-object head; coastline mask for land.
