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
| 3 | AIS vessel attribution | `/attribution` | Traffic filtering, 16 features (incl. track matching + a fixed-platform discriminator), transparent scoring with evidence |
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
- **Drift**: now uses **real Copernicus Marine surface currents** when
  `copernicusmarine login` has been run — the module picks the analysis/forecast
  product for recent dates and the multi-year reanalysis for historical ones, and
  falls back to a clearly-labelled analytic field if the download fails. The API
  reports `forcing.realistic` either way, so a viewer always knows which produced
  a given result.
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

Attribution over the full chain (`python notebooks/eval_attribution.py --n 20`,
real drift ensemble in the loop): **Top-1 45%, Recall@3 75%, MRR 0.63, median
rank 2** (vs. 13.1 by chance), 20/20 scenarios usable.

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

`data/validation/demo_candidates.csv` holds **337 more** real US incidents with a
named vessel in the AIS era (NOAA IncidentNews, curated by
`validation/incidents.py`), ready to run the same way -- each needs its vessel
name resolved to an MMSI (Equasis/MarineTraffic) before it can be added as a
`validation.real_cases` case.

### Broader real-data check: SkyTruth Cerulean

A single documented incident is a demonstration, not a benchmark. Complementing
it, `validation/cerulean_benchmark.py` compares this module's ranking to
SkyTruth Cerulean's -- an independent system that also detects slicks from
Sentinel-1 and correlates them against AIS -- across 25 real slicks over 10 days
on the US shelf:

```bash
python -m validation.cerulean_benchmark --build --n-cases 25 --max-days 12
python -m validation.cerulean_benchmark --run
```

| metric | value |
|---|---|
| Cerulean's #1 vessel present in our AIS feed | 62% |
| agree@1 (given it's present) | 27% |
| Cerulean's #1 in our top-3 | 27% |
| our #1 within Cerulean's top-5 | 33% |

Full numbers and reading notes in `data/validation/CERULEAN_VALIDATION.md`.
Cerulean's ranker also consumes AIS, so this is agreement between two
independent methods, not ground truth -- but AIS-feed coverage (whether the
vessel Cerulean names is even in MarineCadastre) turning out to be the largest
single limiter, ahead of scoring, is itself the useful finding: it says the next
lever to pull is a global AIS feed, not more feature engineering.

### Known weaknesses

- Ship detection is mediocre; needs a small-object head to improve further.
- Land is unreliable and should come from a GSHHG/OSM coastline mask, not the
  segmenter.
- `track_match` is weak under analytic forcing, because the back-tracked path is
  displaced from the truth. Real currents should improve it materially.
- Attribution is **correlation, not proof**. It ranks who *could* have done it
  and says why.
- Real candidate pools near busy coastlines run into the hundreds (see the
  Cerulean benchmark above), and `cerulean_benchmark.py` currently hands
  attribution a Gaussian cloud rather than a real `hindcast_origin()` track --
  coupling the two is the next lever, ahead of more feature engineering.

### Fixed this session

- **`attribution/ais.py`** converted timestamps with `ts.astype("int64") / 1e9`,
  which silently assumes nanosecond resolution. Since pandas 3.0, parsed
  datetimes commonly come back as `datetime64[us]`, and `.astype("int64")` then
  returns a *microsecond* count -- understating every timestamp 1000x and
  collapsing a full day of AIS into ~90 seconds of track. This broke every
  real-schema ingest, including the synthetic simulator's own output (it
  round-trips through ISO strings): `make_scenario` returned `None` for every
  seed, and 7/64 tests failed. A second pandas-3.0 issue in the same function --
  `.astype(str)` on a missing value in pandas' native string dtype returns a
  bare `float`, not `"nan"` -- crashed vessel-name dedup on any real AIS file
  with a blank name. Both fixed; 64/64 tests pass; `validation/real_cases.py`
  re-verified (still #1 of 19).
- **`track_match`** was computed and shown to the UI but never reached either
  scorer's weights -- it lived outside `FEATURE_NAMES`. Now part of the 16
  features both scorers see.
- **`platform_score`** (new) -- an exculpatory feature down-weighting
  broadcasters whose entire track looks stationary, added after the Cerulean
  benchmark below showed fixed offshore platforms winning on proximity alone.
- **`validation/ais_download.py`** hit MarineCadastre's own server directly,
  which reliably dropped connections partway through downloads this large; it
  now defaults to the faster, reliable NOAA OCM Azure mirror and falls back to
  MarineCadastre only if that fails.
- **`validation/fetch_cerulean.py`** took the *first vertex* of a slick's
  polygon as its location -- which can sit tens of km from the slick's actual
  extent -- and crashed outright on the API's flat (`f=json`) response shape.
  Now computes a real coordinate-mean centroid and handles both response shapes.
- **`validation/incidents.py`** crashed building the vessel/oil keyword filter
  whenever the `commodity` column was blank (1,093 of 4,929 rows) -- same
  missing-value-as-`float` pandas 3.0 issue as above. Fixed; regenerated
  `data/validation/demo_candidates.csv` (1,320 oil+vessel incidents, 337
  priority) from the NOAA IncidentNews export.
- Removed two stale, byte-for-byte duplicate `plan.md` / `PLAN (1).md` files
  describing an unrelated earlier repo layout (a single `oilspill_attribution/`
  package, not this repo's `/detection` `/drift` `/attribution` split) --
  `README.md` plus `.claude/claude.md` and `.claude/build_phases.md` are the
  current spec and plan.

## Next

1. Zenodo dataset → retrain → replace every detection number.
2. ~~`copernicusmarine login` → real currents~~ **done** — re-measure attribution against real drift.
3. Ship small-object head; coastline mask for land.
