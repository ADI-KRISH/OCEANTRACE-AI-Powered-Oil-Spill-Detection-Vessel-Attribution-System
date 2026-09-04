# Merge / integration notes

> **Branch note:** everything below lives on `backup/sky-fixes`, not `main`.
> `main` is your teammate's work, untouched. Open a PR from `backup/sky-fixes`
> when you want to bring this in.

## Round 2 — integrating the teammate's next two commits

After the round-1 work below was pushed, two more commits landed on `main`:
**"Real Copernicus Marine currents in the drift module"** (`drift/cmems.py` —
a real CMEMS reader behind the same forcing interface as the analytic field,
picking the forecast or reanalysis product by date, falling back cleanly if
the download fails) and **"Stage 0: MobileNetV3 pre-filter gating the
segmenter"** (`detection/screening.py` — a small classifier that gates the
expensive segmenter, tuned for recall over precision since a miss loses the
spill).

Rebased `backup/sky-fixes` onto both. Reviewed both by actually running them,
not just reading the diff:

- `drift/cmems.py` — code looks correct (bilinear interpolation, NaN-as-still-
  water for land cells, sensible dataset selection); not exercised against a
  live Copernicus account here (none configured on this machine), but the
  analytic fallback path was confirmed still working.
- `detection/screening.py` — loads and runs correctly; confirmed the
  checkpoint file (`mobilenetv3_oil_spill.pth.zip`) is a normal PyTorch
  zip-format checkpoint despite the `.zip`-looking name, not a doubly-zipped
  archive, so it loads directly.
- **Found a real bug via live testing**: `api/main.py`'s new screening step
  in `/api/pipeline/run` referenced `req.image_path`, a field that does not
  exist on `PipelineRequest` — every call with `screen=True` (the default,
  whenever the classifier's weights are present) raised `AttributeError` and
  the endpoint returned HTTP 500. This would not have shown up from reading
  the code or from the existing test suite; it only surfaced by actually
  `curl`ing the running server. Fixed by adding `image_path` to
  `PipelineRequest` and threading it through to `detect_endpoint` too, so a
  real uploaded image and the screening step agree on which image they're
  looking at (previously `pipeline_run` silently ignored `image_path` even
  when detection alone supported it). Verified live afterward: HTTP 200
  across 8 pipeline runs at different seeds/regions, screening output present
  and sensible, including one seed that correctly reports "no oil" with 0
  slicks.
- Also flagging, **not fixed**: the screening commit added `archive (3).zip`
  (66 MB — the raw Kaggle Class_0/Class_1 training images, unzipped nowhere)
  committed directly to git on `main`. Bloats `.git` to ~70 MB. Left alone
  since it's already on the shared `main` and rewriting it needs everyone to
  re-clone; worth a follow-up cleanup commit from whoever owns that commit.

All 68 tests pass after this round; `validation/real_cases.py` not re-run
this round (no attribution-affecting change), `python -m detection.train`
checkpoint from round 1 still present and used by these live tests.

---

# Round 1 — this session

Context: `part1/` (a MobileNetV3 binary oil/no-oil classifier notebook) and this
repo, OCEANTRACE (cloned from
`github.com/ADI-KRISH/OCEANTRACE-AI-Powered-Oil-Spill-Detection-Vessel-Attribution-System`),
were built by a teammate as the rest of the SIH 26143 pipeline around my own
`oilspill_attribution/` AIS-attribution module (kept as the sibling repo this
was cloned alongside; see its `PLAN.md`). The teammate's build turned out to be
a full, independently-built five-module system — detection (real U-Net/DeepLab
segmentation, not just part1's classifier), a physically real backward-drift
ensemble, an attribution engine that already incorporated most of my module's
ideas (its own `validation/fetch_cerulean.py` etc. still carries my old
`oilspill_attribution.tools.*` naming in places), a FastAPI service and a React
frontend. This was a genuine audit-and-fix pass over an already-strong
codebase, not a from-scratch build.

## What I found and fixed

**Two silent, systemic bugs**, both a pandas-3.0 behaviour change that the
existing test suite had already caught (7/64 attribution tests were failing
when I started) but that hadn't been root-caused yet:

1. `attribution/ais.py` converted timestamps via `ts.astype("int64") / 1e9`,
   assuming nanosecond resolution. Pandas 3.0 commonly parses to
   `datetime64[us]`, and `.astype("int64")` then returns microseconds — every
   timestamp was understated 1000x, collapsing a day of AIS into ~90 seconds of
   track. This broke **every** real-schema AIS ingest, including the synthetic
   simulator's own output. Fixed by normalising through `datetime64[ns]` first.
2. `.astype(str)` on pandas' native `"str"` dtype (default since 3.0)
   represents a missing value as a bare `float`, not the string `"nan"`. This
   crashed `build_tracks`' vessel-name dedup on any AIS file with a blank name
   (real MarineCadastre data, always) and separately crashed
   `validation/incidents.py`'s keyword filter on any incident with a blank
   commodity column (1,093 of 4,929 rows). Fixed in both places.

Result: 64/64 tests pass (was 57/64); `validation/real_cases.py` re-verified
end to end (NYK DELPHINUS still ranks #1 of 19 after the fix).

**A wired-but-inert feature.** `track_match` — the whole point of coupling
Module 3 to Module 2's back-tracked discharge path — was computed by
`pipeline.track_match_score` and shown to the UI, but was never added to
`FEATURE_NAMES`, so neither the transparent scorer nor the learned ranker ever
weighted it. It is now one of the 16 features both scorers actually see.

**A download reliability bug.** `validation/ais_download.py` hit
`coast.noaa.gov` directly, which reliably drops connections partway through
files this size (300 MB – 1 GB); every download attempt during this session
failed that way. Switched the default to the NOAA OCM Azure mirror
(`noaaocm.blob.core.windows.net`, zstd-compressed), which streamed every file
correctly; MarineCadastre direct is kept as an explicit fallback.

**Two Cerulean API bugs.** `fetch_cerulean.fetch_items` took the *first vertex*
of a slick polygon as its location (can sit tens of km from the slick's actual
extent) and crashed outright on the API's `f=json` response, which turns out to
be a flat list with WKT geometry, not the GeoJSON FeatureCollection the code
assumed. Fixed: real coordinate-mean centroid, both response shapes handled.

**A stale doc reference.** `README.md` claimed `data/validation/demo_candidates.csv`
existed with 201 incidents; the file didn't exist anywhere and no `incidents.csv`
was present to build it from. I had the NOAA IncidentNews export on hand from the
sibling project, copied it in, fixed the incidents.py crash above, and generated
it for real: **1,320** oil+vessel incidents with coordinates, **337** priority
(named vessel, AIS era).

**Two duplicate stale files.** `plan.md` and `PLAN (1).md` were byte-identical
copies of my `oilspill_attribution/PLAN.md`, describing that repo's layout
(`/src`, a single package), not this one's (`/detection` `/drift`
`/attribution` `/api` `/frontend`). Removed — actively misleading next to the
real spec (`.claude/claude.md`, `.claude/build_phases.md`) and this README.

## What I added

**`platform_score`** (new attribution feature, exculpatory, weight `-2.2`).
Motivated by a real finding, not a hypothetical: several offshore production
platforms (spar/TLP units, which broadcast AIS) topped the ranking in the
Cerulean benchmark below purely on proximity + dwell, because they sit exactly
at a slick's origin for the whole window. Scores how much a broadcaster's
*entire* observed track looks stationary; computed over the whole track (not
just the origin window) so a ship that genuinely loitered near the origin
during the window — the case `loiter_score` exists to reward — isn't penalised
for it.

**`validation/cerulean_benchmark.py`** (new) — broad real-data validation,
complementing the existing single-case `real_cases.py`. Pulls high-confidence
Cerulean slicks on the US shelf (dense AIS coverage; excludes the deep Gulf /
Bay of Campeche on purpose, where Cerulean has slicks but MarineCadastre has no
AIS at all — including it would just relabel "no US AIS" as "attribution
failure") whose #1 probable source is a vessel, downloads the matching AIS day,
and compares this module's ranking to Cerulean's. Run:
```bash
python -m validation.cerulean_benchmark --build --n-cases 25 --max-days 12
python -m validation.cerulean_benchmark --run
```
Result (25 cases / 10 days): Cerulean's #1 vessel is in our AIS feed 62% of the
time; when it is, agree@1 27%, top-3 27%, our own #1 lands in Cerulean's top-5
33% of the time. Full report: `data/validation/CERULEAN_VALIDATION.md`.
Coverage — whether the AIS feed contains the vessel at all — is the dominant
limiter, not the scorer; global AIS (Spire/GFW) would move this more than
further feature engineering.

**`notebooks/eval_attribution.py`** (new) — the full synthetic chain
(slick → real drift ensemble → attribution) as a runnable, reproducible script,
matching `eval_detection.py`'s convention. The README's attribution accuracy
numbers were previously hand-pasted with no way to regenerate them, and could
not have been produced by the code as it stood (the timestamp bug above would
have made `make_scenario` return `None` for every seed). Fresh numbers, 20/20
scenarios usable: **Top-1 45.0%, Recall@3 75.0%, MRR 0.631, median rank 2.0**
(vs. 13.1 by chance).

## What I deliberately left alone

- **A trained detection checkpoint.** Started
  `python -m detection.train --synthetic --epochs 8 --train-n 300 --val-n 80
  --batch-size 4` (thread-limited via `OMP_NUM_THREADS=4`, `nice -n 15`) in the
  background near the end of this session on a machine with no GPU and heavy
  concurrent load from the user's own foreground apps — check
  `detection/checkpoints/unet_best.pt` / rerun the command if it didn't finish.
  Full training (30 epochs, the README's own number) needs real time on this
  hardware; do not read a partial/short run's numbers as representative.
- **Frontend.** Structurally reviewed (`api.js`, `vite.config.js`'s dev proxy,
  `package.json`) and looks correct — did not `npm install` / run it live,
  given the same resource constraints.
- **Real Sentinel-1 / real CMEMS currents / real global AIS.** All three
  require external accounts or manual dataset requests documented in
  `data/README.md`; out of scope for this pass.
- `oilspill_attribution/` (the sibling repo) is unchanged. Its own `PLAN.md`
  now points at this repo's `data/validation/CERULEAN_VALIDATION.md` /
  `VALIDATION.md` results as the more complete version of the same real-data
  validation idea.

## Verifying this yourself

```bash
python -m pytest detection/tests drift/tests attribution/tests -q   # 64 passed
python -m validation.real_cases --case nyk_delphinus                # #1 of 19
python notebooks/eval_attribution.py --n 20                         # fresh numbers
```
