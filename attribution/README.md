# Module 3 — AIS vessel attribution

Given a drift origin estimate, rank the vessels that could have produced the
slick — and say why, in plain language.

```bash
python -m pytest attribution/tests -q      # 14 tests
```

> **Fixed this session:** `ais.py` converted parsed timestamps with
> `ts.astype("int64") / 1e9`, which assumes nanosecond resolution. Since
> pandas 3.0, `pd.to_datetime` commonly returns `datetime64[us]` (microsecond)
> Series, and `.astype("int64")` on those returns a *microsecond* count, not
> nanoseconds — so every timestamp was understated by 1000x, collapsing a full
> day of AIS into about 90 seconds of track. This affected **every** real-schema
> AIS ingest (`synth_ais_day`'s own output included, since it round-trips
> through ISO strings), silently breaking the candidate filter, the full
> synthetic benchmark, and `validation/real_cases.py`. Sample impact before the
> fix: `make_scenario` returned `None` for every seed tried; 7 of 64 tests
> failed outright. Fixed by normalising through `datetime64[ns]` first
> (`_to_unix_seconds`) regardless of what resolution pandas chose. A second,
> related pandas-3.0 issue was fixed alongside it: `.astype(str)` on pandas'
> native `"str"` dtype represents a missing value as a bare Python `float`
> (`nan`), not the string `"nan"`, so `build_tracks`' vessel-name dedup crashed
> on `.lower()` for any track with a missing name -- exactly what real
> MarineCadastre data has.

```python
from drift.hindcast import hindcast_origin
from attribution.pipeline import attribute

est = hindcast_origin(slick_lat, slick_lon, t_detect, age_h=10)
result = attribute("ais.csv", origin=est)
print(result.explain())
```

## Pipeline

1. **Spatiotemporal query** — AIS in a padded bbox over the origin's time window
   plus a lookback, filtered during chunked ingest so a multi-million-row daily
   file never lands in memory whole.
2. **Trajectory reconstruction** — per-MMSI dedup, sequential teleport removal,
   interpolation, AIS-gap preservation.
3. **Candidate filter** — keep vessels with at least one *raw* fix near the
   origin cloud. Testing raw fixes rather than interpolated ones means no vessel
   is made a suspect by a line drawn across a gap it was nowhere near.
4. **Features** — 16.
5. **Scoring** — transparent weighted log-odds, with evidence.

## The 16 features

| group | features |
|---|---|
| proximity | `prox_score`, `min_dist_km`, `mean_dist_km`, `dwell_frac` |
| temporal | `time_gap_min` |
| behavioural | `slow_steaming`, `loiter_score`, `ais_gap_max_min`, `gap_over_origin`, `dark_frac` |
| trajectory | `course_align`, `cpa_sog_kn` |
| priors | `vtype_prior`, `size_score` |
| **infrastructure** | **`platform_score`** |
| **drift-coupled** | **`track_match`** |

Two carry most of the domain reasoning. **`gap_over_origin`** fires when a vessel
went dark in a window covering the origin in *space and time* — a discharge
hiding behind a switched-off transponder. **`course_align`** uses `cos²θ` against
the slick's long axis, because a slick axis is undirected: 070 and 250 describe
the same axis.

**`track_match`** is what coupling to Module 2 buys. Because a moving vessel lays
a *line*, the hindcast returns a discharge track; this feature scores how closely
a vessel followed that path *in step with it*. A vessel that merely crossed the
area once scores poorly. Its tolerance is set from the hindcast's own uncertainty
rather than fixed — with a 17 km cloud, demanding a 5 km match would drive every
candidate to zero and make the feature useless. (It is computed by
`pipeline.track_match_score` and now actually reaches the scorer — see the "fixed
this session" note below; before that it was displayed to the UI but had no
weight in either scorer.)

**`platform_score`** is exculpatory, added after a real-data validation
(`data/validation/CERULEAN_VALIDATION.md`) surfaced a recurring false suspect
class: offshore production platforms (spar/TLP units) broadcast AIS like any
vessel, sit exactly at a slick's origin for the entire window, and beat every
transiting ship on proximity and dwell alone. It scores how much a broadcaster's
*entire* observed track looks stationary (small footprint, near-zero speed) and
carries a negative weight (`-2.2`) — a genuine ship was under way at some point in
its track; a platform never was. Computed over the whole track rather than just
the origin window so a ship that happened to loiter near the origin during the
window (the case `loiter_score` exists to catch) is not penalised for it.

## Scoring policy

The **transparent weighted score is always the primary answer**, and every
suspect carries plain-language evidence sorted by contribution. A LightGBM
LambdaMART re-ranker may be enabled alongside it; when it disagrees, both
positions are shown. It never replaces the explanation and never appears without
it.

This is deliberate. The spec rules out an opaque end-to-end guilt classifier, and
a score nobody can interrogate is useless to someone who has to act on it. A
missing or stale model degrades to the transparent score rather than taking the
whole ranking down.

## Accuracy

Full chain — synthetic slick → real drift ensemble → attribution:

    python notebooks/eval_attribution.py --n 20

| metric | value | |
|---|---|---|
| Top-1 | 45.0% | 20/20 scenarios usable |
| Recall@3 | 75.0% | |
| MRR | 0.631 | |
| median rank | 2.0 | vs. 13.1 by chance (candidate pool size + 1) / 2 |

The origin the attribution stage receives is deliberately imperfect, which is
the point: attribution trained or measured against a perfect origin would prove
nothing. (Superseded numbers previously quoted here — 8 scenarios, Top-1 38%,
Recall@3 88% — predate this session's timestamp-bug fix and the two new
features below, and could not have been reproduced against the code as it
stood; regenerate with the command above rather than trusting either figure.)

## Real-data validation

Two complementary checks against real Sentinel-1 detections and real
MarineCadastre AIS (not the synthetic simulator):

- **`validation/real_cases.py`** — one documented incident with a known
  responsible vessel (NYK DELPHINUS, ranked **#1 of 19**). Easy (the vessel
  caught fire and stayed put) but real ground truth.
- **`validation/cerulean_benchmark.py`** — 25 real Cerulean-detected slicks
  across 10 days on the US shelf, compared to an independent AIS-based system
  rather than to our own simulator. See
  `data/validation/CERULEAN_VALIDATION.md` for the numbers and what they do
  and do not show (coverage — whether the AIS feed even contains the vessel
  Cerulean names — turns out to be the dominant limiter, not the scorer).

## Limits, stated plainly

- **AIS is synthetic**, generated for the scene's location. MarineCadastre covers
  US waters only, so a global demo cannot use real AIS; the problem statement
  permits synthetic data for the region.
- `track_match` is weak under analytic forcing because the back-tracked path is
  itself displaced. Real CMEMS currents should improve it materially.
- The Cerulean benchmark's median rank (35, when the vessel is seen at all) is
  worse than the synthetic figure above -- real candidate pools run into the
  hundreds near busy coastlines, and the origin cloud there is a simple
  Gaussian around the slick centroid, not a real hindcast track. Coupling
  `cerulean_benchmark.py` to a real `hindcast_origin()` call (it currently uses
  `OriginHypothesis.from_point`) is the next lever to pull.
- No MMSI-spoofing or identity-switch detection yet.
- **Attribution is correlation, not proof.** This ranks who *could* have done it
  and states the evidence. It does not establish that anyone did.
