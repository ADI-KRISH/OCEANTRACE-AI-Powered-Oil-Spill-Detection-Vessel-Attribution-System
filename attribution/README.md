# Module 3 — AIS vessel attribution

Given a drift origin estimate, rank the vessels that could have produced the
slick — and say why, in plain language.

```bash
python -m pytest attribution/tests -q      # 14 tests
```

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
4. **Features** — 14, plus `track_match`.
5. **Scoring** — transparent weighted log-odds, with evidence.

## The 14 features, plus one

| group | features |
|---|---|
| proximity | `prox_score`, `min_dist_km`, `mean_dist_km`, `dwell_frac` |
| temporal | `time_gap_min` |
| behavioural | `slow_steaming`, `loiter_score`, `ais_gap_max_min`, `gap_over_origin`, `dark_frac` |
| trajectory | `course_align`, `cpa_sog_kn` |
| priors | `vtype_prior`, `size_score` |
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
candidate to zero and make the feature useless.

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

Full chain — synthetic slick → real drift ensemble → attribution, 8 scenarios:

| metric | |
|---|---|
| Top-1 | 38% |
| Recall@3 | 88% |
| median rank | 2 |

Candidate pools run 7–43 vessels. The origin the attribution stage receives is
deliberately imperfect, which is the point: attribution trained or measured
against a perfect origin would prove nothing.

## Limits, stated plainly

- **AIS is synthetic**, generated for the scene's location. MarineCadastre covers
  US waters only, so a global demo cannot use real AIS; the problem statement
  permits synthetic data for the region.
- `track_match` is weak under analytic forcing because the back-tracked path is
  itself displaced. Real CMEMS currents should improve it materially.
- No MMSI-spoofing or identity-switch detection yet.
- **Attribution is correlation, not proof.** This ranks who *could* have done it
  and states the evidence. It does not establish that anyone did.
