# Module 2 — Drift hindcast & forecast

Takes a detected slick polygon and returns where the oil came from (a probability
heatmap, a discharge track and a time window) and where it is going.

```bash
python -m pytest drift/tests -q        # 18 tests
```

```python
from drift.hindcast import hindcast_origin, forecast
from drift.forcing import get_forcing

est = hindcast_origin(slick_lat, slick_lon, t_detect,
                      age_h=10, age_range_h=(3, 40),
                      forcing=get_forcing("auto"))
print(est.summary())
fc = forecast(slick_lat, slick_lon, t_detect, hours=24)
```

## Why the output is not a point

Backward advection reverses cleanly. **Diffusion does not** — you cannot un-mix —
so a backward run legitimately produces a spreading probability cloud that grows
the further back you go. Reporting a single origin would be false precision.

## Why age is swept rather than assumed

Age-from-area inverts as `t ~ r⁴`. A 20% error in the measured radius becomes a
factor-of-two error in age, which makes age the **dominant** error term — ahead
of current error. So the ensemble samples ages log-uniformly across the interval
the detection stage supplies (uncertainty in age is multiplicative, so a linear
sweep would over-sample the long tail), and the time window comes out as a
distribution instead of an assumption.

## Why the origin is a track

A slick laid down by a moving vessel is a **line, not a point**. Back-tracking it
returns a *track segment*, which `/attribution` matches vessel AIS against. That
is strictly more discriminating than proximity to a single origin, and it is the
main reason the two modules are worth coupling.

## Forcing

| source | when | realistic |
|---|---|---|
| `AnalyticForcing` | offline, no account | **no** — declared in every response |
| `CMEMSCurrents` | after `copernicusmarine login` | yes |

Two Copernicus products are needed, chosen automatically from the requested date
rather than configured — getting it wrong is a silent failure, since the download
just returns nothing for the period:

| product | covers |
|---|---|
| `cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i` | recent days + forecast |
| `cmems_mod_glo_phy_my_0.083deg_P1D-m` | multi-year reanalysis (historical cases) |

**Windage is not applied with CMEMS unless a wind is supplied.** The model's
surface current already carries wind-driven Ekman transport; the extra ~3% a
slick picks up from direct wind stress is separate, and inventing a wind to add
it would be worse than omitting it.

`get_forcing("auto")` prefers CMEMS and falls back to analytic rather than
failing, because an offline demo must never hard-error. The analytic field is a
divergence-free stream function (so particles do not pile up artificially) plus a
slowly veering wind. It is **not** a met-ocean model and says so in
`forcing.realistic` and in the UI.

Wind contributes at `WIND_DRIFT_FACTOR = 0.03` — the standard operational value,
and over a long hindcast the largest single error source: a 5 m/s wind error
displaces the origin ~6 km in 12 h.

## Accuracy, honestly

With matched forcing the hindcast recovers a synthetic origin to **~8 km** with a
17 km RMS ensemble spread — the truth sits comfortably inside the cloud. Under
mismatched forcing (analytic vs the simulator's own field) the error grows, which
is the realistic case: no drift model matches the ocean exactly.

Rule of thumb: a few-hours-old slick gives 1–10 km uncertainty; 24 h+ gives tens
of km. Good enough to rank suspects, never good enough to convict alone.

## Limits

- The hindcast is capped at 24 h. Beyond roughly two days a slick has usually
  dispersed below SAR detectability and the cloud becomes too large for
  attribution to mean anything — an honest refusal to over-reach.
- No beaching, evaporation, emulsification or vertical mixing. OpenDrift's
  OpenOil models all of these; wiring it in is the next step, and
  `opendrift 1.14.11` is already installed.
- No Stokes drift. CMEMS wave products carry it.
