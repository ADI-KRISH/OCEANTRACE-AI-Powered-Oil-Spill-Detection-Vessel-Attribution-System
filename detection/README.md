# Module 1 — Detection & Characterization

Segments Sentinel-1 SAR into the five classes from the Krestenitis dataset
(`sea`, `oil_spill`, `look_alike`, `ship`, `land`), then measures each detected
slick and hands the result to Module 2 (drift).

## Run

```bash
# train (no dataset needed -- generates its own SAR patches)
python -m detection.train --arch unet --synthetic --epochs 25

# train on the real Zenodo dataset once downloaded (see ../data/README.md)
python -m detection.train --arch unet --epochs 40

# inference + characterisation
python -m detection.predict --demo-seed 4
python -m detection.predict --image scene.jpg --pixel-size 10

# tests
python -m pytest detection/tests -q
```

## What it produces

`detection/outputs/detections.json` — the handover to Module 2. Per slick:

| field | why Module 2 / 3 needs it |
|---|---|
| `centroid_rc` / `centroid_lonlat` | where to seed backward drift particles |
| `orientation_deg` | slick long-axis azimuth → compared against vessel COG in attribution |
| `area_km2`, `perimeter_m`, `major/minor_axis_m` | extent, and input to the age prior |
| `elongation`, `solidity`, `compactness`, `eccentricity` | shape evidence for oil vs look-alike |
| `contrast_db` | how much darker than surrounding sea |
| `age_estimate_h`, `age_range_h`, `age_saturated` | **loose prior** on the drift search window |
| `oil_likelihood`, `notes` | plain-language reasoning, for the explainability requirement |

## Design decisions

**Why oil IoU selects the checkpoint, not mIoU or loss.** Sea is ~89% of a
typical scene. A model that segments sea and land perfectly while missing every
slick still posts a respectable mIoU. Checkpointing on oil IoU makes the metric
match the job.

**Why the loss is weighted CE + Dice.** Unweighted cross-entropy on this class
balance is minimised by predicting sea nearly everywhere. Dice scores each class
by region overlap rather than pixel count, so a 1%-of-image class still
contributes. `look_alike` carries the highest weight because separating it *from*
oil is the stated hard problem.

**Why look-alike confusion is reported separately.** mIoU hides it. A model can
look fine on average while systematically calling every algae bloom an oil spill,
which is the failure mode that would discredit the whole system. `ConfusionMatrix`
reports `oil_miss_rate` and `false_oil_rate` directly.

**Why a Lee filter and not a Gaussian blur.** SAR speckle is multiplicative.
Gaussian blurring both fails to remove it and destroys the slick boundary that
characterisation then measures. Lee smooths only where local statistics say the
region is homogeneous.

**Why tiled inference is cosine-blended.** A Sentinel-1 GRD is far bigger than any
sensible network input. Hard-stitched tiles leave seams straight through slicks;
predictions near a tile edge are made with the least context, so they are
down-weighted where tiles overlap.

**Why augmentation is flips and 90° rotations only.** No brightness jitter —
absolute backscatter is physically meaningful in SAR. No elastic warping — it
would corrupt the very shape statistics that separate oil from a look-alike.

## The orientation convention

`orientation_deg` is a compass azimuth in `[0, 180)`, folded because a slick axis
is **undirected**: a ship steaming 070 and one steaming 250 are equally
consistent with an axis of 070. Attribution compares vessel course against it
with `cos²θ` for that reason.

skimage's `regionprops.orientation` is counter-clockwise from the row axis, so
the sign must be negated. Verified against all four cardinal cases:

| slick | skimage orientation | azimuth |
|---|---|---|
| N–S | 0° | 0 |
| E–W | 90° | 90 |
| NE–SW | −45° | 45 |
| NW–SE | +45° | 135 |

## Age: read this before using it

Age is derived by inverting Fay gravity-viscous spreading (`r ~ t^¼`). Inverted,
that is **`t ~ r⁴`** — a 20% error in the measured radius becomes a factor-of-two
error in age. It is a weak estimator by construction and is reported as one:

- always an interval (a factor of four either way), never a single number
- always `age_confidence: "low"`
- beyond the model's range it reports `age_saturated` and the summary prints
  `"> 48 h"` rather than a precise-looking `96.0`

The `FAY_K` constant absorbs discharge volume, oil type and sea state — none of
which one SAR scene reveals. **Module 2 should treat age as a loose prior on its
search window, never as a constraint.**

## Architecture benchmark

All three trained on identical data, identical loss, identical metrics —
25 epochs, 600 train / 150 val at 256×256, RTX 4050.

| arch | params | **oil IoU** | mIoU | look-alike | ship | land | false alarm | s/epoch |
|---|---|---|---|---|---|---|---|---|
| **U-Net** | **7.8M** | **0.939** | 0.763 | 0.852 | 0.275 | 0.775 | 5.6% | 37 |
| DeepLabv3+ (R50) | 42.0M | 0.929 | **0.774** | **0.858** | **0.279** | **0.824** | **2.3%** | 29 |
| DeepLab-MobileNetV3 | 11.0M | 0.847 | 0.705 | 0.753 | 0.116 | 0.841 | 3.8% | 28 |

**U-Net wins on the metric that matters** — oil IoU, the class the whole system
exists to find — at **one fifth** the parameters of DeepLabv3+. It stays the
default.

DeepLabv3+ is the better *all-rounder*: higher mIoU, better look-alike
separation, and less than half the false-alarm rate. Its atrous pyramid sees more
context, which is exactly what large diffuse look-alikes need. If false alarms
turn out to matter more than raw slick recall on real data, it is the one to
switch to — hence keeping both.

**MobileNet is clearly the weakest here**: oil IoU 0.847 against U-Net's 0.939,
and ship IoU less than half. It is not meaningfully faster than DeepLabv3+ per
epoch at this input size either, so its usual advantage — speed — does not
materialise. Worth knowing before committing a demo to a MobileNet-class model.

> Synthetic data, so these compare the architectures *on this generator*. The
> ordering is informative; the absolute values are not transferable to
> Sentinel-1. Re-run on the Zenodo dataset before choosing finally.

## Results (synthetic data — see the warning below)

U-Net, 7.8M params, 30 epochs, 600 train / 150 val patches at 256×256, ~21 s per
epoch on an RTX 4050. Checkpoint selected on oil IoU.

| class | IoU |
|---|---|
| sea | 0.983 |
| **oil_spill** | **0.941** |
| look_alike | 0.855 |
| land | 0.786 |
| ship | 0.392 |
| **mIoU** | **0.791** |

Oil ↔ look-alike confusion, the metric the task actually turns on:

| | rate |
|---|---|
| oil called look-alike (missed slick) | 1.5% |
| look-alike called oil (**false alarm**) | 5.3% |

### Ships, measured properly

Pixel IoU is the wrong yardstick for a 15×3 px object — shift it two pixels and
IoU collapses even though the ship was found. `object_detection_metrics()` scores
ship *instances* instead. Over 60 held-out scenes (128 real ships):

| | |
|---|---|
| detected | 79 |
| missed | 49 |
| spurious | 63 |
| **precision / recall / F1** | **0.56 / 0.62 / 0.59** |

So the model finds about **three ships in five**, and roughly **two in five of its
ship calls are wrong**. That is honestly mediocre — usable as a supporting cue,
not as evidence on its own. It is a large improvement on the previous build
(ship IoU 0.33 → 0.39, and the earlier model detected essentially none), and the
biggest single cause was a bug in the *generator*, not the model: synthetic ships
were being drawn 5–12 px long (50–120 m), inventing a sub-pixel problem that does
not exist at Sentinel-1 resolution, where a 200 m tanker is ~20 px. Ships are now
drawn at 60–220 m and the ship class weight was raised from 2.0 to 8.0.

> **All of these numbers describe the synthetic generator, not Sentinel-1.** They
> show the pipeline converges and the metrics are wired correctly. Retrain on the
> Zenodo dataset before putting any figure in a deck.

### Land is still unreliable

The model hallucinates land in open-sea scenes and gets coastline shape wrong. In
production land should come from a coastline mask (GSHHG/OSM), not the segmenter
— a known shoreline is both more accurate and cheaper than learning one.

## Status

- [x] SAR preprocessing (dB, Lee despeckle, percentile normalisation)
- [x] U-Net + DeepLabv3+ (1-channel adapted, pretrained stem preserved)
- [x] Weighted CE + Dice, per-class IoU, oil↔look-alike confusion reporting
- [x] Synthetic SAR generator (runs with no dataset present)
- [x] Tiled inference with cosine blending
- [x] Characterisation → GeoJSON handover
- [x] 31 tests
- [x] Object-level detection metrics (precision/recall/F1 per instance)
- [ ] Train on the real Zenodo dataset — **all current numbers are synthetic**
- [x] Ship detection improved (realistic vessel sizes + class weight; F1 0.59)
- [ ] Ship detection is still mediocre — needs a small-object head to go further
- [ ] Replace learned land with a GSHHG/OSM coastline mask
- [ ] Georeferencing from real GeoTIFF metadata (`transform` is wired, untested on real scenes)
- [ ] Polygon (not just centroid) export for particle seeding
- [x] Benchmarked U-Net vs DeepLabv3+ vs MobileNetV3 (see above)
