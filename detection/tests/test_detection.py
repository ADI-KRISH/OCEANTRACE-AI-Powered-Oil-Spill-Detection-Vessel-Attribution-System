"""Tests for the detection module.

    python -m pytest detection/tests -q

Focused on the things that would corrupt downstream results silently rather than
crash: mask decoding, label-preserving resize, orientation convention, area
scaling, and the guarantee that age is never presented as precise.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ..characterize import (AGE_MAX_H, characterize, classify_oil_likelihood,
                            estimate_age, orientation_to_azimuth, pixel_to_lonlat)
from ..config import CLASS_COLORS, LAND, LOOKALIKE, NUM_CLASSES, OIL, SEA, SHIP
from ..data import SyntheticSAR, _resize, rgb_to_index, synth_scene
from ..losses import CombinedLoss, ConfusionMatrix, DiceLoss
from ..model import build_model, count_parameters
from ..preprocess import lee_filter, normalise, preprocess_scene, to_db


# ------------------------------------------------------------- preprocess ---

def test_to_db_is_monotonic_and_floored():
    x = np.array([1e-9, 0.01, 0.1, 1.0], dtype=np.float32)
    db = to_db(x, floor_db=-50)
    assert db[0] == -50.0
    assert np.all(np.diff(db) >= 0)
    assert db[-1] == pytest.approx(0.0)


def test_lee_filter_smooths_flat_areas_but_keeps_edges():
    flat = np.ones((32, 32), dtype=np.float32)
    rng = np.random.default_rng(0)
    noisy = flat + rng.normal(0, 0.1, flat.shape).astype(np.float32)
    out = lee_filter(noisy, size=5)
    assert out.std() < noisy.std(), "speckle should be reduced"

    edge = np.zeros((32, 32), dtype=np.float32)
    edge[:, 16:] = 1.0
    filt = lee_filter(edge, size=5)
    # The step must survive: a Gaussian blur would flatten it.
    assert filt[:, :8].mean() < 0.2 and filt[:, 24:].mean() > 0.8


def test_normalise_is_robust_to_a_single_bright_outlier():
    img = np.full((64, 64), 0.5, dtype=np.float32)
    img[0, 0] = 1000.0
    out, _ = normalise(img)
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out[32, 32] == pytest.approx(0.0, abs=0.6)


def test_preprocess_handles_rgb_and_returns_unit_range():
    rgb = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    out = preprocess_scene(rgb)
    assert out.ndim == 2 and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


# ------------------------------------------------------------------- data ---

def test_rgb_to_index_decodes_every_palette_colour():
    m = np.zeros((1, NUM_CLASSES, 3), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        m[0, i] = CLASS_COLORS[i]
    assert rgb_to_index(m)[0].tolist() == list(range(NUM_CLASSES))


def test_rgb_to_index_tolerates_compression_drift():
    """A few levels of JPEG drift must not silently drop a class to sea."""
    drift = np.clip(np.array(CLASS_COLORS[OIL], dtype=np.int16) - 8, 0, 255)
    m = drift.astype(np.uint8).reshape(1, 1, 3)
    assert rgb_to_index(m)[0, 0] == OIL


def test_resize_never_interpolates_labels():
    lbl = np.zeros((64, 64), dtype=np.int64)
    lbl[10:20, 10:20] = OIL
    lbl[30:40, 30:40] = SHIP
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    _, out = _resize(img, lbl, 32)
    assert set(np.unique(out).tolist()).issubset({SEA, OIL, SHIP}), \
        "nearest-neighbour resize must not invent intermediate class ids"


def test_synth_scene_produces_the_expected_classes():
    img, lbl = synth_scene(256, seed=4)
    assert img.shape == lbl.shape == (256, 256)
    present = set(np.unique(lbl).tolist())
    assert SEA in present
    assert present.issubset(set(range(NUM_CLASSES)))


def test_synthetic_dataset_is_deterministic():
    a = SyntheticSAR(n=4, size=128, seed=1)
    b = SyntheticSAR(n=4, size=128, seed=1)
    assert torch.equal(a[2][0], b[2][0]) and torch.equal(a[2][1], b[2][1])


def test_synthetic_oil_is_elongated_and_lookalikes_are_not():
    """The generator must encode the real discriminator, not just darkness."""
    from skimage import measure

    oil_ar, look_ar = [], []
    for seed in range(40):
        _, lbl = synth_scene(256, seed=seed)
        for cls, store in ((OIL, oil_ar), (LOOKALIKE, look_ar)):
            for r in measure.regionprops(measure.label(lbl == cls)):
                if r.area > 200 and r.axis_minor_length > 1:
                    store.append(r.axis_major_length / r.axis_minor_length)
    assert oil_ar and look_ar
    assert np.median(oil_ar) > np.median(look_ar) * 1.8


# ------------------------------------------------------------------ model ---

def test_unet_preserves_spatial_size():
    m = build_model("unet")
    out = m(torch.randn(2, 1, 128, 128))
    assert out.shape == (2, NUM_CLASSES, 128, 128)


def test_unet_handles_non_power_of_two_input():
    """Odd sizes must pad, not crop -- the output has to match the input."""
    m = build_model("unet", depth=3)
    out = m(torch.randn(1, 1, 100, 140))
    assert out.shape[-2:] == (100, 140)


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("resnet")


def test_model_is_reasonably_sized_for_6gb():
    assert count_parameters(build_model("unet")) < 15_000_000


# ----------------------------------------------------------------- losses ---

def test_dice_is_zero_for_a_perfect_prediction():
    target = torch.randint(0, NUM_CLASSES, (2, 16, 16))
    logits = torch.zeros(2, NUM_CLASSES, 16, 16)
    logits.scatter_(1, target[:, None], 20.0)
    assert float(DiceLoss()(logits, target)) < 0.02


def test_dice_ignores_classes_absent_from_the_batch():
    """A scene with no ship must not earn free credit for the ship class."""
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, NUM_CLASSES, 8, 8)
    logits[:, SEA] = 20.0
    assert float(DiceLoss()(logits, target)) < 0.02


def test_class_weights_penalise_predicting_only_sea():
    """The imbalance guard: all-sea must cost more than the correct answer."""
    target = torch.full((1, 16, 16), SEA, dtype=torch.long)
    target[0, :4, :4] = OIL
    crit = CombinedLoss()
    all_sea = torch.zeros(1, NUM_CLASSES, 16, 16)
    all_sea[:, SEA] = 10.0
    correct = torch.zeros(1, NUM_CLASSES, 16, 16)
    correct.scatter_(1, target[:, None], 10.0)
    assert float(crit(all_sea, target)) > float(crit(correct, target))


def test_confusion_matrix_iou_and_nan_handling():
    cm = ConfusionMatrix()
    t = torch.full((4, 4), SEA, dtype=torch.long)
    cm.update(t, t)
    iou = cm.iou()
    assert iou[SEA] == pytest.approx(1.0)
    # Classes neither present nor predicted are undefined, not zero.
    assert np.isnan(iou[SHIP])
    assert cm.pixel_accuracy() == pytest.approx(1.0)


def test_oil_lookalike_confusion_is_reported():
    cm = ConfusionMatrix()
    target = torch.full((10, 10), LOOKALIKE, dtype=torch.long)
    pred = torch.full((10, 10), OIL, dtype=torch.long)
    cm.update(pred, target)
    c = cm.oil_lookalike_confusion()
    assert c["false_oil_rate"] == pytest.approx(1.0)
    assert "look-alike" in cm.report()


# --------------------------------------------------------- characterisation ---

def test_orientation_convention_matches_compass_azimuth():
    assert orientation_to_azimuth(0.0) == pytest.approx(0.0)             # N-S
    assert orientation_to_azimuth(math.pi / 2) == pytest.approx(90.0)    # E-W
    assert orientation_to_azimuth(-math.pi / 4) == pytest.approx(45.0)   # NE-SW
    assert orientation_to_azimuth(math.pi / 4) == pytest.approx(135.0)   # NW-SE
    assert 0.0 <= orientation_to_azimuth(-1.2) < 180.0


def test_orientation_recovered_from_a_known_slick():
    """A horizontal bar must report an east-west axis."""
    mask = np.zeros((128, 128), dtype=np.int64)
    mask[60:68, 20:110] = OIL
    s = characterize(mask, pixel_size_m=10.0)[0]
    assert s.orientation_deg == pytest.approx(90.0, abs=5.0)
    assert s.elongation > 5


def test_area_scales_with_pixel_size_squared():
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[20:40, 20:40] = OIL
    a10 = characterize(mask, pixel_size_m=10.0)[0].area_m2
    a20 = characterize(mask, pixel_size_m=20.0)[0].area_m2
    assert a20 == pytest.approx(a10 * 4)
    assert characterize(mask, pixel_size_m=10.0)[0].area_px == 400


def test_small_regions_are_dropped_as_speckle():
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[0:3, 0:3] = OIL          # 9 px, below MIN_AREA_PX
    assert characterize(mask, pixel_size_m=10.0) == []


def test_characterize_returns_empty_when_no_oil():
    assert characterize(np.zeros((32, 32), dtype=np.int64)) == []


def test_slicks_sorted_by_area_descending():
    mask = np.zeros((128, 128), dtype=np.int64)
    mask[10:20, 10:40] = OIL       # 300 px
    mask[60:80, 60:100] = OIL      # 800 px
    s = characterize(mask, pixel_size_m=10.0)
    assert len(s) == 2 and s[0].area_px > s[1].area_px


def test_age_is_never_reported_as_precise():
    """CLAUDE.md: age must always carry an interval and low confidence."""
    a = estimate_age(200_000.0, elongation=4.0)
    assert a["confidence"] == "low"
    assert a["range"][0] < a["hours"] < a["range"][1]
    assert "order-of-magnitude" in a["basis"]


def test_age_saturation_is_flagged_not_silently_clipped():
    """A huge slick must be reported as saturated, not as a precise ceiling."""
    a = estimate_age(50e6, elongation=3.0)
    assert a["saturated"] == "upper"
    assert a["hours"] == pytest.approx(AGE_MAX_H)
    assert "report as" in a["basis"]


def test_age_increases_with_area():
    small = estimate_age(50_000.0, elongation=3.0)["hours"]
    big = estimate_age(300_000.0, elongation=3.0)["hours"]
    assert big > small


def test_oil_likelihood_prefers_elongated_dark_regions():
    likely, _ = classify_oil_likelihood(elongation=6.0, solidity=0.92,
                                        contrast_db=-5.0, confidence=0.9)
    unlikely, _ = classify_oil_likelihood(elongation=1.2, solidity=0.6,
                                          contrast_db=-1.0, confidence=0.4)
    assert likely == "likely_oil"
    assert unlikely == "uncertain"


def test_pixel_to_lonlat_applies_geotransform():
    assert pixel_to_lonlat(0, 0, None) is None
    lon, lat = pixel_to_lonlat(10, 20, (73.0, 0.001, 0.0, 19.0, 0.0, -0.001))
    assert lon == pytest.approx(73.02)
    assert lat == pytest.approx(18.99)


def test_characterize_end_to_end_on_a_synthetic_scene():
    img, lbl = synth_scene(256, seed=4)
    slicks = characterize(lbl, image=img, pixel_size_m=10.0)
    for s in slicks:
        assert s.area_km2 > 0
        assert 0.0 <= s.orientation_deg < 180.0
        assert s.age_confidence == "low"
        assert s.oil_likelihood in ("likely_oil", "possible_oil", "uncertain")
