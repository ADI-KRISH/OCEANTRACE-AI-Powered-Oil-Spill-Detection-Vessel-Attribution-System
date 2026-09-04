"""Characterise a detected slick: geometry, shape descriptors, and a rough age.

This is the handover point to stage B (drift). Two outputs matter most
downstream: the **centroid** (where to seed backward particles) and the
**orientation** of the major axis, which becomes the slick long-axis bearing the
attribution stage compares vessel courses against.

CLAUDE.md is explicit that age is never presented as exact. Every age here
carries an interval and a `confidence` of `"low"`, and the reasoning is returned
alongside it so nobody downstream can mistake it for a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .config import DEFAULT_PIXEL_SIZE_M, OIL

#: Slicks below this area are dropped as speckle survivors rather than reported.
MIN_AREA_PX = 60


@dataclass
class SlickFeature:
    """One detected oil slick and everything measured about it."""

    id: int
    area_px: int
    area_m2: float
    area_km2: float
    perimeter_m: float
    centroid_rc: tuple            # (row, col) in pixels
    centroid_lonlat: tuple | None # (lon, lat) when a geotransform is supplied
    polygon_rc: list              # outline as [(row, col), ...]
    polygon_lonlat: list | None   # outline as [(lon, lat), ...] when georeferenced
    orientation_deg: float        # major-axis azimuth, degrees CW from north
    major_axis_m: float
    minor_axis_m: float
    elongation: float             # major / minor
    eccentricity: float
    solidity: float               # area / convex-hull area
    compactness: float            # 4*pi*area / perimeter^2  (1.0 = circle)
    mean_backscatter: float
    contrast_db: float            # slick darkness relative to surrounding sea
    confidence: float             # mean model probability over the region
    age_estimate_h: float | None = None
    age_range_h: tuple | None = None
    age_confidence: str = "low"
    age_saturated: str | None = None
    age_basis: str = ""
    oil_likelihood: str = "unknown"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def pixel_to_lonlat(row, col, transform):
    """Apply an affine geotransform (GDAL order) to pixel coordinates."""
    if transform is None:
        return None
    a, b, c, d, e, f = transform[:6]
    x = a + col * b + row * c
    y = d + col * e + row * f
    return (float(x), float(y))


def orientation_to_azimuth(orientation_rad: float) -> float:
    """skimage `regionprops.orientation` -> compass azimuth in [0, 180).

    skimage measures the angle between the major axis and the *row* axis,
    counter-clockwise and in [-pi/2, pi/2]. Rows run north-south, so the sign has
    to be negated to get a clockwise compass azimuth. Verified against all four
    cardinal cases:

        N-S    orientation   0 deg -> azimuth   0
        E-W    orientation  90 deg -> azimuth  90
        NE-SW  orientation -45 deg -> azimuth  45
        NW-SE  orientation +45 deg -> azimuth 135

    The result is folded to [0, 180) because a slick axis is undirected -- 070 and
    250 describe the same axis, and the attribution stage compares vessel courses
    against it with cos^2 for exactly that reason.
    """
    return (-math.degrees(orientation_rad)) % 180.0


#: Lumped Fay gravity-viscous spreading constant, m/s^(1/4), for r = K * t^(1/4).
#: Calibrated so a ~0.2 km^2 slick reads as a few hours old, which is the regime
#: most operational detections fall in. It absorbs discharge volume, oil type and
#: sea state -- none of which a single SAR scene reveals -- so it is a scale
#: factor, not a physical constant.
FAY_K = 20.0

#: Below/above these the estimate is reported as saturated rather than as a value.
AGE_MIN_H, AGE_MAX_H = 0.5, 48.0


def estimate_age(area_m2: float, elongation: float,
                 wind_speed_ms: float | None = None) -> dict:
    """Very rough slick age from spreading, returned as an interval.

    Inverts the gravity-viscous phase of Fay spreading (radius ~ t^(1/4)) for
    time. Note what that inversion implies: t ~ r^4, so a 20% error in the
    measured radius becomes a **factor-of-two** error in age. Age-from-area is
    therefore a weak estimator by construction, and is treated as one --
    the interval spans a factor of four either way and confidence is always low.

    Beyond the bounds the result is reported as saturated ("> 48 h") instead of a
    number, because a precise-looking 96.0 would be false precision -- exactly
    what CLAUDE.md forbids.

    Real age depends on discharge volume, oil type, sea state and weathering,
    none of which are observable from one SAR scene. Stage B should treat this as
    a loose prior on its search window, never as a constraint.
    """
    if area_m2 <= 0:
        return {"hours": None, "range": None, "confidence": "low",
                "saturated": None, "basis": "no area"}

    radius_m = math.sqrt(area_m2 / math.pi)
    raw_h = (radius_m / FAY_K) ** 4 / 3600.0

    # Elongated slicks have been stretched by wind and current, implying age;
    # a compact patch is more consistent with a recent release.
    if elongation > 4:
        raw_h *= 1.4
    elif elongation < 2:
        raw_h *= 0.7
    if wind_speed_ms is not None and wind_speed_ms > 8:
        # Strong wind disperses oil faster, so a given area implies a younger slick.
        raw_h *= 0.6

    saturated = None
    if raw_h > AGE_MAX_H:
        saturated = "upper"
    elif raw_h < AGE_MIN_H:
        saturated = "lower"

    hours = float(np.clip(raw_h, AGE_MIN_H, AGE_MAX_H))
    basis = (f"Fay gravity-viscous spreading (r ~ t^1/4, K={FAY_K}), adjusted for "
             "elongation" + ("" if wind_speed_ms is None else " and wind")
             + ". Since t ~ r^4, this is order-of-magnitude only.")
    if saturated == "upper":
        basis += (f" Slick is larger than the model resolves -- report as "
                  f"'> {AGE_MAX_H:.0f} h', not as {hours:.0f} h.")
    elif saturated == "lower":
        basis += f" Slick is very small -- report as '< {AGE_MIN_H:.1f} h'."

    return {
        "hours": round(hours, 1),
        "range": (round(hours / 4.0, 1), round(hours * 4.0, 1)),
        "confidence": "low",
        "saturated": saturated,
        "basis": basis,
    }


def classify_oil_likelihood(elongation: float, solidity: float,
                            contrast_db: float, confidence: float) -> tuple:
    """A transparent second opinion on whether a region is oil or a look-alike.

    The network already decided; this re-states the decision in measurable terms
    so an analyst can see *why* a region looks like oil rather than being handed
    a bare softmax score. Real slicks are elongated, sharply bounded and strongly
    dark; low-wind zones and algae are rounder, softer and shallower.
    """
    notes = []
    score = 0.0
    if elongation >= 3.0:
        score += 0.30
        notes.append(f"Elongated (aspect {elongation:.1f}:1), consistent with "
                     "wind/current stretching")
    elif elongation < 1.8:
        score -= 0.20
        notes.append(f"Compact (aspect {elongation:.1f}:1) -- look-alikes such as "
                     "low-wind zones are typically round")
    if solidity >= 0.85:
        score += 0.15
        notes.append(f"Well-defined boundary (solidity {solidity:.2f})")
    else:
        score -= 0.10
        notes.append(f"Ragged/diffuse boundary (solidity {solidity:.2f})")
    if contrast_db <= -3.0:
        score += 0.25
        notes.append(f"Strong dampening ({contrast_db:.1f} dB below surrounding sea)")
    elif contrast_db > -1.5:
        score -= 0.15
        notes.append(f"Weak contrast ({contrast_db:.1f} dB) -- may be a wind shadow")
    score += 0.3 * (confidence - 0.5) * 2

    if score >= 0.45:
        return "likely_oil", notes
    if score >= 0.15:
        return "possible_oil", notes
    return "uncertain", notes


def _remove_small(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components with fewer than `min_area` pixels."""
    from skimage import measure

    lab = measure.label(binary, connectivity=2)
    if lab.max() == 0:
        return binary
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = counts >= min_area
    return keep[lab]


def extract_polygon(region_mask: np.ndarray, simplify_px: float = 1.5) -> list:
    """Trace the outline of a binary region as a list of (row, col) vertices.

    The frontend draws these as the spill polygon, and stage B seeds drift
    particles across them, so the outline is simplified (Douglas-Peucker) rather
    than returned per-pixel -- a raw contour of a large slick is tens of
    thousands of points and will not survive a JSON round-trip to a browser.
    """
    from skimage import measure

    contours = measure.find_contours(region_mask.astype(float), 0.5)
    if not contours:
        return []
    # The longest contour is the outer boundary; interior holes are shorter.
    contour = max(contours, key=len)
    if simplify_px > 0 and len(contour) > 4:
        contour = measure.approximate_polygon(contour, tolerance=simplify_px)
    return [(round(float(r), 2), round(float(c), 2)) for r, c in contour]


def characterize(
    mask: np.ndarray,
    image: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M,
    transform=None,
    target_class: int = OIL,
    min_area_px: int = MIN_AREA_PX,
    wind_speed_ms: float | None = None,
) -> list[SlickFeature]:
    """Measure every connected oil region in a predicted mask.

    Parameters
    ----------
    mask: HxW class-index array from the segmentation model.
    image: the preprocessed SAR image, for backscatter and contrast.
    probs: CxHxW softmax output, for per-region confidence.
    pixel_size_m: ground sample distance; area scales with its square.
    transform: optional affine geotransform for lon/lat centroids.
    """
    from skimage import measure

    binary = (np.asarray(mask) == target_class)
    if not binary.any():
        return []

    # Remove speckle survivors and close pinholes before measuring: a hole in the
    # middle of a slick would otherwise corrupt solidity and perimeter. Done by
    # hand rather than via skimage.morphology, whose min_size/area_threshold
    # arguments changed meaning in 0.26 -- this keeps behaviour identical across
    # skimage versions.
    binary = _remove_small(binary, min_area_px)
    binary = ~_remove_small(~binary, min_area_px)
    if not binary.any():
        return []

    labelled = measure.label(binary, connectivity=2)
    px_area = pixel_size_m ** 2

    # Sea reference for contrast: everything the model did not call oil.
    sea_ref = None
    if image is not None:
        sea_pixels = np.asarray(image)[np.asarray(mask) == 0]
        if sea_pixels.size > 50:
            sea_ref = float(np.median(sea_pixels))

    out = []
    for i, region in enumerate(measure.regionprops(labelled), start=1):
        area_px = int(region.area)
        area_m2 = area_px * px_area
        major = float(region.axis_major_length) * pixel_size_m
        minor = float(region.axis_minor_length) * pixel_size_m
        elong = major / minor if minor > 1e-6 else float("inf")
        perim = float(region.perimeter) * pixel_size_m
        compact = (4 * math.pi * area_m2 / perim ** 2) if perim > 0 else 0.0

        mean_bs, contrast = float("nan"), float("nan")
        if image is not None:
            vals = np.asarray(image)[labelled == region.label]
            mean_bs = float(np.mean(vals))
            if sea_ref and sea_ref > 1e-6 and mean_bs > 1e-6:
                contrast = float(10 * np.log10(mean_bs / sea_ref))

        conf = float("nan")
        if probs is not None:
            conf = float(np.mean(np.asarray(probs)[target_class][labelled == region.label]))

        az = orientation_to_azimuth(float(region.orientation))
        age = estimate_age(area_m2, elong if np.isfinite(elong) else 1.0, wind_speed_ms)
        likelihood, notes = classify_oil_likelihood(
            elong if np.isfinite(elong) else 1.0,
            float(region.solidity),
            contrast if np.isfinite(contrast) else -2.0,
            conf if np.isfinite(conf) else 0.5,
        )

        cy, cx = region.centroid
        poly_rc = extract_polygon(labelled == region.label)
        poly_ll = ([pixel_to_lonlat(r, c, transform) for r, c in poly_rc]
                   if transform is not None else None)
        out.append(SlickFeature(
            id=i, area_px=area_px, area_m2=round(area_m2, 1),
            area_km2=round(area_m2 / 1e6, 4), perimeter_m=round(perim, 1),
            centroid_rc=(round(float(cy), 2), round(float(cx), 2)),
            centroid_lonlat=pixel_to_lonlat(cy, cx, transform),
            polygon_rc=poly_rc, polygon_lonlat=poly_ll,
            orientation_deg=round(az, 1),
            major_axis_m=round(major, 1), minor_axis_m=round(minor, 1),
            elongation=round(float(elong), 2) if np.isfinite(elong) else -1.0,
            eccentricity=round(float(region.eccentricity), 3),
            solidity=round(float(region.solidity), 3),
            compactness=round(float(compact), 3),
            mean_backscatter=round(mean_bs, 4) if np.isfinite(mean_bs) else -1.0,
            contrast_db=round(contrast, 2) if np.isfinite(contrast) else float("nan"),
            confidence=round(conf, 3) if np.isfinite(conf) else -1.0,
            age_estimate_h=age["hours"], age_range_h=age["range"],
            age_confidence=age["confidence"], age_saturated=age["saturated"],
            age_basis=age["basis"],
            oil_likelihood=likelihood, notes=notes,
        ))

    out.sort(key=lambda s: -s.area_m2)
    return out


def summarise(slicks: list[SlickFeature]) -> str:
    if not slicks:
        return "No oil slicks detected."
    lines = [f"{len(slicks)} slick(s) detected:"]
    for s in slicks:
        lines.append(
            f"\n  #{s.id}  {s.area_km2:.3f} km^2  axis {s.orientation_deg:.0f}deg"
            f"  aspect {s.elongation:.1f}:1  [{s.oil_likelihood}]")
        if s.age_estimate_h is not None:
            if s.age_saturated == "upper":
                age_txt = f"age > {s.age_estimate_h:.0f} h (beyond model range)"
            elif s.age_saturated == "lower":
                age_txt = f"age < {s.age_estimate_h:.1f} h (below model range)"
            else:
                age_txt = (f"age ~{s.age_estimate_h:.1f} h "
                           f"(range {s.age_range_h[0]}-{s.age_range_h[1]} h)")
            lines.append(f"      {age_txt}, {s.age_confidence} confidence "
                         f"-- an estimate, not a measurement")
        for n in s.notes:
            lines.append(f"      - {n}")
    return "\n".join(lines)
