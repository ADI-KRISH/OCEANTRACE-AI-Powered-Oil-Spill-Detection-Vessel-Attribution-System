"""Dataset loading for the Krestenitis Sentinel-1 oil-spill dataset.

The dataset requires a manual download (see `data/README.md`), so this module
also provides a synthetic SAR generator with the same five classes and the same
statistics that make the task hard -- dark slicks, dark *look-alikes* that are
not oil, bright ships, speckle. That keeps the whole module runnable, testable
and demoable before the real archive is on disk, and the switch between them is
one argument.

Expected real layout (either naming is accepted)::

    data/oil_spill/train/images/*.jpg      data/oil_spill/train/labels/*.png
    data/oil_spill/val/images/*.jpg        data/oil_spill/val/labels/*.png
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (CLASS_COLORS, DATA_DIR, LAND, LOOKALIKE, NUM_CLASSES, OIL,
                     SEA, SHIP)
from .preprocess import preprocess_scene

IMG_DIRS = ("images", "image", "img")
LBL_DIRS = ("labels", "label", "masks", "mask", "annotations")


# ---------------------------------------------------------------------------
# Mask decoding
# ---------------------------------------------------------------------------

def rgb_to_index(mask_rgb: np.ndarray, tol: int = 60) -> np.ndarray:
    """Decode an RGB mask into class indices by nearest palette colour.

    Nearest-colour rather than exact match because JPEG-compressed or resampled
    masks drift by a few levels; an exact lookup silently drops those pixels to
    sea, which quietly deletes training signal for the rare classes.
    """
    m = np.asarray(mask_rgb)
    if m.ndim == 2:
        return m.astype(np.int64)
    m = m[:, :, :3].astype(np.int16)

    palette = np.array([CLASS_COLORS[i] for i in range(NUM_CLASSES)], dtype=np.int16)
    d = np.linalg.norm(m[:, :, None, :] - palette[None, None, :, :], axis=-1)
    idx = np.argmin(d, axis=-1)
    # Anything far from every palette entry is treated as sea (the background).
    idx[np.min(d, axis=-1) > tol * np.sqrt(3)] = SEA
    return idx.astype(np.int64)


def load_mask(path: str) -> np.ndarray:
    from PIL import Image

    im = Image.open(path)
    if im.mode == "P":
        arr = np.array(im)
        # An indexed PNG may already hold class ids, or a palette; ids are the
        # common case and are recognisable by their small value range.
        if arr.max() < NUM_CLASSES:
            return arr.astype(np.int64)
        return rgb_to_index(np.array(im.convert("RGB")))
    if im.mode in ("L", "I"):
        arr = np.array(im)
        if arr.max() < NUM_CLASSES:
            return arr.astype(np.int64)
        return rgb_to_index(np.stack([arr] * 3, -1))
    return rgb_to_index(np.array(im.convert("RGB")))


# ---------------------------------------------------------------------------
# Synthetic SAR
# ---------------------------------------------------------------------------

def _blob(h, w, cy, cx, ry, rx, angle, rng, roughness=0.35):
    """An irregular filled ellipse, used for slicks, look-alikes and land."""
    yy, xx = np.mgrid[0:h, 0:w]
    ca, sa = np.cos(angle), np.sin(angle)
    y = (yy - cy) * ca + (xx - cx) * sa
    x = -(yy - cy) * sa + (xx - cx) * ca
    r = np.sqrt((y / max(ry, 1)) ** 2 + (x / max(rx, 1)) ** 2)
    # Perturb the radius with smooth noise so edges are ragged, not elliptical.
    theta = np.arctan2(y, np.maximum(np.abs(x), 1e-6) * np.sign(x + 1e-9))
    wob = sum(rng.uniform(-1, 1) * np.sin(k * theta + rng.uniform(0, 6.28))
              for k in (2, 3, 5))
    return r < (1.0 + roughness * wob / 3.0)


def synth_scene(size: int = 256, seed: int = 0, oil_prob: float = 0.75):
    """Generate one synthetic SAR patch and its label mask.

    Look-alikes are drawn with the *same* darkness distribution as oil and are
    separated only by shape statistics -- slicks are elongated and smooth-edged
    (they follow wind and current), look-alikes rounder and more diffuse. That is
    deliberately the real discriminator, so a model that learns "dark = oil"
    scores badly here, exactly as it would on the real archive.
    """
    rng = np.random.default_rng(seed)
    h = w = size

    # Sea: moderately bright, with a slow large-scale wind gradient.
    base = 0.55 + 0.10 * rng.standard_normal()
    yy, xx = np.mgrid[0:h, 0:w] / size
    img = base + 0.10 * np.sin(2 * np.pi * (rng.uniform() + xx * rng.uniform(.5, 2)))
    img += 0.06 * yy * rng.uniform(-1, 1)
    label = np.full((h, w), SEA, dtype=np.int64)

    # Land along one edge, sometimes.
    if rng.random() < 0.30:
        edge = rng.integers(0, 4)
        band = int(size * rng.uniform(0.10, 0.28))
        sl = [slice(None), slice(None)]
        sl[0 if edge < 2 else 1] = slice(0, band) if edge % 2 == 0 else slice(size - band, size)
        m = np.zeros((h, w), bool)
        m[tuple(sl)] = True
        m &= _blob(h, w, h / 2, w / 2, h, w, 0, rng, roughness=0.9) | True
        img[m] = 0.75 + 0.12 * rng.standard_normal(m.sum())
        label[m] = LAND

    # Oil slicks: dark, elongated, oriented -- orientation is what stage B needs.
    if rng.random() < oil_prob:
        for _ in range(rng.integers(1, 3)):
            ry = rng.uniform(size * 0.03, size * 0.07)
            rx = ry * rng.uniform(3.0, 8.0)          # high aspect ratio
            m = _blob(h, w, rng.uniform(.2, .8) * h, rng.uniform(.2, .8) * w,
                      ry, rx, rng.uniform(0, np.pi), rng, roughness=0.25)
            m &= label == SEA
            img[m] = 0.18 + 0.05 * rng.standard_normal(m.sum())
            label[m] = OIL

    # Look-alikes: equally dark, but rounder and softer-edged.
    if rng.random() < 0.55:
        for _ in range(rng.integers(1, 3)):
            ry = rng.uniform(size * 0.04, size * 0.10)
            rx = ry * rng.uniform(1.0, 1.8)          # low aspect ratio
            m = _blob(h, w, rng.uniform(.2, .8) * h, rng.uniform(.2, .8) * w,
                      ry, rx, rng.uniform(0, np.pi), rng, roughness=0.55)
            m &= label == SEA
            img[m] = 0.20 + 0.06 * rng.standard_normal(m.sum())
            label[m] = LOOKALIKE

    # Ships: bright, elongated, and sized like real vessels. At the ~10 m/px of
    # a Sentinel-1 GRD a 200 m tanker is ~20 px long and 3 px wide, so semi-axes
    # of 3-11 px cover small coasters through large tankers. Drawing them any
    # smaller (as an earlier version did) invents a sub-pixel detection problem
    # that does not exist in the real data.
    for _ in range(rng.integers(0, 5)):
        cy, cx = rng.uniform(.08, .92) * h, rng.uniform(.08, .92) * w
        half_len = rng.uniform(3.0, 11.0)          # 60-220 m at 10 m/px
        half_beam = max(half_len / rng.uniform(5.0, 8.0), 1.2)
        heading = rng.uniform(0, np.pi)
        m = _blob(h, w, cy, cx, half_beam, half_len, heading, rng, roughness=0.08)
        img[m] = np.clip(0.95 + 0.05 * rng.standard_normal(m.sum()), 0, 1.4)
        label[m] = SHIP

    # Multiplicative speckle -- the defining nuisance of SAR.
    looks = rng.uniform(3.0, 8.0)
    img = img * rng.gamma(looks, 1.0 / looks, size=(h, w))
    img = np.clip(img, 0, 1.6).astype(np.float32)
    return img, label


class SyntheticSAR(Dataset):
    """Synthetic patches with the five real classes. Deterministic per index."""

    def __init__(self, n: int = 400, size: int = 256, seed: int = 0,
                 despeckle: bool = True, augment: bool = False):
        self.n, self.size, self.seed = n, size, seed
        self.despeckle, self.augment = despeckle, augment

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, lbl = synth_scene(self.size, seed=self.seed * 100_000 + i)
        img = preprocess_scene(img, already_db=True, despeckle=self.despeckle)
        if self.augment:
            img, lbl = _augment(img, lbl, np.random.default_rng(i))
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lbl.copy()).long())


def _augment(img, lbl, rng):
    """Flips and 90-degree rotations only.

    No brightness or elastic warping: absolute backscatter level is physically
    meaningful in SAR, and stretching a slick would corrupt the shape statistics
    that separate oil from a look-alike.
    """
    if rng.random() < 0.5:
        img, lbl = img[:, ::-1], lbl[:, ::-1]
    if rng.random() < 0.5:
        img, lbl = img[::-1], lbl[::-1]
    k = int(rng.integers(0, 4))
    if k:
        img, lbl = np.rot90(img, k), np.rot90(lbl, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)


# ---------------------------------------------------------------------------
# Real dataset
# ---------------------------------------------------------------------------

def _find_split_dirs(root: str, split: str):
    base = os.path.join(root, split)
    if not os.path.isdir(base):
        return None, None
    img_dir = next((os.path.join(base, d) for d in IMG_DIRS
                    if os.path.isdir(os.path.join(base, d))), None)
    lbl_dir = next((os.path.join(base, d) for d in LBL_DIRS
                    if os.path.isdir(os.path.join(base, d))), None)
    return img_dir, lbl_dir


class OilSpillDataset(Dataset):
    """Krestenitis et al. Sentinel-1 oil-spill dataset."""

    def __init__(self, root: str | None = None, split: str = "train",
                 size: int = 256, despeckle: bool = True, augment: bool = False):
        self.root = root or os.path.join(DATA_DIR, "oil_spill")
        self.size, self.despeckle, self.augment = size, despeckle, augment

        img_dir, lbl_dir = _find_split_dirs(self.root, split)
        if img_dir is None or lbl_dir is None:
            raise FileNotFoundError(
                f"No '{split}' split under {self.root}. See data/README.md for "
                "download steps, or use SyntheticSAR for development."
            )
        self.images = sorted(sum(
            [glob.glob(os.path.join(img_dir, f"*{e}"))
             for e in (".jpg", ".jpeg", ".png", ".tif", ".tiff")], []))
        self.labels = self._pair(self.images, lbl_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found in {img_dir}")

    @staticmethod
    def _pair(images, lbl_dir):
        out = []
        for p in images:
            stem = os.path.splitext(os.path.basename(p))[0]
            hit = next((c for e in (".png", ".jpg", ".tif", ".tiff", ".bmp")
                        for c in [os.path.join(lbl_dir, stem + e)]
                        if os.path.exists(c)), None)
            if hit is None:
                raise FileNotFoundError(f"No label found for {p}")
            out.append(hit)
        return out

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from PIL import Image

        img = np.array(Image.open(self.images[i]).convert("L"), dtype=np.float32) / 255.0
        lbl = load_mask(self.labels[i])

        if self.size and img.shape[0] != self.size:
            img, lbl = _resize(img, lbl, self.size)
        img = preprocess_scene(img, already_db=True, despeckle=self.despeckle)
        if self.augment:
            img, lbl = _augment(img, lbl, np.random.default_rng())
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lbl.copy()).long())


def _resize(img, lbl, size):
    """Bilinear for the image, nearest for the mask -- never interpolate labels."""
    import cv2

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    lbl = cv2.resize(lbl.astype(np.uint8), (size, size),
                     interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return img, lbl


def get_dataset(split: str = "train", synthetic: bool = False, **kw):
    """Real dataset when present, synthetic otherwise (with a clear warning)."""
    if synthetic:
        n = kw.pop("n", 400 if split == "train" else 100)
        return SyntheticSAR(n=n, seed=0 if split == "train" else 7,
                            augment=(split == "train"), **kw)
    try:
        return OilSpillDataset(split=split, augment=(split == "train"), **kw)
    except FileNotFoundError as exc:
        print(f"[data] {exc}\n[data] Falling back to SYNTHETIC data -- metrics "
              f"from this are NOT comparable to published results.")
        n = 400 if split == "train" else 100
        return SyntheticSAR(n=n, seed=0 if split == "train" else 7,
                            augment=(split == "train"))


def class_pixel_counts(ds, max_items: int = 200) -> np.ndarray:
    """Per-class pixel counts, for sanity-checking imbalance before training."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(min(len(ds), max_items)):
        _, lbl = ds[i]
        counts += np.bincount(lbl.numpy().ravel(), minlength=NUM_CLASSES)
    return counts
