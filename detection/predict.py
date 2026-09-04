"""Inference: SAR scene in, segmentation mask + characterised slicks out.

    python -m detection.predict --image scene.jpg --checkpoint detection/checkpoints/unet_best.pt
    python -m detection.predict --demo-seed 4          # synthetic scene

The JSON this writes is the contract with stage B (drift): each slick carries a
centroid, a long-axis orientation and an age *interval*. Stage B seeds particles
across the polygon; the attribution stage later compares vessel courses against
the orientation.

Large scenes are processed by sliding window with overlap, because a Sentinel-1
GRD is far larger than any sensible network input and naive tiling leaves visible
seams straight through slicks.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .characterize import characterize, summarise
from .config import CLASS_COLORS, CLASS_NAMES, DEFAULT_PIXEL_SIZE_M, NUM_CLASSES, OUT_DIR
from .model import build_model
from .preprocess import preprocess_scene


def load_checkpoint(path: str, device=None):
    """Restore a trained model plus the metadata saved alongside it."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("arch", "unet")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_tiles(model, image: np.ndarray, tile: int = 256, overlap: int = 64,
                  device=None, batch_size: int = 8):
    """Sliding-window inference with cosine blending. Returns CxHxW probabilities.

    Tiles are blended with a raised-cosine weight rather than hard-stitched:
    predictions near a tile edge are made with little context and are the least
    reliable, so they are down-weighted where tiles overlap instead of one tile
    arbitrarily winning.
    """
    device = device or next(model.parameters()).device
    h, w = image.shape[-2:]
    tile = min(tile, h, w)
    step = max(tile - overlap, 1)

    acc = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)

    ramp = np.hanning(tile) if tile > 2 else np.ones(tile)
    blend = np.outer(ramp, ramp).astype(np.float32) + 1e-3

    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)

    coords, batch = [], []
    for y in ys:
        for x in xs:
            batch.append(image[y:y + tile, x:x + tile])
            coords.append((y, x))
            if len(batch) == batch_size:
                _run_batch(model, batch, coords, acc, wsum, blend, device)
                batch, coords = [], []
    if batch:
        _run_batch(model, batch, coords, acc, wsum, blend, device)

    return acc / np.maximum(wsum, 1e-6)


def _run_batch(model, batch, coords, acc, wsum, blend, device):
    x = torch.from_numpy(np.stack(batch)[:, None]).float().to(device)
    probs = F.softmax(model(x), dim=1).cpu().numpy()
    t = blend.shape[0]
    for p, (y, x0) in zip(probs, coords):
        acc[:, y:y + t, x0:x0 + t] += p * blend
        wsum[y:y + t, x0:x0 + t] += blend


def colorise(mask: np.ndarray) -> np.ndarray:
    """Class-index mask -> RGB image using the dataset palette."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, col in CLASS_COLORS.items():
        out[mask == idx] = col
    return out


def detect(
    image: np.ndarray,
    model,
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M,
    transform=None,
    despeckle: bool = True,
    tile: int = 256,
    wind_speed_ms: float | None = None,
    already_preprocessed: bool = False,
):
    """Full detection + characterisation on one scene.

    Returns ``(mask, probs, slicks)``.
    """
    img = image if already_preprocessed else preprocess_scene(
        image, already_db=True, despeckle=despeckle)
    probs = predict_tiles(model, img, tile=tile)
    mask = probs.argmax(0).astype(np.int64)
    slicks = characterize(mask, image=img, probs=probs,
                          pixel_size_m=pixel_size_m, transform=transform,
                          wind_speed_ms=wind_speed_ms)
    return mask, probs, slicks


def to_geojson(slicks, transform=None) -> dict:
    """Slick centroids as GeoJSON points, for the map layer in stage 4."""
    feats = []
    for s in slicks:
        lonlat = s.centroid_lonlat
        if lonlat is None:
            # No geotransform: emit pixel coordinates and say so explicitly, so a
            # consumer cannot mistake image space for a real position.
            lonlat = (s.centroid_rc[1], s.centroid_rc[0])
        props = s.to_dict()
        props["coordinate_space"] = "geographic" if transform is not None else "pixel"
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": list(lonlat)},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="path to a SAR scene")
    ap.add_argument("--demo-seed", type=int, help="use a synthetic scene instead")
    ap.add_argument("--checkpoint", default="detection/checkpoints/unet_best.pt")
    ap.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE_M)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--wind", type=float, default=None, help="wind speed, m/s")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()

    if a.image:
        from PIL import Image
        raw = np.array(Image.open(a.image).convert("L"), dtype=np.float32) / 255.0
        truth = None
    elif a.demo_seed is not None:
        from .data import synth_scene
        raw, truth = synth_scene(512, seed=a.demo_seed)
    else:
        ap.error("supply --image or --demo-seed")

    if not os.path.exists(a.checkpoint):
        raise SystemExit(f"No checkpoint at {a.checkpoint}. Train one first:\n"
                         f"  python -m detection.train --synthetic --epochs 15")

    model, ckpt = load_checkpoint(a.checkpoint)
    print(f"loaded {ckpt.get('arch')} (epoch {ckpt.get('epoch')}, "
          f"oil IoU {ckpt.get('oil_iou', float('nan')):.4f})")

    mask, probs, slicks = detect(raw, model, pixel_size_m=a.pixel_size,
                                 tile=a.tile, wind_speed_ms=a.wind)
    print()
    print(summarise(slicks))

    os.makedirs(a.out_dir, exist_ok=True)
    from PIL import Image as PILImage
    PILImage.fromarray(colorise(mask)).save(os.path.join(a.out_dir, "pred_mask.png"))
    with open(os.path.join(a.out_dir, "detections.json"), "w") as fh:
        json.dump(to_geojson(slicks), fh, indent=2)

    if truth is not None:
        from .losses import ConfusionMatrix
        cm = ConfusionMatrix()
        cm.update(torch.from_numpy(mask), torch.from_numpy(truth))
        print(); print(cm.report())
        PILImage.fromarray(colorise(truth)).save(
            os.path.join(a.out_dir, "truth_mask.png"))

    print(f"\nmask       -> {os.path.join(a.out_dir, 'pred_mask.png')}")
    print(f"detections -> {os.path.join(a.out_dir, 'detections.json')}")


if __name__ == "__main__":
    main()
