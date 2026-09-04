"""Quick evaluation + qualitative figure for the detection model.

    python notebooks/eval_detection.py

Writes detection/outputs/eval_grid.png -- SAR scene, ground truth and prediction
side by side for several scenes, plus the per-class metrics. Kept as a script
rather than a .ipynb so it runs in CI and diffs cleanly in git.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.characterize import characterize
from detection.config import CLASS_NAMES, OUT_DIR
from detection.data import synth_scene
from detection.losses import ConfusionMatrix
from detection.predict import colorise, detect, load_checkpoint
from detection.preprocess import preprocess_scene

CKPT = "detection/checkpoints/unet_best.pt"
N_SCENES = 4


def main():
    if not os.path.exists(CKPT):
        raise SystemExit(f"No checkpoint at {CKPT}. Train first:\n"
                         "  python -m detection.train --synthetic --epochs 25")
    model, ckpt = load_checkpoint(CKPT)
    print(f"{ckpt.get('arch')} epoch {ckpt.get('epoch')}  "
          f"oil IoU {ckpt.get('oil_iou'):.4f}")

    cm = ConfusionMatrix()
    fig, axes = plt.subplots(N_SCENES, 3, figsize=(11, 3.1 * N_SCENES))

    for i in range(N_SCENES):
        raw, truth = synth_scene(256, seed=1000 + i)
        img = preprocess_scene(raw)
        mask, probs, slicks = detect(img, model, pixel_size_m=10.0,
                                     already_preprocessed=True)
        cm.update(torch.from_numpy(mask), torch.from_numpy(truth))

        for ax, data, title in (
            (axes[i, 0], img, "SAR (preprocessed)"),
            (axes[i, 1], colorise(truth), "ground truth"),
            (axes[i, 2], colorise(mask), "prediction"),
        ):
            ax.imshow(data, cmap="gray" if data.ndim == 2 else None)
            ax.set_title(title if i == 0 else "", fontsize=10)
            ax.axis("off")
        n_oil = len(slicks)
        axes[i, 2].set_xlabel(f"{n_oil} slick(s)")
        axes[i, 0].text(3, 18, f"scene {i}", color="yellow", fontsize=9)
        if slicks:
            s = slicks[0]
            axes[i, 2].text(3, 18, f"{s.area_km2:.2f} km2  axis {s.orientation_deg:.0f}",
                            color="white", fontsize=8)

    print()
    print(cm.report())

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "eval_grid.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nfigure -> {out}")
    print("\nNOTE: synthetic data. Retrain on the Zenodo dataset before quoting.")


if __name__ == "__main__":
    main()
