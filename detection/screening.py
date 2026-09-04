"""Stage 0 — fast yes/no screen before the segmenter runs.

A satellite pass produces thousands of scenes and nearly all are empty ocean.
Running the segmenter on every one wastes most of the compute, so a lightweight
classifier gates it: only scenes that might contain oil go through to Module 1.

The model here is a MobileNetV3-Small binary classifier trained on the CSIRO
oil-spill dataset (real SAR imagery, oil / no-oil labels). It answers *whether*
there is oil, never *where* -- it emits a single number, not a mask. That is why
it screens rather than replaces the U-Net: drift and attribution both need the
slick polygon, which only segmentation produces.

Threshold
---------
The default is deliberately low. For a screen, a miss is far more costly than a
false positive: a false positive costs one unnecessary segmentation, a miss loses
the spill entirely. Measured on 1000 held-out CSIRO images:

    threshold   recall   precision   scenes screened out
        0.50     0.770       0.906              57.5%
        0.25     0.900       0.811              44.5%
        0.10     0.982       0.697              29.6%   <- default
        0.05     0.996       0.636              21.7%

At 0.10 the screen discards ~30% of the workload while missing under 2% of oil.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Trained by a teammate on the CSIRO dataset; see the module docstring.
DEFAULT_WEIGHTS = os.path.join(ROOT, "mobilenetv3_oil_spill.pth.zip")

#: Favours recall -- see the table above.
DEFAULT_THRESHOLD = 0.10

#: ImageNet statistics, matching how the classifier was trained.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ScreenResult:
    """Outcome of the screen for one scene."""

    oil_likely: bool
    confidence: float          # sigmoid output in [0, 1]
    threshold: float
    model: str = "mobilenet_v3_small"
    trained_on: str = "CSIRO oil-spill dataset (real SAR)"

    def to_dict(self) -> dict:
        return {
            "oil_likely": bool(self.oil_likely),
            "confidence": round(float(self.confidence), 4),
            "threshold": self.threshold,
            "model": self.model,
            "trained_on": self.trained_on,
            "note": ("Binary screen only -- says whether oil is present, not "
                     "where. Segmentation still produces the slick outline."),
        }


class OilScreener:
    """MobileNetV3 binary screen. Loads once, reused across scenes."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS,
                 threshold: float = DEFAULT_THRESHOLD, device=None):
        import torch
        import torch.nn as nn
        import torchvision

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"Screening weights not found at {weights}. "
                "Run without screening, or point --screen-weights at the file.")

        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = torchvision.models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
        model.load_state_dict(torch.load(weights, map_location="cpu",
                                         weights_only=False))
        self.model = model.eval().to(self.device)

    @staticmethod
    def available(weights: str = DEFAULT_WEIGHTS) -> bool:
        return os.path.exists(weights)

    def _prepare(self, img: np.ndarray) -> "np.ndarray":
        """SAR scene -> the 3-channel 224x224 tensor the classifier expects.

        Our pipeline carries single-channel SAR in [0, 1]; the classifier was
        trained on RGB JPGs. The grey channel is replicated across RGB rather
        than colour-mapped, which is what the training images effectively were.
        """
        import torch

        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        if a.max() > 1.5:                      # 0-255 input
            a = a / 255.0
        a = np.clip(a, 0.0, 1.0)

        t = torch.from_numpy(a)[None, None]    # 1x1xHxW
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear",
                                            align_corners=False)
        t = t.repeat(1, 3, 1, 1)[0].numpy().transpose(1, 2, 0)
        t = (t - _MEAN) / _STD
        return torch.from_numpy(t.transpose(2, 0, 1))[None].float()

    def screen(self, img: np.ndarray) -> ScreenResult:
        """Is there oil in this scene?"""
        import torch

        x = self._prepare(img).to(self.device)
        with torch.no_grad():
            p = float(torch.sigmoid(self.model(x)).squeeze())
        return ScreenResult(oil_likely=p >= self.threshold, confidence=p,
                            threshold=self.threshold)


_SCREENER = None


def get_screener(threshold: float = DEFAULT_THRESHOLD):
    """Cached screener, or None when the weights are absent.

    Returning None rather than raising lets the pipeline run unscreened when the
    classifier is unavailable -- the screen is an optimisation, not a dependency.
    """
    global _SCREENER
    if not OilScreener.available():
        return None
    if _SCREENER is None or _SCREENER.threshold != threshold:
        _SCREENER = OilScreener(threshold=threshold)
    return _SCREENER
