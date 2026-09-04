"""Losses and segmentation metrics.

Sea covers ~90% of a typical scene and oil ~1-4%, so plain cross-entropy is
optimised by predicting sea nearly everywhere. Two corrections are applied:
class-weighted CE, and a Dice term that scores each class by *region overlap*
rather than per-pixel correctness, so a class contributes equally regardless of
how few pixels it occupies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASS_NAMES, CLASS_WEIGHTS, NUM_CLASSES, OIL, LOOKALIKE


class DiceLoss(nn.Module):
    """Soft multi-class Dice, averaged over classes present in the batch.

    Classes absent from a batch are skipped rather than scored as perfect; a
    scene with no ship should not hand the model free credit for the ship class.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        present = oh.sum(dims) > 0
        return 1.0 - (dice[present].mean() if present.any() else dice.mean())


class CombinedLoss(nn.Module):
    """`w_ce * weighted CE + w_dice * Dice`."""

    def __init__(self, class_weights=None, w_ce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        weights = torch.tensor(CLASS_WEIGHTS if class_weights is None
                               else class_weights, dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.w_ce, self.w_dice = w_ce, w_dice
        self.dice = DiceLoss()

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weights)
        return self.w_ce * ce + self.w_dice * self.dice(logits, target)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """Accumulates a class-by-class confusion matrix across batches."""

    def __init__(self, n_classes: int = NUM_CLASSES):
        self.n = n_classes
        self.mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred, target):
        p = pred.detach().cpu().numpy().ravel()
        t = target.detach().cpu().numpy().ravel()
        k = (t >= 0) & (t < self.n)
        self.mat += np.bincount(self.n * t[k].astype(int) + p[k],
                                minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        """Per-class IoU. NaN for classes absent from the ground truth."""
        tp = np.diag(self.mat).astype(float)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = tp + fp + fn
        out = np.full(self.n, np.nan)
        nz = denom > 0
        out[nz] = tp[nz] / denom[nz]
        # A class never present and never predicted is undefined, not zero.
        out[(self.mat.sum(1) == 0) & (self.mat.sum(0) == 0)] = np.nan
        return out

    def miou(self) -> float:
        return float(np.nanmean(self.iou()))

    def pixel_accuracy(self) -> float:
        tot = self.mat.sum()
        return float(np.diag(self.mat).sum() / tot) if tot else float("nan")

    def oil_lookalike_confusion(self) -> dict:
        """The metric that matters most: how often oil and look-alike swap.

        CLAUDE.md names look-alike suppression as the hard problem, and mIoU
        hides it -- a model can post a respectable mIoU while systematically
        calling every algae bloom an oil spill.
        """
        oil_as_look = int(self.mat[OIL, LOOKALIKE])
        look_as_oil = int(self.mat[LOOKALIKE, OIL])
        oil_total = int(self.mat[OIL].sum())
        look_total = int(self.mat[LOOKALIKE].sum())
        return {
            "oil_pixels_called_lookalike": oil_as_look,
            "lookalike_pixels_called_oil": look_as_oil,
            "oil_miss_rate": oil_as_look / oil_total if oil_total else float("nan"),
            "false_oil_rate": look_as_oil / look_total if look_total else float("nan"),
        }

    def report(self) -> str:
        iou = self.iou()
        lines = [f"{'class':<12} {'IoU':>8}"]
        for name, v in zip(CLASS_NAMES, iou):
            lines.append(f"{name:<12} {'n/a' if np.isnan(v) else f'{v:8.4f}'}")
        lines.append(f"{'mIoU':<12} {self.miou():8.4f}")
        lines.append(f"{'pixel acc':<12} {self.pixel_accuracy():8.4f}")
        c = self.oil_lookalike_confusion()
        lines.append("")
        lines.append("Oil <-> look-alike confusion (the hard problem):")
        lines.append(f"  oil called look-alike : {c['oil_miss_rate']*100:6.2f}%"
                     f"  ({c['oil_pixels_called_lookalike']:,} px)")
        lines.append(f"  look-alike called oil : {c['false_oil_rate']*100:6.2f}%"
                     f"  ({c['lookalike_pixels_called_oil']:,} px)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Object-level detection metrics
# ---------------------------------------------------------------------------

def object_detection_metrics(pred_mask, true_mask, cls: int,
                             min_area: int = 4, iou_thresh: float = 0.10):
    """Instance-level precision/recall for one class.

    Pixel IoU is the wrong yardstick for small objects: a 15x3 px ship shifted by
    two pixels loses most of its IoU while still being, operationally, a correct
    detection. What matters for a ship is whether it was *found* and whether the
    finding was spurious -- so ground-truth and predicted regions are matched as
    objects, greedily, at a deliberately loose IoU threshold.

    Returns precision, recall, F1 and the raw TP/FP/FN counts.
    """
    from skimage import measure

    gt_lab = measure.label(np.asarray(true_mask) == cls)
    pr_lab = measure.label(np.asarray(pred_mask) == cls)

    gt = [r for r in measure.regionprops(gt_lab) if r.area >= min_area]
    pr = [r for r in measure.regionprops(pr_lab) if r.area >= min_area]

    matched_pred = set()
    tp = 0
    for g in gt:
        g_pix = set(map(tuple, g.coords))
        best, best_iou = None, 0.0
        for i, p in enumerate(pr):
            if i in matched_pred:
                continue
            p_pix = set(map(tuple, p.coords))
            inter = len(g_pix & p_pix)
            if not inter:
                continue
            iou = inter / len(g_pix | p_pix)
            if iou > best_iou:
                best, best_iou = i, iou
        if best is not None and best_iou >= iou_thresh:
            matched_pred.add(best)
            tp += 1

    fn = len(gt) - tp
    fp = len(pr) - len(matched_pred)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) > 0 else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_true": len(gt), "n_pred": len(pr)}
