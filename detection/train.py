"""Train the segmentation model.

    python -m detection.train --arch unet --epochs 30
    python -m detection.train --synthetic --epochs 15      # no dataset needed

Checkpoints are selected on **oil-spill IoU**, not on mIoU or loss. A model that
segments sea and land beautifully while missing slicks is useless for this task,
and mIoU averages that failure away.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CKPT_DIR, CLASS_NAMES, NUM_CLASSES, OIL, OUT_DIR
from .data import class_pixel_counts, get_dataset
from .losses import CombinedLoss, ConfusionMatrix
from .model import build_model, count_parameters


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimiser, device, scaler=None, train=True):
    model.train(train)
    cm = ConfusionMatrix(NUM_CLASSES)
    total, n = 0.0, 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, masks)
                if train:
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()

        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
        cm.update(logits.argmax(1), masks)

    return total / max(n, 1), cm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="unet",
                    choices=["unet", "deeplabv3+", "mobilenet"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--synthetic", action="store_true",
                    help="train on generated data (no dataset download needed)")
    ap.add_argument("--train-n", type=int, default=400)
    ap.add_argument("--val-n", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--out", default=CKPT_DIR)
    a = ap.parse_args()

    device = get_device(a.device)
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    kw = {"size": a.size}
    train_ds = get_dataset("train", synthetic=a.synthetic,
                           **({"n": a.train_n, **kw} if a.synthetic else kw))
    val_ds = get_dataset("val", synthetic=a.synthetic,
                         **({"n": a.val_n, **kw} if a.synthetic else kw))
    print(f"train {len(train_ds)} | val {len(val_ds)}")

    counts = class_pixel_counts(train_ds, max_items=min(len(train_ds), 100))
    tot = counts.sum()
    print("class balance (train sample):")
    for name, c in zip(CLASS_NAMES, counts):
        print(f"  {name:<12} {c/tot*100:6.2f}%")

    train_ld = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                          num_workers=a.workers, pin_memory=(device.type == "cuda"),
                          drop_last=len(train_ds) > a.batch_size)
    val_ld = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=(device.type == "cuda"))

    model = build_model(a.arch).to(device)
    print(f"{a.arch}: {count_parameters(model):,} trainable parameters")

    criterion = CombinedLoss().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda") if (a.amp and device.type == "cuda") else None

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    best_oil, history = -1.0, []
    ckpt_path = os.path.join(a.out, f"{a.arch}_best.pt")

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, train_ld, criterion, optimiser, device,
                               scaler, train=True)
        va_loss, cm = run_epoch(model, val_ld, criterion, optimiser, device,
                                None, train=False)
        sched.step()

        iou = cm.iou()
        oil_iou = float(iou[OIL]) if not np.isnan(iou[OIL]) else 0.0
        conf = cm.oil_lookalike_confusion()
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "miou": cm.miou(), "oil_iou": oil_iou,
                        "false_oil_rate": conf["false_oil_rate"]})

        flag = ""
        if oil_iou > best_oil:
            best_oil = oil_iou
            torch.save({"model": model.state_dict(), "arch": a.arch,
                        "epoch": epoch, "oil_iou": oil_iou, "miou": cm.miou(),
                        "classes": CLASS_NAMES, "size": a.size}, ckpt_path)
            flag = "  <- best"

        print(f"epoch {epoch:3d}/{a.epochs}  train {tr_loss:.4f}  val {va_loss:.4f}"
              f"  mIoU {cm.miou():.4f}  oilIoU {oil_iou:.4f}"
              f"  falseOil {conf['false_oil_rate']*100:5.1f}%"
              f"  {time.time()-t0:4.0f}s{flag}")

    print("\nFinal validation:")
    print(cm.report())
    hist_path = os.path.join(OUT_DIR, f"history_{a.arch}.json")
    with open(hist_path, "w") as fh:
        json.dump({"args": vars(a), "history": history,
                   "best_oil_iou": best_oil,
                   "final_confusion": cm.mat.tolist()}, fh, indent=2)
    print(f"\nbest oil IoU {best_oil:.4f} -> {ckpt_path}")
    print(f"history -> {hist_path}")
    if a.synthetic:
        print("\nNOTE: trained on SYNTHETIC data. These numbers describe the "
              "generator, not Sentinel-1. Retrain on the Zenodo dataset before "
              "quoting any figure.")


if __name__ == "__main__":
    main()
