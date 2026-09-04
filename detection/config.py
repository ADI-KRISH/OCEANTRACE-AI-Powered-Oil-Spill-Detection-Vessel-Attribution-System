"""Class definitions, palette and shared paths for the detection module.

The five classes and their mask colours come from the Krestenitis et al.
Sentinel-1 oil-spill dataset, which is the base dataset named in CLAUDE.md.
Index order is fixed and must not be reordered -- checkpoints encode it.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")
OUT_DIR = os.path.join(HERE, "outputs")

#: Class index -> name. Order matches the Krestenitis dataset's own labelling.
CLASS_NAMES = ["sea", "oil_spill", "look_alike", "ship", "land"]
NUM_CLASSES = len(CLASS_NAMES)

SEA, OIL, LOOKALIKE, SHIP, LAND = range(NUM_CLASSES)

#: RGB colours used in the dataset's mask PNGs, for decoding and for display.
CLASS_COLORS = {
    SEA:       (0, 0, 0),
    OIL:       (0, 255, 255),
    LOOKALIKE: (255, 0, 0),
    SHIP:      (153, 76, 0),
    LAND:      (0, 153, 0),
}

#: Sentinel-1 GRD ground resolution, metres/pixel. The dataset's scenes are
#: resampled, so this is configurable -- area estimates scale with its square.
DEFAULT_PIXEL_SIZE_M = 10.0

#: Loss weights. Sea covers the overwhelming majority of every scene, so an
#: unweighted loss converges to "predict sea everywhere" and reports a fine
#: pixel accuracy while detecting nothing. Oil and look-alike are the classes
#: the task actually turns on, and look-alike carries the highest weight because
#: separating it FROM oil is the stated hard problem.
CLASS_WEIGHTS = [0.3, 3.0, 3.5, 8.0, 0.8]

IGNORE_INDEX = 255
