#!/usr/bin/env bash
# Train all three architectures on identical data and metrics.
set -e
for A in unet mobilenet "deeplabv3+"; do
  echo "=================== $A ==================="
  python -u -m detection.train --arch "$A" --synthetic \
      --epochs 25 --train-n 600 --val-n 150 --batch-size 8
done
