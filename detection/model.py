"""Segmentation models: U-Net and DeepLabv3+ (both named in CLAUDE.md).

U-Net is the default. Its skip connections carry full-resolution detail into the
decoder, which matters here because slick *edges* are what characterisation
measures downstream -- a model that finds the right blob with a sloppy boundary
still corrupts the area and orientation estimates that stage B consumes.

DeepLabv3+ is offered for comparison: its atrous pyramid sees more context, which
helps on large diffuse look-alikes, at the cost of a coarser boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm + ReLU."""

    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net for single-channel SAR input.

    `base` controls width; 32 keeps the whole model under ~8M parameters, which
    trains comfortably in 6 GB at 256x256 with room for a batch of 8.
    """

    def __init__(self, in_ch: int = 1, n_classes: int = NUM_CLASSES,
                 base: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth + 1)]

        self.downs = nn.ModuleList()
        c = in_ch
        for i in range(depth):
            self.downs.append(ConvBlock(c, chs[i]))
            c = chs[i]
        self.bottleneck = ConvBlock(c, chs[depth], dropout=dropout)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.up_convs.append(ConvBlock(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            # Odd input sizes leave a one-pixel mismatch; pad rather than crop so
            # the output keeps the input's exact resolution.
            if x.shape[-2:] != skip.shape[-2:]:
                dy = skip.shape[-2] - x.shape[-2]
                dx = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)


class DeepLabV3Plus(nn.Module):
    """torchvision DeepLabv3-ResNet50, adapted to 1-channel input.

    The pretrained stem expects 3 channels; its weights are summed across the
    input dimension rather than discarded, which preserves the learned edge
    filters instead of reinitialising the first layer from scratch.
    """

    def __init__(self, n_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50

        weights = "DEFAULT" if pretrained else None
        self.net = deeplabv3_resnet50(weights=weights, aux_loss=True)

        old = self.net.backbone.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        self.net.backbone.conv1 = new

        self.net.classifier[-1] = nn.Conv2d(256, n_classes, 1)
        if self.net.aux_classifier is not None:
            self.net.aux_classifier[-1] = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        out = self.net(x)
        return out["out"] if isinstance(out, dict) else out


def build_model(arch: str = "unet", n_classes: int = NUM_CLASSES, **kw) -> nn.Module:
    arch = arch.lower()
    if arch == "unet":
        return UNet(n_classes=n_classes, **kw)
    if arch in ("deeplabv3+", "deeplabv3plus", "deeplab"):
        return DeepLabV3Plus(n_classes=n_classes, **kw)
    raise ValueError(f"unknown arch {arch!r}; use 'unet' or 'deeplabv3+'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
