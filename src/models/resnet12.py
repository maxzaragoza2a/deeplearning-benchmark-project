"""ResNet-12 backbone, identical to the one of the NCA run (scripts/nca_paper/model.py).

Same architecture and same layer names as Sara's `resnet12()`: BasicBlocks [1, 1, 2, 1],
widths 64-160-320-640, a stride-1 3x3 stem without max-pooling, global average pooling.
Its state_dict keys match hers, so a checkpoint from one loads into the other.

Differences, on purpose:
- `forward` returns the embedding only (n, 640), not an `(embedding, logits)` pair;
- no classification head, no projection, no multi-layer `cat` mode (unused here).
"""

from __future__ import annotations

import torch
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet12(nn.Module):
    """Normalized images (n, 3, H, W) -> embeddings (n, 640), for any input size."""

    def __init__(self, widths: tuple[int, ...] = (64, 160, 320, 640), blocks: tuple[int, ...] = (1, 1, 2, 1)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(widths[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(widths[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(widths[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(widths[3], blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = widths[3]

        # Same initialization as the NCA code (torchvision's).
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, n_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)
