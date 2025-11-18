"""
Optional U-Net with ResNet encoder (spec Section 4.2).

For now this is a thin convenience wrapper around torchvision's ResNet-34
with a simple decoder. It can be extended later if needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNetEncoder(nn.Module):
  def __init__(self, in_channels: int, pretrained: bool = False):
    super().__init__()
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet34(weights=weights)
    if in_channels != 3:
      # replace first conv to accept arbitrary channels
      conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
      nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
      m.conv1 = conv1
    self.layer0 = nn.Sequential(m.conv1, m.bn1, m.relu)
    self.pool = m.maxpool
    self.layer1 = m.layer1
    self.layer2 = m.layer2
    self.layer3 = m.layer3
    self.layer4 = m.layer4

  def forward(self, x: torch.Tensor):
    x0 = self.layer0(x)
    x1 = self.layer1(self.pool(x0))
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    return x0, x1, x2, x3, x4


class UpBlock(nn.Module):
  def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv = nn.Sequential(
      nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    x = self.up(x)
    # align shapes
    if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
      diff_y = skip.size(2) - x.size(2)
      diff_x = skip.size(3) - x.size(3)
      x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
    x = torch.cat([skip, x], dim=1)
    return self.conv(x)


class UNet2DResNetBackbone(nn.Module):
  def __init__(self, in_channels: int, num_classes: int = 1, pretrained: bool = False):
    super().__init__()
    self.encoder = ResNetEncoder(in_channels, pretrained=pretrained)
    self.up4 = UpBlock(512, 256, 256)
    self.up3 = UpBlock(256, 128, 128)
    self.up2 = UpBlock(128, 64, 64)
    self.up1 = UpBlock(64, 64, 64)
    self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x0, x1, x2, x3, x4 = self.encoder(x)
    x = self.up4(x4, x3)
    x = self.up3(x, x2)
    x = self.up2(x, x1)
    x = self.up1(x, x0)
    return self.out_conv(x)

