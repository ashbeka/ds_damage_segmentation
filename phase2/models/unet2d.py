"""
Plain 2D U-Net for segmentation (spec Phase 2 Section 4.1).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


class Down(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.pool = nn.MaxPool2d(2)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(self.pool(x))


class Up(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    x = self.up(x)
    # pad if needed
    diff_y = skip.size(2) - x.size(2)
    diff_x = skip.size(3) - x.size(3)
    x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
    x = torch.cat([skip, x], dim=1)
    return self.conv(x)


class UNet2D(nn.Module):
  def __init__(self, in_channels: int, base_channels: int = 64, depth: int = 4, num_classes: int = 1):
    super().__init__()
    self.in_channels = in_channels
    self.base_channels = base_channels
    self.depth = depth
    self.num_classes = num_classes

    chs = [base_channels * (2 ** i) for i in range(depth)]
    self.inc = DoubleConv(in_channels, chs[0])
    downs = []
    for i in range(1, depth):
      downs.append(Down(chs[i - 1], chs[i]))
    self.down_blocks = nn.ModuleList(downs)

    ups = []
    for i in range(depth - 1, 0, -1):
      ups.append(Up(chs[i], chs[i - 1]))
    self.up_blocks = nn.ModuleList(ups)

    self.out_conv = nn.Conv2d(chs[0], num_classes, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x1 = self.inc(x)
    xs = [x1]
    x_cur = x1
    for down in self.down_blocks:
      x_cur = down(x_cur)
      xs.append(x_cur)
    x = x_cur
    for i, up in enumerate(self.up_blocks):
      skip = xs[-(i + 2)]
      x = up(x, skip)
    logits = self.out_conv(x)
    return logits


