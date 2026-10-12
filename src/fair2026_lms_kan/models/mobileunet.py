"""MobileUNet used to crop the lung field before classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.depthwise(x)))
        return self.activation(self.bn2(self.pointwise(x)))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = DepthwiseSeparableConv(in_channels, out_channels, stride=2)
        self.refine = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv1 = DepthwiseSeparableConv(out_channels + skip_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class MobileUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 16):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True),
        )
        self.enc1 = EncoderStage(c1, c2)
        self.enc2 = EncoderStage(c2, c3)
        self.enc3 = EncoderStage(c3, c4)
        self.enc4 = EncoderStage(c4, c5)
        self.dec4 = DecoderStage(c5, c4, c4)
        self.dec3 = DecoderStage(c4, c3, c3)
        self.dec2 = DecoderStage(c3, c2, c2)
        self.dec1 = DecoderStage(c2, c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.stem(x)
        skip1 = self.enc1(skip0)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        encoded = self.enc4(skip3)
        decoded = self.dec4(encoded, skip3)
        decoded = self.dec3(decoded, skip2)
        decoded = self.dec2(decoded, skip1)
        decoded = self.dec1(decoded, skip0)
        return self.head(decoded)


def build_mobileunet(base: int = 16) -> MobileUNet:
    return MobileUNet(base=base)

