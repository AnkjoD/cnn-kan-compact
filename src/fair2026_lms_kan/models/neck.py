"""MobileFusion neck for combining four backbone stages."""

import torch
import torch.nn as nn


class MobileFusionNeck(nn.Module):
    def __init__(self, in_channels: list[int], fusion_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                    nn.Conv2d(channels, fusion_dim, kernel_size=1),
                    nn.BatchNorm2d(fusion_dim),
                    nn.GELU(),
                )
                for channels in in_channels
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Keep the module names identical to the training notebooks so that
        # their saved state_dict keys load without remapping.
        self.stage_attn = nn.Sequential(nn.Linear(fusion_dim * len(in_channels), len(in_channels)))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = fusion_dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        pooled = [self.pool(projection(feature)).flatten(1) for feature, projection in zip(features, self.projections)]
        stacked = torch.stack(pooled, dim=1)
        attention = torch.softmax(self.stage_attn(torch.cat(pooled, dim=1)), dim=1)
        fused = (stacked * attention.unsqueeze(-1)).sum(dim=1)
        return self.dropout(fused)
