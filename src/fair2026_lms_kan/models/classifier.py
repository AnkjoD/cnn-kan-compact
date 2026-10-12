"""LMS-ConvNeXt classifier with selectable FastKAN, linear, or MLP head."""

import torch
import torch.nn as nn

from .backbone import build_lms_convnext_backbone
from .heads import FastKANHead, LinearHead, MLPHead
from .neck import MobileFusionNeck


class LMSConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        head: str = "fastkan",
        backbone_size: str = "pico",
        fusion_dim: int = 256,
        hidden_dim: int = 64,
        grid_size: int = 4,
        dropout: float = 0.057746090438249055,
        num_classes: int = 2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = build_lms_convnext_backbone(
            backbone_size, drop_path_rate=drop_path_rate
        )
        self.neck = MobileFusionNeck(self.backbone.out_channels, fusion_dim, dropout)
        if head == "fastkan":
            self.head = FastKANHead(fusion_dim, hidden_dim, num_classes, grid_size, dropout)
        elif head == "linear":
            self.head = LinearHead(fusion_dim, num_classes, dropout)
        elif head == "mlp":
            self.head = MLPHead(fusion_dim, hidden_dim, num_classes, dropout)
        else:
            raise ValueError("head must be one of: fastkan, linear, mlp")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(**kwargs) -> LMSConvNeXtClassifier:
    return LMSConvNeXtClassifier(**kwargs)

