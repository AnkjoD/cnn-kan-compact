"""Local Multi-Scale ConvNeXt backbone used in the FAIR 2026 study."""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class LMSBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dw_small = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw_mid = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dw_large = nn.Conv2d(
            dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim
        )
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 4, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fuse(
            torch.cat([self.dw_small(x), self.dw_mid(x), self.dw_large(x)], dim=1)
        )
        x = self.norm(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class LMSConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        dims: tuple[int, ...] = (48, 96, 192, 384),
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("The backbone requires four stages")

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers = nn.ModuleList([nn.Identity()])
        for index in range(3):
            self.downsample_layers.append(Downsample(dims[index], dims[index + 1]))

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cursor = 0
        for stage_index in range(4):
            blocks = [
                LMSBlock(
                    dims[stage_index],
                    drop_path=rates[cursor + block_index],
                    layer_scale_init=layer_scale_init,
                )
                for block_index in range(depths[stage_index])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cursor += depths[stage_index]

        self.out_channels = list(dims)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = self.stem(x)
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return features


LMS_CONVNEXT_CONFIGS = {
    "pico": {"depths": (2, 2, 2, 2), "dims": (32, 64, 128, 256)},
    "femto": {"depths": (2, 2, 6, 2), "dims": (48, 96, 192, 384)},
    "tiny": {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)},
}


def build_lms_convnext_backbone(size: str = "femto", **overrides) -> LMSConvNeXtBackbone:
    if size not in LMS_CONVNEXT_CONFIGS:
        raise ValueError(f"Unknown backbone size: {size}")
    config = {**LMS_CONVNEXT_CONFIGS[size], **overrides}
    return LMSConvNeXtBackbone(**config)

