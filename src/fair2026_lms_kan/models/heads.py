"""Classifier heads used in the controlled head ablation."""

import torch
import torch.nn as nn


class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2.0, grid_max: float = 2.0, num_grids: int = 8):
        super().__init__()
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids), persistent=False)
        self.denominator = (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class FastKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_grids: int = 8):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(num_grids=num_grids)
        self.spline_linear = nn.Linear(input_dim * num_grids, output_dim, bias=False)
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.base_activation = nn.SiLU()
        nn.init.trunc_normal_(self.spline_linear.weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.rbf(self.layernorm(x)).reshape(x.shape[0], -1)
        return self.spline_linear(basis) + self.base_linear(self.base_activation(x))


class FastKANHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2, grid_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.kan1 = FastKANLayer(in_dim, hidden_dim, num_grids=grid_size)
        self.dropout = nn.Dropout(dropout)
        self.kan2 = FastKANLayer(hidden_dim, num_classes, num_grids=grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan2(self.dropout(self.kan1(x)))


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
