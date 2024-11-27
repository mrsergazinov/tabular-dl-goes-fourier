

import torch
from torch import nn
from torch.nn import Parameter, Module

class FourierFeaturesScaled(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
    ) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.normal(0.0, frequency_scale, (n_features, n_frequencies)))
        self.scale = Parameter(torch.ones((n_features, 2 * n_frequencies)))
        self.d_out = n_features * 2 * n_frequencies
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * torch.pi * self.frequencies * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        x = x * self.scale
        return x
    
class FourierFeatures(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(torch.normal(0.0, frequency_scale, (n_features, n_frequencies)))
        self.d_out = n_features * 2 * n_frequencies
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * torch.pi * self.frequencies * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x