
import numpy as np

import torch
from torch import nn
from torch.nn import Parameter, Module

    
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

    def forward(
            self, 
            x: torch.Tensor,
            trainable: bool = True,
        ) -> torch.Tensor:
        x = 2 * torch.pi * self.frequencies * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x.flatten(1)
    
    def regularization_loss(self):
        return 0
    
    def clamp_weights(self):
        pass


class BinningFeatures(Module):
    def __init__(
            self, 
            n_features: int, 
            n_bins: int, 
            delta_scale: float = 1.0,
            delta_min: float = 30.0,
            delta_max: float = 100.0,
        ) -> None:
        super().__init__()
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta = Parameter(
            torch.rand(n_features, n_bins) * delta_scale  # Learnable delta for each feature
        )
        self.register_buffer("u", torch.rand(n_features, n_bins))
        self.factor = Parameter(
            torch.Tensor([np.sqrt(1.0 / n_bins)])
        )
        self.d_out = n_features * n_bins

    def forward(
            self, 
            x: torch.Tensor,
            trainable: bool = True
        ) -> torch.Tensor:
        """
        Compute the randomized feature map z(x) for input x.
        """
        # Scale u
        scaled_u = self.u * self.delta

        # Compute bin indices
        bin_indices = torch.ceil((x.unsqueeze(-1) - scaled_u[None]) / self.delta[None])
        
        # One-hot encode if not trainable
        if not trainable:
            bin_indices = torch.clamp(bin_indices.long(), min=0, max=9)
            bin_indices = torch.nn.functional.one_hot(bin_indices, num_classes=10) 
        
        return bin_indices.flatten(1)
    
    def regularization_loss(self):
        return -0.01 * torch.norm(self.delta, p=1)
    
    def clamp_weights(self):
        with torch.no_grad():
            self.delta.clamp_(min=self.delta_min, max=self.delta_max)
    