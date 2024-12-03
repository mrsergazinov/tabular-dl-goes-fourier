
import numpy as np

import torch
from torch import nn

    
class FourierFeatures(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.normal(0.0, frequency_scale, (n_features, n_frequencies)))
        self.d_out = n_features * 2 * n_frequencies

    def forward(
            self, 
            x: torch.Tensor,
        ) -> torch.Tensor:
        x = 2 * torch.pi * self.frequencies * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x.flatten(1)
    
    def regularization_loss(self):
        return 0
    
    def clamp_weights(self):
        pass


class BinningFeatures(nn.Module):
    def __init__(
            self, 
            n_features: int, 
            n_bins: int, 
            delta_scale: float = 1.0,
            delta_min: float = 30.0,
            delta_max: float = 100.0,
            trainable: bool = True,
        ) -> None:
        super().__init__()
        self.trainable = trainable
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta = nn.Parameter(
            torch.rand(n_features, n_bins) * delta_scale  # Learnable delta for each feature
        )
        self.register_buffer("u", torch.rand(n_features, n_bins))
        self.factor = nn.Parameter(
            torch.Tensor([np.sqrt(1.0 / n_bins)])
        )
        self.d_out = n_features * n_bins
        if not self.trainable:
            self.d_out = n_features * n_bins * 10

    def forward(
            self, 
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute the randomized feature map z(x) for input x.
        """
        # Scale u
        scaled_u = self.u * self.delta

        # Compute bin indices
        bin_indices = torch.ceil((x.unsqueeze(-1) - scaled_u[None]) / self.delta[None])
        
        # One-hot encode if not trainable
        if not self.trainable:
            bin_indices = torch.clamp(bin_indices.long(), min=0, max=9)
            bin_indices = torch.nn.functional.one_hot(bin_indices, num_classes=10) 
        
        return bin_indices.flatten(1)
    
    def regularization_loss(self):
        return -0.01 * torch.norm(self.delta, p=1)
    
    def clamp_weights(self):
        with torch.no_grad():
            self.delta.clamp_(min=self.delta_min, max=self.delta_max)

class ComboFeatures(nn.Module):
    def __init__(
            self, 
            n_features: int, 
            n_bins: int, 
            n_frequencies: int,
            frequency_scale: float,
            delta_scale: float = 1.0,
            delta_min: float = 30.0,
            delta_max: float = 100.0,
            trainable: bool = True,
        ) -> None:
        super().__init__()
        self.binning = BinningFeatures(
            n_features=n_features,
            n_bins=n_bins,
            delta_scale=delta_scale,
            delta_min=delta_min,
            delta_max=delta_max,
            trainable=trainable,
        )
        self.fourier = FourierFeatures(
            n_features=n_features,
            n_frequencies=n_frequencies,
            frequency_scale=frequency_scale,
            trainable=trainable,
        )
        self.d_out = self.binning.d_out + self.fourier.d_out

    def forward(
            self, 
            x: torch.Tensor,
        ) -> torch.Tensor:
        return torch.cat([self.binning(x), self.fourier(x)], -1)
    
    def regularization_loss(self):
        return self.binning.regularization_loss() + self.fourier.regularization_loss()
    
    def clamp_weights(self):
        self.binning.clamp_weights()
        self.fourier.clamp_weights()

class SquareScalingFeatures(nn.Module):
    def __init__(
            self, 
            n_features: int, 
        ) -> None:
        super().__init__()
        self.scales = nn.Parameter(
            torch.rand(n_features)
        )
        self.d_out = n_features

    def forward(
            self, 
            x: torch.Tensor,
            trainable: bool = True
        ) -> torch.Tensor:
        """
        Compute the randomized feature map z(x) for input x.
        """
        return x * self.scales ** 2
    
    def regularization_loss(self):
        return 0
    
    def clamp_weights(self):
        pass
    