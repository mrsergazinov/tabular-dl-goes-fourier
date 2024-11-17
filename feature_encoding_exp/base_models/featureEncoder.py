

import torch
from torch.nn import Parameter, Module

class FourierFeatures(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )
        self.lite = lite
        if lite:
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.weight = Parameter(torch.empty(n_features, 2 * n_frequencies, d_embedding))
            self.bias = Parameter(torch.empty(n_features, d_embedding))
            self.reset_parameters()
        self.activation = nn.ReLU()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for i in range(self.weight.shape[0]):
                layer = nn.Linear(self.weight.shape[1], self.weight.shape[2])
                self.weight[i] = layer.weight.T
                self.bias[i] = layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        if self.lite:
            x = self.linear(x)
        else:
            x = x[..., None] * self.weight[None]
            x = x.sum(-2)
            x = x + self.bias[None]
        x = self.activation(x)
        return x