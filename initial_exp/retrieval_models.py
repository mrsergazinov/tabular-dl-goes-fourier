import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional

class PLREmbeddings(nn.Module):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.
    
    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
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

class ModernNCA(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        dim: int,
        dropout: float,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
        temperature: float = 1.0,
        sample_rate: float = 0.8
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dim = dim
        self.dropout = dropout
        self.temperature = temperature
        self.sample_rate = sample_rate

        # Initialize PLREmbeddings directly
        self.num_embeddings = PLREmbeddings(
            n_features=d_in,
            n_frequencies=n_frequencies,
            frequency_scale=frequency_scale,
            d_embedding=d_embedding,
            lite=lite
        )

        # Define the encoder layer
        self.encoder = nn.Linear(d_in * d_embedding, dim)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        candidate_x: torch.Tensor,
        candidate_y: torch.Tensor,
        is_train: bool
    ) -> torch.Tensor:
        if is_train:
            data_size = candidate_x.shape[0]
            retrieval_size = int(data_size * self.sample_rate)
            sample_idx = torch.randperm(data_size)[:retrieval_size]
            candidate_x = candidate_x[sample_idx]
            candidate_y = candidate_y[sample_idx]

        # Apply PLR embeddings to numerical features
        x = self.num_embeddings(x).flatten(1)          # Shape: [batch_size, d_in * d_embedding]
        candidate_x = self.num_embeddings(candidate_x).flatten(1)  # Shape: [num_candidates, d_in * d_embedding]

        x = self.encoder(x)        # Shape: [batch_size, dim]
        candidate_x = self.encoder(candidate_x) # Shape: [num_candidates, dim]

        # if is_train:
        #     assert y is not None, "Labels `y` must be provided during training."
        #     candidate_x = torch.cat([x, candidate_x])
        #     candidate_y = torch.cat([y, candidate_y])
        # else:
        #     assert y is None, "Labels `y` should be None during evaluation."

        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).double()
        elif candidate_y.dim() == 1:
            candidate_y = candidate_y.unsqueeze(-1).double()

        logits, logsumexp = 0, 0
        for idx in range(0, candidate_y.shape[0], 5000):
            batch_candidate_y = candidate_y[idx:idx+5000]
            batch_candidate_x = candidate_x[idx:idx+5000]
            distances = torch.cdist(x, batch_candidate_x, p=2) / self.temperature
            exp_distances = torch.exp(-distances)
            logsumexp += torch.logsumexp(exp_distances, dim=1)
            logits += torch.mm(exp_distances, batch_candidate_y)
        logits = torch.log(logits) - logsumexp.unsqueeze(1)
        
        return logits
